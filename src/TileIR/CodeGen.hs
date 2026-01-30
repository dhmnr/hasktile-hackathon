{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

module TileIR.CodeGen
  ( generateMLIR
  , compileToBytecode
  ) where

import TileIR.Types
import Data.Proxy
import GHC.TypeLits (KnownNat, natVal)
import qualified Data.Text as T
import Data.Text (Text)
import Control.Monad.State
import qualified Data.Map as Map
import System.Process (readProcess)
import System.IO.Temp (withSystemTempFile)
import System.IO (hPutStr, hClose)

-- | Code generation state
data CodeGenState = CodeGenState
  { varCounter :: Int
  , generatedCode :: [Text]
  }

type CodeGen = State CodeGenState

-- | Generate fresh SSA value name
freshVar :: CodeGen Text
freshVar = do
  s <- get
  let n = varCounter s
  put s { varCounter = n + 1 }
  return $ T.pack $ "%v" ++ show n

-- | Emit a line of code
emit :: Text -> CodeGen ()
emit line = modify $ \s -> s { generatedCode = generatedCode s ++ [line] }

-- | Generate MLIR from a tile kernel
generateMLIR :: TileKernel -> Text
generateMLIR kernel = case kernel of
  TileKernel2 name f -> generateKernel2 name f
  _ -> error "Only 2-argument kernels supported for now"

-- | Generate MLIR for a 2-argument kernel by analyzing the AST
generateKernel2 :: forall n a b c. KnownNat n
                => String
                -> (Tile n a -> Tile n b -> Tile n c)
                -> Text
generateKernel2 name f =
  let tileSize = natVal (Proxy @n)
      -- Create input tiles and run the function to get the AST
      inputA = Tile (Proxy @n) (TileVar "arg0") :: Tile n a
      inputB = Tile (Proxy @n) (TileVar "arg1") :: Tile n b
      result = f inputA inputB
      resultExpr = tileExpr result

      -- Generate code from the AST
      (outputVar, finalState) = runState (genTileExpr tileSize resultExpr)
                                         (CodeGenState 0 [])
      bodyLines = generatedCode finalState

  in T.unlines $
    [ "cuda_tile.module @kernels {"
    , "  entry @" <> T.pack name <> "("
    , "    %ptr_a: !cuda_tile.tile<ptr<f32>>,"
    , "    %ptr_b: !cuda_tile.tile<ptr<f32>>,"
    , "    %ptr_out: !cuda_tile.tile<ptr<f32>>"
    , "  ) {"
    ] ++ map ("    " <>) bodyLines ++
    [ ""
    , "    // Store result"
    , "    %tok_out = store_ptr_tko weak %out_ptrs, " <> outputVar <> " : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32> -> !cuda_tile.token"
    , ""
    , "    return"
    , "  }"
    , "}"
    ]

-- | Generate code for a tile expression
genTileExpr :: Integer -> TileExpr a -> CodeGen Text
genTileExpr tileSize expr = case expr of
  -- Variable reference - need to load from pointer
  TileVar "arg0" -> do
    emit "// Setup pointer A"
    emit $ "%offset = iota : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
    emit ""
    emit "// Prepare pointer A"
    emit "%a_base = reshape %ptr_a : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>"
    emit $ "%a_bcast = broadcast %a_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit $ "%a_ptrs = offset %a_bcast, %offset : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit ""
    emit "// Load A"
    emit $ "%a_val, %tok_a = load_ptr_tko weak %a_ptrs : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>, !cuda_tile.token"
    return "%a_val"

  TileVar "arg1" -> do
    emit "// Prepare pointer B"
    emit "%b_base = reshape %ptr_b : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>"
    emit $ "%b_bcast = broadcast %b_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit $ "%b_ptrs = offset %b_bcast, %offset : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit ""
    emit "// Load B"
    emit $ "%b_val, %tok_b = load_ptr_tko weak %b_ptrs : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>, !cuda_tile.token"
    return "%b_val"

  TileVar name -> return $ T.pack $ "%" ++ name

  -- ZipWith - analyze the scalar function to determine operation
  TileZipWith scalarFn exprA exprB -> do
    varA <- genTileExpr tileSize exprA
    varB <- genTileExpr tileSize exprB

    -- Setup output pointer (reuse offset)
    emit ""
    emit "// Prepare output pointer"
    emit "%out_base = reshape %ptr_out : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>"
    emit $ "%out_bcast = broadcast %out_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit $ "%out_ptrs = offset %out_bcast, %offset : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit ""

    -- Generate operation based on scalar function
    resultVar <- freshVar
    let op = analyzeScalarOp scalarFn
    emit $ "// Compute: " <> op
    emit $ resultVar <> " = " <> op <> " " <> varA <> ", " <> varB <> " rounding<nearest_even> : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"
    return resultVar

  _ -> error $ "Unsupported tile expression: " ++ show expr

-- | Analyze scalar function to determine the operation
analyzeScalarOp :: (ScalarExpr a -> ScalarExpr b -> ScalarExpr c) -> Text
analyzeScalarOp f =
  let testA = ScalarVar "test_a"
      testB = ScalarVar "test_b"
      result = f testA testB
  in case result of
    Add _ _ -> "addf"
    Mul _ _ -> "mulf"
    Sub _ _ -> "subf"
    _ -> error "Unsupported scalar operation"

-- | Compile MLIR to TileIR bytecode
compileToBytecode :: Text -> IO FilePath
compileToBytecode mlir = do
  withSystemTempFile "kernel.mlir" $ \mlirPath mlirHandle -> do
    hPutStr mlirHandle (T.unpack mlir)
    hClose mlirHandle

    let bcPath = mlirPath ++ ".tilebc"

    _ <- readProcess "/home/shadeform/cuda-tile/build/bin/cuda-tile-translate"
      [ mlirPath
      , "--mlir-to-cudatilebc"
      , "--no-implicit-module"
      , "--bytecode-version=13.1"
      , "-o", bcPath
      ] ""

    return bcPath
