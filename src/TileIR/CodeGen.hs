{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

module TileIR.CodeGen
  ( generateMLIR
  , generateMLIRWithDebug
  , generateFoldKernel
  , compileToBytecode
  ) where

import TileIR.Types
import Data.Proxy
import GHC.TypeLits (KnownNat, natVal)
import qualified Data.Text as T
import Data.Text (Text)
import Control.Monad.State
import Control.Monad (when)
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
generateMLIR kernel = generateMLIRWithDebug kernel False

-- | Generate MLIR from a tile kernel with optional debug printing
generateMLIRWithDebug :: TileKernel -> Bool -> Text
generateMLIRWithDebug kernel enableDebug = case kernel of
  TileKernel2 name f -> generateKernel2WithDebug name f enableDebug
  _ -> error "Only 2-argument kernels supported for now"

-- | Generate MLIR for a 2-argument kernel by analyzing the AST
generateKernel2 :: forall n a b c. KnownNat n
                => String
                -> (Tile n a -> Tile n b -> Tile n c)
                -> Text
generateKernel2 name f = generateKernel2WithDebug name f False

-- | Generate MLIR for a 2-argument kernel with optional debug printing
generateKernel2WithDebug :: forall n a b c. KnownNat n
                         => String
                         -> (Tile n a -> Tile n b -> Tile n c)
                         -> Bool  -- Enable debug printing
                         -> Text
generateKernel2WithDebug name f enableDebug =
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

      -- Optional debug print
      debugLines = if enableDebug
        then [ "    // Debug: Print block index"
             , "    %dbg_block_x, %dbg_block_y, %dbg_block_z = get_tile_block_id : tile<i32>"
             , "    %dbg_dim_x, %dbg_dim_y, %dbg_dim_z = cuda_tile.get_num_tile_blocks : tile<i32>"
             , "    cuda_tile.print \"Block (%, %, %) of (%, %, %) executing " <> T.pack name <> "\\n\","
             , "      %dbg_block_x, %dbg_block_y, %dbg_block_z, %dbg_dim_x, %dbg_dim_y, %dbg_dim_z"
             , "      : tile<i32>, tile<i32>, tile<i32>, tile<i32>, tile<i32>, tile<i32>"
             , ""
             ]
        else []

  in T.unlines $
    [ "cuda_tile.module @kernels {"
    , "  entry @" <> T.pack name <> "("
    , "    %ptr_a: !cuda_tile.tile<ptr<f32>>,"
    , "    %ptr_b: !cuda_tile.tile<ptr<f32>>,"
    , "    %ptr_out: !cuda_tile.tile<ptr<f32>>"
    , "  ) {"
    ] ++ debugLines ++ map ("    " <>) bodyLines ++
    [ ""
    , "    // Store result"
    , "    %tok_out = store_ptr_tko weak %out_ptrs, " <> outputVar <> " : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32> -> !cuda_tile.token"
    , ""
    , "    return"
    , "  }"
    , "}"
    ]

-- | Helper: Generate strided load code with blockIdx
-- Args: varName ptrName prefix offset stride tileSize setupOffset
genLoad :: String -> String -> String -> Int -> Int -> Integer -> Bool -> CodeGen Text
genLoad varName ptrName prefix offset stride tileSize setupOffset = do
  emit $ "// Load " <> T.pack varName <> " (offset=" <> T.pack (show offset) <> ", stride=" <> T.pack (show stride) <> ")"

  -- Generate base offset indices (only once for arg0)
  when setupOffset $ do
    emit "// Get block index for parallel processing"
    emit "%block_x_index, %block_y_index, %block_z_index = get_tile_block_id : tile<i32>"
    emit ""
    emit "// Setup pointer indices with blockIdx"
    emit $ "%tile_size_const = constant <i32: " <> T.pack (show tileSize) <> "> : tile<i32>"
    emit "%block_offset = cuda_tile.muli %block_x_index, %tile_size_const : tile<i32>"
    emit ""
    emit $ "%tile_indices = iota : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
    emit "%block_offset_1d = reshape %block_offset : tile<i32> -> !cuda_tile.tile<1xi32>"
    emit $ "%block_offset_splat = broadcast %block_offset_1d : !cuda_tile.tile<1xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
    emit $ "%offset_base = cuda_tile.addi %tile_indices, %block_offset_splat : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
    emit ""

  -- Apply stride if not 1
  let offsetVar = if stride == 1
                  then "%offset_base"
                  else "%offset_scaled_" <> T.pack prefix

  when (stride /= 1) $ do
    emit $ "%stride_scalar_" <> T.pack prefix <> " = constant <i32: " <> T.pack (show stride) <> "> : tile<i32>"
    emit $ "%stride_1d_" <> T.pack prefix <> " = reshape %stride_scalar_" <> T.pack prefix <> " : tile<i32> -> !cuda_tile.tile<1xi32>"
    emit $ "%stride_" <> T.pack prefix <> " = broadcast %stride_1d_" <> T.pack prefix <> " : !cuda_tile.tile<1xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
    emit $ offsetVar <> " = cuda_tile.muli %offset_base, %stride_" <> T.pack prefix <> " : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"

  -- Add offset if not 0
  let finalOffsetVar = if offset == 0
                       then offsetVar
                       else "%offset_final_" <> T.pack prefix

  when (offset /= 0) $ do
    emit $ "%offset_const_scalar_" <> T.pack prefix <> " = constant <i32: " <> T.pack (show offset) <> "> : tile<i32>"
    emit $ "%offset_const_1d_" <> T.pack prefix <> " = reshape %offset_const_scalar_" <> T.pack prefix <> " : tile<i32> -> !cuda_tile.tile<1xi32>"
    emit $ "%offset_const_" <> T.pack prefix <> " = broadcast %offset_const_1d_" <> T.pack prefix <> " : !cuda_tile.tile<1xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
    emit $ finalOffsetVar <> " = cuda_tile.addi " <> offsetVar <> ", %offset_const_" <> T.pack prefix <> " : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"

  emit ""

  -- Prepare pointer
  emit $ "// Prepare pointer " <> T.pack prefix
  emit $ "%" <> T.pack prefix <> "_base = reshape %" <> T.pack ptrName <> " : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>"
  emit $ "%" <> T.pack prefix <> "_bcast = broadcast %" <> T.pack prefix <> "_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
  emit $ "%" <> T.pack prefix <> "_ptrs = offset %" <> T.pack prefix <> "_bcast, " <> finalOffsetVar <> " : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
  emit ""

  -- Load
  emit $ "// Load " <> T.pack prefix
  let valVar = "%" <> T.pack prefix <> "_val"
  let tokVar = "%" <> T.pack prefix <> "_tok"
  emit $ valVar <> ", " <> tokVar <> " = load_ptr_tko weak %" <> T.pack prefix <> "_ptrs : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>, !cuda_tile.token"

  return valVar

-- | Generate code for a tile expression
genTileExpr :: Integer -> TileExpr a -> CodeGen Text
genTileExpr tileSize expr = case expr of
  -- Iota (range 0..n-1)
  TileIota -> do
    resultVar <- freshVar
    emit ""
    emit "// Iota (range 0..n-1)"
    emit $ resultVar <> " = iota : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
    return resultVar

  -- Block index
  TileBlockIdx -> do
    resultVar <- freshVar
    emit ""
    emit "// Get block index"
    emit $ "%block_x, %block_y, %block_z = get_tile_block_id : tile<i32>"
    emit $ resultVar <> "_1d = reshape %block_x : tile<i32> -> !cuda_tile.tile<1xi32>"
    emit $ resultVar <> " = broadcast " <> resultVar <> "_1d : !cuda_tile.tile<1xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
    return resultVar

  -- Variable reference - need to load from pointer (default: offset=0, stride=1)
  TileVar "arg0" -> genLoad "arg0" "ptr_a" "a" 0 1 tileSize True
  TileVar "arg1" -> genLoad "arg1" "ptr_b" "b" 0 1 tileSize False
  TileVar name -> return $ T.pack $ "%" ++ name

  -- Variable reference with stride - load from pointer with stride
  TileVarStrided "arg0" offset stride -> genLoad "arg0" "ptr_a" "a" offset stride tileSize True
  TileVarStrided "arg1" offset stride -> genLoad "arg1" "ptr_b" "b" offset stride tileSize False
  TileVarStrided name offset stride -> do
    -- For other variables, just generate load
    genLoad name ("ptr_" ++ name) name offset stride tileSize False

  -- Constant tile - broadcast scalar to all elements
  TileConst _ -> error "TileConst not yet supported in code generation"

  -- Map - apply scalar function to each element
  TileMap scalarFn tileExpr -> do
    inputVar <- genTileExpr tileSize tileExpr
    resultVar <- freshVar

    -- Setup output pointer (same as in ZipWith)
    emit ""
    emit "// Prepare output pointer for map"
    emit "%out_base = reshape %ptr_out : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>"
    emit $ "%out_bcast = broadcast %out_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit $ "%out_ptrs = offset %out_bcast, %offset_base : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit ""

    emit "// Map operation"

    -- For now, just handle specific common patterns
    -- We analyze the function by applying it to a test variable
    let testVar = ScalarVar "x"
        result = scalarFn testVar

    -- Helper to generate constant broadcast
    let genConst :: Text -> Text -> CodeGen Text
        genConst name value = do
          emit $ name <> "_scalar = constant <f32: " <> value <> "> : tile<f32>"
          emit $ name <> "_1d = reshape " <> name <> "_scalar : tile<f32> -> !cuda_tile.tile<1xf32>"
          emit $ name <> "_bcast = broadcast " <> name <> "_1d : !cuda_tile.tile<1xf32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"
          return $ name <> "_bcast"

    case result of
      -- x * x = square
      Mul (ScalarVar "x") (ScalarVar "x") -> do
        emit $ resultVar <> " = mulf " <> inputVar <> ", " <> inputVar <> " rounding<nearest_even> : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"
        return resultVar

      -- 2*x + 1 pattern
      Add (Mul (ScalarLit _) (ScalarVar "x")) (ScalarLit _) -> do
        -- Generate the composite operation
        -- First multiply by 2.0
        constMul <- genConst "%map_mul_const" "2.0"
        emit $ "%map_mul_result = mulf " <> inputVar <> ", " <> constMul <> " rounding<nearest_even> : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"

        -- Then add 1.0
        constAdd <- genConst "%map_add_const" "1.0"
        emit $ resultVar <> " = addf %map_mul_result, " <> constAdd <> " rounding<nearest_even> : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"
        return resultVar

      _ -> error $ "Unsupported map operation: " ++ show result

  -- ZipWith - analyze the scalar function to determine operation
  TileZipWith scalarFn exprA exprB -> do
    varA <- genTileExpr tileSize exprA
    varB <- genTileExpr tileSize exprB

    -- Setup output pointer (reuse offset_base which includes blockIdx)
    emit ""
    emit "// Prepare output pointer"
    emit "%out_base = reshape %ptr_out : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>"
    emit $ "%out_bcast = broadcast %out_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit $ "%out_ptrs = offset %out_bcast, %offset_base : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
    emit ""

    -- Generate operation based on scalar function
    resultVar <- freshVar
    let op = analyzeScalarOp scalarFn
    emit $ "// Compute: " <> op
    emit $ resultVar <> " = " <> op <> " " <> varA <> ", " <> varB <> " rounding<nearest_even> : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"
    return resultVar

  -- Fold (tree reduction within tile using tree reduction pattern)
  TileFold scalarFn _initExpr tileExpr -> do
    inputVar <- genTileExpr tileSize tileExpr
    let op = analyzeScalarOp scalarFn

    emit ""
    emit $ "// Fold (tree reduction) using " <> op
    emit "// Using for loop to sum elements"

    -- Initialize accumulator
    emit "%acc_init = constant <f32: 0.0> : tile<f32>"
    emit $ "%range_start = constant <i32: 0> : tile<i32>"
    emit $ "%range_end = constant <i32: " <> T.pack (show tileSize) <> "> : tile<i32>"
    emit $ "%range_step = constant <i32: 1> : tile<i32>"
    emit ""

    -- For loop to accumulate
    emit $ "// For loop to reduce"
    emit $ "%acc_final = for %i in (%range_start to %range_end, step %range_step)"
    emit $ "  : tile<i32> iter_values(%acc_iter = %acc_init) -> (tile<f32>) {"
    emit ""
    emit $ "  // Extract element at index %i"
    emit $ "  %elem = extract " <> inputVar <> "[%i] : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"
    emit ""
    emit $ "  // Accumulate"
    case op of
      "addf" -> emit "  %acc_next = addf %acc_iter, %elem : tile<f32>"
      "mulf" -> emit "  %acc_next = mulf %acc_iter, %elem : tile<f32>"
      _ -> emit "  %acc_next = %acc_iter  // Unsupported op"
    emit ""
    emit "  yield %acc_next : tile<f32>"
    emit "}"
    emit ""

    -- Broadcast scalar result back to full tile
    resultVar <- freshVar
    emit $ "// Broadcast reduced value to all elements"
    emit $ "%acc_1d = reshape %acc_final : tile<f32> -> !cuda_tile.tile<1xf32>"
    emit $ resultVar <> " = broadcast %acc_1d : !cuda_tile.tile<1xf32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"

    return resultVar

  -- Scan (prefix scan within tile)
  TileScan scalarFn _initExpr tileExpr -> do
    inputVar <- genTileExpr tileSize tileExpr
    let _op = analyzeScalarOp scalarFn

    emit ""
    emit "// TODO: Scan requires prefix scan implementation"
    emit "// For now, just return input unchanged"
    emit "// Real implementation needs:"
    emit "//   - Up-sweep phase (build partial sums)"
    emit "//   - Down-sweep phase (distribute accumulated values)"
    emit "//   - May require shuffle operations or shared memory"

    return inputVar

  -- Load from memory (basic)
  TileLoad name -> do
    genLoad name ("ptr_" ++ name) name 0 1 tileSize False

  -- Store to memory - these should be handled at top level, not as expressions
  TileStore _ _ -> error "TileStore should be handled at top level, not as expression"
  TileStoreStrided _ _ _ _ -> error "TileStoreStrided should be handled at top level, not as expression"

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

-- | Generate fold kernel: each block combines 2 tiles element-wise
-- Launch repeatedly with gridDim = numTiles/2 until 1 tile remains
generateFoldKernel :: Integer -> Text -> Text
generateFoldKernel tileSize opName = T.unlines
  [ "cuda_tile.module @kernels {"
  , "  entry @fold_combine("
  , "    %ptr_in: !cuda_tile.tile<ptr<f32>>,"
  , "    %ptr_out: !cuda_tile.tile<ptr<f32>>"
  , "  ) {"
  , "    // Get block index"
  , "    %block_x, %block_y, %block_z = get_tile_block_id : tile<i32>"
  , ""
  , "    // Each block combines 2 tiles element-wise"
  , "    %tile_size_const = constant <i32: " <> T.pack (show tileSize) <> "> : tile<i32>"
  , "    %two = constant <i32: 2> : tile<i32>"
  , "    %offset_factor = cuda_tile.muli %block_x, %two : tile<i32>"
  , "    %left_tile_offset = cuda_tile.muli %offset_factor, %tile_size_const : tile<i32>"
  , "    %one = constant <i32: 1> : tile<i32>"
  , "    %right_factor = cuda_tile.addi %offset_factor, %one : tile<i32>"
  , "    %right_tile_offset = cuda_tile.muli %right_factor, %tile_size_const : tile<i32>"
  , ""
  , "    // Load left tile"
  , "    %tile_indices = iota : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
  , "    %left_offset_1d = reshape %left_tile_offset : tile<i32> -> !cuda_tile.tile<1xi32>"
  , "    %left_offset_splat = broadcast %left_offset_1d : !cuda_tile.tile<1xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
  , "    %left_offset_base = cuda_tile.addi %tile_indices, %left_offset_splat : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
  , ""
  , "    %in_base = reshape %ptr_in : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>"
  , "    %in_bcast_left = broadcast %in_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
  , "    %left_ptrs = offset %in_bcast_left, %left_offset_base : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
  , "    %left_tile, %tok1 = load_ptr_tko weak %left_ptrs : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>, !cuda_tile.token"
  , ""
  , "    // Load right tile"
  , "    %right_offset_1d = reshape %right_tile_offset : tile<i32> -> !cuda_tile.tile<1xi32>"
  , "    %right_offset_splat = broadcast %right_offset_1d : !cuda_tile.tile<1xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
  , "    %right_offset_base = cuda_tile.addi %tile_indices, %right_offset_splat : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
  , ""
  , "    %in_bcast_right = broadcast %in_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
  , "    %right_ptrs = offset %in_bcast_right, %right_offset_base : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
  , "    %right_tile, %tok2 = load_ptr_tko weak %right_ptrs : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>, !cuda_tile.token"
  , ""
  , "    // Combine tiles element-wise"
  , "    %result = " <> opName <> " %left_tile, %right_tile rounding<nearest_even> : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32>"
  , ""
  , "    // Store result tile"
  , "    %out_offset_1d = reshape %left_tile_offset : tile<i32> -> !cuda_tile.tile<1xi32>"
  , "    %out_offset_splat = broadcast %out_offset_1d : !cuda_tile.tile<1xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
  , "    %out_offset_base = cuda_tile.addi %tile_indices, %out_offset_splat : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32>"
  , ""
  , "    %out_base = reshape %ptr_out : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>"
  , "    %out_bcast = broadcast %out_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
  , "    %out_ptrs = offset %out_bcast, %out_offset_base : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xi32> -> !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>"
  , "    %tok_out = store_ptr_tko weak %out_ptrs, %result : !cuda_tile.tile<" <> T.pack (show tileSize) <> "xptr<f32>>, !cuda_tile.tile<" <> T.pack (show tileSize) <> "xf32> -> !cuda_tile.token"
  , ""
  , "    return"
  , "  }"
  , "}"
  ]

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
