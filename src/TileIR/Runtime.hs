{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module TileIR.Runtime
  ( TiledKernel(..)
  , Array(..)
  , loadArray
  ) where

import TileIR.Types
import TileIR.DSL
import TileIR.CodeGen
import TileIR.CUDADriver
import GHC.TypeLits (KnownNat, natVal)
import Data.Proxy
import Foreign.Ptr
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc (alloca)
import Foreign.Storable
import Foreign.StablePtr
import Data.Word (Word64)
import System.Process (callProcess)
import System.IO.Temp (withSystemTempDirectory)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.FilePath ((</>))

-- | Array type (host-side)
newtype Array a = Array [a]
  deriving (Show)

-- | Load array from list
loadArray :: [a] -> IO (Array a)
loadArray xs = return $ Array xs

-- | Typeclass for kernels with different numbers of inputs
class TiledKernel f where
  type InputArrays f :: *
  type OutputArray f :: *
  runTiled :: f -> InputArrays f -> IO (OutputArray f)

-- | Instance for 1-input kernels: Tile n a -> Tile n c
instance (KnownNat n, Storable a, Num a, Storable c, Num c)
  => TiledKernel (Tile n a -> Tile n c) where
  type InputArrays (Tile n a -> Tile n c) = Array a
  type OutputArray (Tile n a -> Tile n c) = Array c
  runTiled f arrA = runTiled1 False f arrA

-- | Instance for 2-input kernels: Tile n a -> Tile n b -> Tile n c
instance (KnownNat n, Storable a, Storable b, Storable c, Num c)
  => TiledKernel (Tile n a -> Tile n b -> Tile n c) where
  type InputArrays (Tile n a -> Tile n b -> Tile n c) = (Array a, Array b)
  type OutputArray (Tile n a -> Tile n b -> Tile n c) = Array c
  runTiled f (arrA, arrB) = runTiled2 False f arrA arrB

-- | Instance for 3-input kernels: Tile n a -> Tile n b -> Tile n c -> Tile n d
instance (KnownNat n, Storable a, Storable b, Storable c, Storable d, Num d)
  => TiledKernel (Tile n a -> Tile n b -> Tile n c -> Tile n d) where
  type InputArrays (Tile n a -> Tile n b -> Tile n c -> Tile n d) = (Array a, Array b, Array c)
  type OutputArray (Tile n a -> Tile n b -> Tile n c -> Tile n d) = Array d
  runTiled f (arrA, arrB, arrC) = runTiled3 False f arrA arrB arrC

-- ============================================================================
-- Internal implementations for different arities
-- ============================================================================

-- | 1-input kernel implementation
runTiled1 :: forall n a c. (KnownNat n, Storable a, Num a, Storable c, Num c)
          => Bool  -- Enable debug
          -> (Tile n a -> Tile n c)
          -> Array a
          -> IO (Array c)
runTiled1 enableDebug f (Array listA) =
  -- Wrap in 2-input form by ignoring second argument
  let f2 = \tileA (_ :: Tile n a) -> f tileA
      -- Create dummy array with same size (filled with zeros, ignored anyway)
      dummyArray = Array (replicate (length listA) (fromInteger 0 :: a))
  in runTiled2 enableDebug f2 (Array listA) dummyArray

-- | 2-input kernel implementation (original runTiled)
runTiled2 :: forall n a b c. (KnownNat n, Storable a, Storable b, Storable c, Num c)
          => Bool  -- Enable debug
          -> (Tile n a -> Tile n b -> Tile n c)
          -> Array a
          -> Array b
          -> IO (Array c)
runTiled2 = runTiledWithDebug

-- | 3-input kernel implementation (TODO: needs kernel3 codegen)
runTiled3 :: forall n a b c d. (KnownNat n, Storable a, Storable b, Storable c, Storable d, Num d)
          => Bool  -- Enable debug
          -> (Tile n a -> Tile n b -> Tile n c -> Tile n d)
          -> Array a
          -> Array b
          -> Array c
          -> IO (Array d)
runTiled3 _enableDebug _f _arrA _arrB _arrC =
  error "3-input kernels not yet implemented - need to add kernel3 to CodeGen"

-- | Run a tiled kernel on arrays with optional debug printing
runTiledWithDebug :: forall n a b c. (KnownNat n, Storable a, Storable b, Storable c, Num c)
                  => Bool  -- Enable debug printing
                  -> (Tile n a -> Tile n b -> Tile n c)
                  -> Array a
                  -> Array b
                  -> IO (Array c)
runTiledWithDebug enableDebug f (Array listA) (Array listB) = do
  -- Create test tiles to analyze AST
  let tileA = Tile (Proxy @n) (TileVar "arg0") :: Tile n a
  let tileB = Tile (Proxy @n) (TileVar "arg1") :: Tile n b
  let result = f tileA tileB
  let resultExpr = tileExpr result

  -- Check if kernel contains fold operation
  if containsFold resultExpr
    then do
      putStrLn "=== Detected FOLD operation - using multi-phase execution ==="
      runTiledFold enableDebug f (Array listA) (Array listB)
    else do
      putStrLn "=== Regular kernel execution ==="
      runTiledNormal enableDebug f (Array listA) (Array listB)

-- | Normal single-phase execution
runTiledNormal :: forall n a b c. (KnownNat n, Storable a, Storable b, Storable c, Num c)
               => Bool
               -> (Tile n a -> Tile n b -> Tile n c)
               -> Array a
               -> Array b
               -> IO (Array c)
runTiledNormal enableDebug f (Array listA) (Array listB) = do
  -- Generate MLIR
  let kernel = kernel2 @n "vector_add" f
  let mlir = generateMLIRWithDebug kernel enableDebug

  putStrLn "=== Generated MLIR ==="
  TIO.putStrLn mlir

  withSystemTempDirectory "hasktile" $ \tmpDir -> do
    -- Write MLIR to file
    let mlirPath = tmpDir </> "kernel.mlir"
    let tileBcPath = tmpDir </> "kernel.tilebc"
    let cubinPath = tmpDir </> "kernel.cubin"

    TIO.writeFile mlirPath mlir

    -- Compile MLIR -> TileBC
    putStrLn "\n=== Compiling MLIR to bytecode ==="
    callProcess "/home/shadeform/cuda-tile/build/bin/cuda-tile-translate"
      [ mlirPath
      , "--mlir-to-cudatilebc"
      , "--no-implicit-module"
      , "--bytecode-version=13.1"
      , "-o", tileBcPath
      ]

    -- Compile TileBC -> CUBIN (AOT)
    putStrLn "\n=== Compiling bytecode to cubin ==="
    callProcess "/usr/local/cuda/bin/tileiras"
      [ "--gpu-name", "sm_120"
      , tileBcPath
      , "-o", cubinPath
      ]

    putStrLn "\n=== Kernel compiled successfully ==="
    putStrLn $ "CUBIN at: " ++ cubinPath

    -- Initialize CUDA and load kernel
    putStrLn "\n=== Launching kernel ==="
    initCUDA
    device <- getDevice 0
    _ <- createContext device
    modul <- loadModule cubinPath
    func <- getFunction modul "vector_add"

    -- Allocate and copy input data
    let sizeA = length listA * sizeOf (undefined :: a)
    let sizeB = length listB * sizeOf (undefined :: b)
    let resultSize = length listA
    let sizeOut = resultSize * sizeOf (undefined :: c)

    dPtrA <- allocDevice sizeA
    dPtrB <- allocDevice sizeB
    dPtrOut <- allocDevice sizeOut

    copyHtoD dPtrA listA
    copyHtoD dPtrB listB

    -- Prepare kernel arguments (array of pointers to device pointers)
    alloca $ \pA -> alloca $ \pB -> alloca $ \pOut -> do
      poke pA dPtrA
      poke pB dPtrB
      poke pOut dPtrOut

      let params = [castPtr pA, castPtr pB, castPtr pOut]

      -- Launch with single block (grid size = 1)
      launchKernel1D func 1 params

    -- Copy results back
    result <- copyDtoH dPtrOut resultSize

    -- Cleanup
    freeDevice dPtrA
    freeDevice dPtrB
    freeDevice dPtrOut

    putStrLn "=== Kernel execution complete ==="
    return $ Array result

-- | Multi-phase fold execution
runTiledFold :: forall n a b c. (KnownNat n, Storable a, Storable b, Storable c, Num c)
             => Bool
             -> (Tile n a -> Tile n b -> Tile n c)
             -> Array a
             -> Array b
             -> IO (Array c)
runTiledFold _enableDebug f (Array listA) (Array _listB) = do
  let tileSize = natVal (Proxy @n)
  let tileSizeInt = fromIntegral tileSize :: Int
  let numElements = length listA
  let numTiles = numElements `div` tileSizeInt

  putStrLn $ "=== Fold: " ++ show numElements ++ " elements, " ++ show numTiles ++ " tiles ==="

  -- Generate fold kernel (combines 2 tiles → 1 tile)
  let mlirFold = generateFoldKernel tileSize (T.pack "addf")

  putStrLn "\n=== Fold Kernel ==="
  TIO.putStrLn mlirFold

  withSystemTempDirectory "hasktile-fold" $ \tmpDir -> do
    -- Compile kernel
    let mlirPath = tmpDir </> "fold.mlir"
    let tileBcPath = tmpDir </> "fold.tilebc"
    let cubinPath = tmpDir </> "fold.cubin"

    TIO.writeFile mlirPath mlirFold

    putStrLn "\n=== Compiling ==="
    callProcess "/home/shadeform/cuda-tile/build/bin/cuda-tile-translate"
      [mlirPath, "--mlir-to-cudatilebc", "--no-implicit-module", "--bytecode-version=13.1", "-o", tileBcPath]
    callProcess "/usr/local/cuda/bin/tileiras"
      ["--gpu-name", "sm_120", tileBcPath, "-o", cubinPath]

    -- Initialize CUDA
    initCUDA
    device <- getDevice 0
    _ <- createContext device
    modul <- loadModule cubinPath
    func <- getFunction modul "fold_combine"

    -- Allocate buffers (ping-pong between input/output)
    let sizeA = numElements * sizeOf (undefined :: a)
    dPtrA <- allocDevice sizeA
    dPtrB <- allocDevice sizeA  -- Same size for intermediate results

    copyHtoD dPtrA listA

    -- Tree reduction: launch log2(numTiles) times
    finalResult <- treeReduce func tileSize tileSizeInt numTiles dPtrA dPtrB

    freeDevice dPtrA
    freeDevice dPtrB

    return $ Array finalResult

-- Helper: Tree reduction - launch kernel log2(N) times
treeReduce :: forall c. Storable c
           => CUfunction -> Integer -> Int -> Int
           -> CUdeviceptr -> CUdeviceptr -> IO [c]
treeReduce func tileSize tileSizeInt currentTiles ptrIn ptrOut = do
  if currentTiles == 1
    then do
      -- Done! Copy final tile back
      result <- copyDtoH ptrIn tileSizeInt :: IO [c]
      return result
    else do
      -- Launch kernel: gridDim = currentTiles/2
      let nextTiles = currentTiles `div` 2
      putStrLn $ "=== Reducing " ++ show currentTiles ++ " tiles → " ++ show nextTiles ++ " tiles ==="

      alloca $ \pIn -> alloca $ \pOut -> do
        poke pIn ptrIn
        poke pOut ptrOut
        let params = [castPtr pIn, castPtr pOut]
        launchKernel1D func nextTiles params

      -- Recurse with output as next input (ping-pong buffers)
      treeReduce func tileSize tileSizeInt nextTiles ptrOut ptrIn
