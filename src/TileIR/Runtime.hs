{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TileIR.Runtime
  ( runTiled
  , Array(..)
  , loadArray
  ) where

import TileIR.Types
import TileIR.DSL
import TileIR.CodeGen
import TileIR.CUDADriver
import GHC.TypeLits (KnownNat)
import Foreign.Ptr
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc (alloca)
import Foreign.Storable
import Foreign.StablePtr
import Data.Word (Word64)
import System.Process (callProcess)
import System.IO.Temp (withSystemTempDirectory)
import qualified Data.Text.IO as TIO
import System.FilePath ((</>))

-- | Array type (host-side)
newtype Array a = Array [a]
  deriving (Show)

-- | Load array from list
loadArray :: [a] -> IO (Array a)
loadArray xs = return $ Array xs

-- | Run a tiled kernel on arrays
runTiled :: forall n a b c. (KnownNat n, Storable a, Storable b, Storable c, Num c)
         => (Tile n a -> Tile n b -> Tile n c)
         -> Array a
         -> Array b
         -> IO (Array c)
runTiled f (Array listA) (Array listB) = do
  -- Generate MLIR
  let kernel = kernel2 @n "vector_add" f
  let mlir = generateMLIR kernel

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
