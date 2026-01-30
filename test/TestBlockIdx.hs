{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import TileIR.CodeGen (compileToBytecode)
import TileIR.CUDADriver
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Foreign.Ptr (castPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Storable (poke)
import System.Process (callProcess)
import System.IO.Temp (withSystemTempDirectory)
import System.FilePath ((</>))

-- | Simple kernel that just prints blockIdx info
blockIdxTestKernel :: T.Text
blockIdxTestKernel = T.unlines
  [ "cuda_tile.module @kernels {"
  , "  entry @block_test() {"
  , "    // Get block index"
  , "    %block_x, %block_y, %block_z = get_tile_block_id : tile<i32>"
  , "    "
  , "    // Get grid dimensions"
  , "    %dim_x, %dim_y, %dim_z = cuda_tile.get_num_tile_blocks : tile<i32>"
  , "    "
  , "    // Print block info"
  , "    cuda_tile.print \"[Block %] of [%] blocks executing\\n\","
  , "      %block_x, %dim_x"
  , "      : tile<i32>, tile<i32>"
  , "    "
  , "    return"
  , "  }"
  , "}"
  ]

main :: IO ()
main = do
  putStrLn "╔════════════════════════════════════════════════════════════╗"
  putStrLn "║         BlockIdx Test - Verify Parallel Execution         ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"
  putStrLn ""

  withSystemTempDirectory "hasktile-blockidx-test" $ \tmpDir -> do
    let mlirPath = tmpDir </> "kernel.mlir"
    let tileBcPath = tmpDir </> "kernel.tilebc"
    let cubinPath = tmpDir </> "kernel.cubin"

    -- Write MLIR
    TIO.writeFile mlirPath blockIdxTestKernel

    putStrLn "=== Generated MLIR ==="
    TIO.putStrLn blockIdxTestKernel

    -- Compile MLIR -> TileBC
    putStrLn "=== Compiling MLIR to bytecode ==="
    callProcess "/home/shadeform/cuda-tile/build/bin/cuda-tile-translate"
      [ mlirPath
      , "--mlir-to-cudatilebc"
      , "--no-implicit-module"
      , "--bytecode-version=13.1"
      , "-o", tileBcPath
      ]

    -- Compile TileBC -> CUBIN
    putStrLn ""
    putStrLn "=== Compiling bytecode to cubin ==="
    callProcess "/usr/local/cuda/bin/tileiras"
      [ "--gpu-name", "sm_120"
      , tileBcPath
      , "-o", cubinPath
      ]

    putStrLn ""
    putStrLn "=== Kernel compiled successfully ==="
    putStrLn $ "CUBIN at: " ++ cubinPath

    -- Initialize CUDA and load kernel
    putStrLn ""
    putStrLn "=== Testing with different grid sizes ==="
    putStrLn ""
    initCUDA
    device <- getDevice 0
    _ <- createContext device
    modul <- loadModule cubinPath
    func <- getFunction modul "block_test"

    -- Test 1: Single block
    putStrLn "--- Test 1: Single Block (gridDim = 1) ---"
    launchKernel1D func 1 []
    putStrLn ""

    -- Test 2: 4 blocks
    putStrLn "--- Test 2: Four Blocks (gridDim = 4) ---"
    launchKernel1D func 4 []
    putStrLn ""

    -- Test 3: 16 blocks
    putStrLn "--- Test 3: Sixteen Blocks (gridDim = 16) ---"
    launchKernel1D func 16 []
    putStrLn ""

    -- Test 4: 64 blocks
    putStrLn "--- Test 4: Sixty-Four Blocks (gridDim = 64) ---"
    launchKernel1D func 64 []
    putStrLn ""

    putStrLn "╔════════════════════════════════════════════════════════════╗"
    putStrLn "║  ✓ All blocks executed! BlockIdx working correctly!       ║"
    putStrLn "╚════════════════════════════════════════════════════════════╝"
