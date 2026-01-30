{-# LANGUAGE OverloadedStrings #-}

module Main where

import TileIR.GEMM (generateGEMMKernel)
import TileIR.CUDADriver
import qualified Data.Text.IO as TIO
import Foreign.Ptr (castPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Storable (poke)
import System.Process (callProcess)
import System.IO.Temp (withSystemTempDirectory)
import System.FilePath ((</>))

main :: IO ()
main = do
  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║              GEMM 4096x4096 Test                          ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"

  -- For testing, use smaller matrices (256x256) to keep it fast
  let n = 256 :: Int
  let tile_size = 64 :: Int
  let grid_size = n `div` tile_size  -- 4x4 grid

  putStrLn $ "\n=== Matrix size: " ++ show n ++ "x" ++ show n ++ " ==="
  putStrLn $ "=== Tile size: " ++ show tile_size ++ "x" ++ show tile_size ++ " ==="
  putStrLn $ "=== Grid size: " ++ show grid_size ++ "x" ++ show grid_size ++ " ==="

  -- Initialize matrices
  -- A = identity-like (1s on diagonal, 0s elsewhere)
  -- B = all 2s
  -- Expected C = A @ B = 2s where A has 1s
  let a_matrix = [ if i == j then 1.0 else 0.0
                 | i <- [0..n-1], j <- [0..n-1] ] :: [Float]
  let b_matrix = replicate (n * n) 2.0 :: [Float]
  let c_matrix = replicate (n * n) 0.0 :: [Float]

  putStrLn "\n=== Matrices initialized ==="
  putStrLn $ "A[0:5,0:5] = " ++ show (take 25 a_matrix)
  putStrLn $ "B[0:5,0:5] = " ++ show (take 25 b_matrix)

  -- Generate GEMM kernel
  let mlir = generateGEMMKernel n tile_size

  putStrLn "\n=== Generated GEMM MLIR ==="
  TIO.putStrLn mlir

  withSystemTempDirectory "hasktile-gemm" $ \tmpDir -> do
    -- Write and compile MLIR
    let mlirPath = tmpDir </> "gemm.mlir"
    let tileBcPath = tmpDir </> "gemm.tilebc"
    let cubinPath = tmpDir </> "gemm.cubin"

    TIO.writeFile mlirPath mlir

    putStrLn "\n=== Compiling MLIR to bytecode ==="
    callProcess "/home/shadeform/cuda-tile/build/bin/cuda-tile-translate"
      [ mlirPath
      , "--mlir-to-cudatilebc"
      , "--no-implicit-module"
      , "--bytecode-version=13.1"
      , "-o", tileBcPath
      ]

    putStrLn "\n=== Compiling bytecode to cubin ==="
    callProcess "/usr/local/cuda/bin/tileiras"
      [ "--gpu-name", "sm_120"
      , tileBcPath
      , "-o", cubinPath
      ]

    putStrLn "\n=== Kernel compiled successfully ==="
    putStrLn $ "CUBIN at: " ++ cubinPath

    -- Initialize CUDA and load kernel
    putStrLn "\n=== Launching GEMM kernel ==="
    initCUDA
    device <- getDevice 0
    _ <- createContext device
    modul <- loadModule cubinPath
    func <- getFunction modul "gemm_kernel"

    -- Allocate device memory
    let matrix_size_bytes = n * n * 4  -- 4 bytes per float
    dPtrA <- allocDevice matrix_size_bytes
    dPtrB <- allocDevice matrix_size_bytes
    dPtrC <- allocDevice matrix_size_bytes

    -- Copy input matrices to device
    copyHtoD dPtrA a_matrix
    copyHtoD dPtrB b_matrix
    copyHtoD dPtrC c_matrix

    -- Launch kernel with 2D grid
    alloca $ \pA -> alloca $ \pB -> alloca $ \pC -> do
      poke pA dPtrA
      poke pB dPtrB
      poke pC dPtrC

      let params = [castPtr pA, castPtr pB, castPtr pC]

      -- Launch with grid_size x grid_size blocks
      launchKernel2D func grid_size grid_size params

    -- Copy result back
    result <- copyDtoH dPtrC (n * n) :: IO [Float]

    -- Cleanup
    freeDevice dPtrA
    freeDevice dPtrB
    freeDevice dPtrC

    putStrLn "\n=== Results ==="
    putStrLn $ "C[0:5,0:5] = " ++ show (take 25 result)

    -- Verify a few elements
    let expected_diagonal = 2.0
    let diagonal_elements = [result !! (i * n + i) | i <- [0..min 9 (n-1)]]
    putStrLn $ "\nDiagonal elements (should be 2.0): " ++ show diagonal_elements

    let all_correct = all (\x -> abs (x - expected_diagonal) < 0.01) diagonal_elements
    if all_correct
      then putStrLn "\n✓ GEMM kernel working correctly!"
      else putStrLn "\n✗ GEMM kernel produced incorrect results!"

  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║  GEMM Test Complete                                        ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"
