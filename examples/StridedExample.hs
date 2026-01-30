{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import TileIR.DSL
import TileIR.CodeGen
import qualified Data.Vector.Storable as V

-- Example: Process interleaved data
-- Input: [a0, b0, a1, b1, a2, b2, ...] (interleaved A and B)
-- Load even indices (A): [a0, a1, a2, ...] using stride=2, offset=0
-- Load odd indices (B):  [b0, b1, b2, ...] using stride=2, offset=1

-- Define kernel that works on pre-loaded strided data
processInterleaved :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
processInterleaved a b = tileZipWith (.+.) a b

-- Example: Matrix column access (for transpose)
-- For NxN matrix in row-major order, access column with stride=N
transposeColumn :: Tile 16 Float -> Tile 16 Float
transposeColumn col = tileMap (\x -> x) col  -- Identity (just for demo)

-- Example: GEMM iteration - load different K slices
-- In GEMM, each iteration loads a different slice along K dimension
gemmIteration :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
gemmIteration a_slice b_slice = tileZipWith (.*.) a_slice b_slice

main :: IO ()
main = do
  putStrLn "╔═══════════════════════════════════════════════════════════╗"
  putStrLn "║         Strided Load/Store Examples                      ║"
  putStrLn "╚═══════════════════════════════════════════════════════════╝"
  putStrLn ""

  -- Example 1: Basic kernel
  putStrLn "=== Example 1: Standard Kernel (stride=1, offset=0) ==="
  let kernel1 = kernel2 @16 "process_interleaved" processInterleaved
  let mlir1 = generateMLIR kernel1
  putStrLn $ show mlir1
  putStrLn ""

  -- Documentation for strided usage
  putStrLn "=== How to Use Strided Loads ==="
  putStrLn ""
  putStrLn "1. Load every Nth element (stride):"
  putStrLn "   tileLoadStrided \"data\" 0 N"
  putStrLn "   Example: stride=2 loads [0, 2, 4, 6, ...]"
  putStrLn ""
  putStrLn "2. Load starting from offset:"
  putStrLn "   tileLoadOffset \"data\" offset"
  putStrLn "   Example: offset=100 loads [100, 101, 102, ...]"
  putStrLn ""
  putStrLn "3. Combine offset and stride:"
  putStrLn "   tileLoadStrided \"data\" offset stride"
  putStrLn "   Example: offset=1, stride=2 loads [1, 3, 5, 7, ...] (odd indices)"
  putStrLn ""

  putStrLn "=== Use Case: Interleaved Data ==="
  putStrLn "Input data: [a0, b0, a1, b1, a2, b2, a3, b3, ...]"
  putStrLn ""
  putStrLn "To process separately:"
  putStrLn "  Block 0: Load A with stride=2, offset=0  → [a0, a1, a2, a3, ...]"
  putStrLn "           Load B with stride=2, offset=1  → [b0, b1, b2, b3, ...]"
  putStrLn "           Compute: A + B"
  putStrLn ""
  putStrLn "Haskell setup:"
  putStrLn "  -- Prepare interleaved data on CPU"
  putStrLn "  let interleaved = V.fromList [a0, b0, a1, b1, ...]"
  putStrLn ""
  putStrLn "  -- Copy to GPU"
  putStrLn "  gpu_data <- toGPU interleaved"
  putStrLn ""
  putStrLn "  -- Launch kernel (in future: specify strides in launch config)"
  putStrLn "  result <- runKernel kernel gpu_data"
  putStrLn ""

  putStrLn "=== Use Case: Matrix Operations ==="
  putStrLn "For 64×64 matrix in row-major order:"
  putStrLn ""
  putStrLn "Access row 0:    offset=0,    stride=1   → [0, 1, 2, ..., 63]"
  putStrLn "Access row 1:    offset=64,   stride=1   → [64, 65, 66, ..., 127]"
  putStrLn "Access column 0: offset=0,    stride=64  → [0, 64, 128, ..., 4032]"
  putStrLn "Access column 1: offset=1,    stride=64  → [1, 65, 129, ..., 4033]"
  putStrLn ""

  putStrLn "=== Use Case: GEMM K-loop ==="
  putStrLn "Matrix multiply C = A × B for 64×64 blocks:"
  putStrLn ""
  putStrLn "Each block iterates over K dimension (e.g., K=1024, tile=64):"
  putStrLn "  Iteration 0: Load A[:,0:64]    offset=0,    stride=1"
  putStrLn "               Load B[0:64,:]    offset=0,    stride=1"
  putStrLn "  Iteration 1: Load A[:,64:128]  offset=64,   stride=1"
  putStrLn "               Load B[64:128,:]  offset=4096, stride=1"
  putStrLn "  Iteration 2: Load A[:,128:192] offset=128,  stride=1"
  putStrLn "               Load B[128:192,:] offset=8192, stride=1"
  putStrLn "  ..."
  putStrLn ""
  putStrLn "The offset changes per iteration to load different slices!"
  putStrLn ""

  putStrLn "=== Implementation Status ==="
  putStrLn "✓ TileVarStrided AST node added"
  putStrLn "✓ tileLoadStrided DSL function added"
  putStrLn "✓ tileStoreStrided DSL function added"
  putStrLn "✓ Code generation for strided loads"
  putStrLn "✓ MLIR generation with stride and offset"
  putStrLn ""
  putStrLn "Next steps:"
  putStrLn "  - Test with real kernels"
  putStrLn "  - Add runtime support for passing stride configs"
  putStrLn "  - Support for dynamic offsets (based on block ID)"
