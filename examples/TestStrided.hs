{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import TileIR.DSL
import TileIR.CodeGen

-- Example 1: SAXPY with strided access
-- Compute: Y = a*X + Y where a is scalar
-- Using strided loads to access non-contiguous data
saxpyStrided :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
saxpyStrided x y = tileZipWith (.+.) y (tileMap (\xi -> lit 2.0 .*. xi) x)

-- Example 2: Load with explicit stride
-- This loads every 4th element: [0, 4, 8, 12, 16, ...]
stridedLoad :: Tile 16 Float
stridedLoad = tileLoadStrided "A" 0 4

-- Example 3: Load with offset
-- This loads starting from element 100: [100, 101, 102, ...]
offsetLoad :: Tile 16 Float
offsetLoad = tileLoadOffset "A" 100

-- Example 4: Load with both offset and stride
-- This loads [64, 68, 72, 76, ...] (start at 64, stride by 4)
offsetStridedLoad :: Tile 16 Float
offsetStridedLoad = tileLoadStrided "A" 64 4

-- Example 5: Matrix column access
-- For a matrix stored in row-major order, access a column with stride=width
-- e.g., for 64x64 matrix, stride=64 accesses one column
columnAccess :: Tile 16 Float
columnAccess = tileLoadStrided "matrix" 0 64

-- Example 6: Different strides for inputs
-- Process interleaved data: even indices from A, odd indices from B
interleavedOp :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
interleavedOp a b =
  -- This would require explicit TileVarStrided in the kernel function
  -- For now, just multiply normally
  tileZipWith (.*.) a b

main :: IO ()
main = do
  putStrLn "=== Example 1: Basic SAXPY (stride=1, offset=0) ==="
  let kernel1 = kernel2 @16 "saxpy" saxpyStrided
  let mlir1 = generateMLIR kernel1
  putStrLn $ show mlir1
  putStrLn ""

  putStrLn "=== Example 2: Strided Load (stride=4) ==="
  putStrLn "Define a kernel that loads A with stride 4:"
  putStrLn "  let a = tileLoadStrided \"A\" 0 4"
  putStrLn "This will generate load instructions with offset=[0,4,8,12,...]"
  putStrLn ""

  putStrLn "=== Example 3: Offset Load (start at 100) ==="
  putStrLn "Define a kernel that loads A starting at index 100:"
  putStrLn "  let a = tileLoadOffset \"A\" 100"
  putStrLn "This will generate load instructions with offset=[100,101,102,...]"
  putStrLn ""

  putStrLn "=== Example 4: Matrix Column Access ==="
  putStrLn "For a 64x64 matrix in row-major order:"
  putStrLn "  let column = tileLoadStrided \"matrix\" 0 64"
  putStrLn "This loads one column (stride=64 skips to next row)"
  putStrLn ""

  putStrLn "=== Use Cases for Strided Loads ==="
  putStrLn "1. GEMM: Load different slices for K-loop iterations"
  putStrLn "   Block 0 iter 0: load A[:, 0:16]   (offset=0,   stride=1)"
  putStrLn "   Block 0 iter 1: load A[:, 16:32]  (offset=16,  stride=1)"
  putStrLn "   Block 0 iter 2: load A[:, 32:48]  (offset=32,  stride=1)"
  putStrLn ""
  putStrLn "2. Matrix transpose: Access columns with stride"
  putStrLn "   Column 0: offset=0,  stride=width"
  putStrLn "   Column 1: offset=1,  stride=width"
  putStrLn ""
  putStrLn "3. Interleaved data: Process every Nth element"
  putStrLn "   Even indices: offset=0, stride=2"
  putStrLn "   Odd indices:  offset=1, stride=2"
  putStrLn ""
  putStrLn "4. Block-diagonal matrices: Skip to diagonal blocks"
  putStrLn "   Block (i,i): offset=i*block_size*(width+1), stride=1"
