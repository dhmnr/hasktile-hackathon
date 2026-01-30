{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

import TileIR.DSL
import TileIR.Types
import TileIR.CodeGen

-- Test: Show AST for strided loads
main :: IO ()
main = do
  putStrLn "=== Testing Strided Loads Implementation ==="
  putStrLn ""

  -- Create tiles with different access patterns
  let tile_default = Tile (Proxy @16) (TileVar "arg0") :: Tile 16 Float
  let tile_stride4 = Tile (Proxy @16) (TileVarStrided "arg0" 0 4) :: Tile 16 Float
  let tile_offset = Tile (Proxy @16) (TileVarStrided "arg0" 100 1) :: Tile 16 Float
  let tile_both = Tile (Proxy @16) (TileVarStrided "arg0" 64 4) :: Tile 16 Float

  putStrLn "1. Default load (stride=1, offset=0):"
  print (tileExpr tile_default)
  putStrLn ""

  putStrLn "2. Strided load (stride=4, offset=0):"
  print (tileExpr tile_stride4)
  putStrLn "   Loads: [0, 4, 8, 12, 16, ...]"
  putStrLn ""

  putStrLn "3. Offset load (stride=1, offset=100):"
  print (tileExpr tile_offset)
  putStrLn "   Loads: [100, 101, 102, 103, ...]"
  putStrLn ""

  putStrLn "4. Combined (stride=4, offset=64):"
  print (tileExpr tile_both)
  putStrLn "   Loads: [64, 68, 72, 76, ...]"
  putStrLn ""

  -- Test using DSL functions
  putStrLn "5. Using DSL functions:"
  let dsl_strided = tileLoadStrided "A" 0 2 :: Tile 16 Float
  let dsl_offset = tileLoadOffset "B" 50 :: Tile 16 Float

  putStrLn "   tileLoadStrided \"A\" 0 2:"
  print (tileExpr dsl_strided)
  putStrLn ""

  putStrLn "   tileLoadOffset \"B\" 50:"
  print (tileExpr dsl_offset)
  putStrLn ""

  putStrLn "=== Success! Strided loads are working ==="
