{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import TileIR.DSL
import TileIR.CodeGen

-- Multiply instead of add
vectorMul :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorMul a b = tileZipWith (.*.) a b

main :: IO ()
main = do
  putStrLn "=== Vector Multiply (from AST) ==="
  let kernel = kernel2 @16 "vector_mul" vectorMul
  let mlir = generateMLIR kernel
  putStrLn $ show mlir
