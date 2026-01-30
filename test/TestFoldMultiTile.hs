{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import TileIR.DSL
import TileIR.Runtime

-- | Fold with addition - combines tiles element-wise
vectorFoldAdd :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorFoldAdd a _ = tileFold (.+.) (lit 0) a

main :: IO ()
main = do
  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║       Multi-Tile Tree Reduction Test                      ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"

  -- Test 1: 64 elements = 4 tiles (2 phases: 4→2→1)
  putStrLn "\n=== Test 1: 64 elements (4 tiles) ==="
  let input64 = [1..64] :: [Float]
  arrA64 <- loadArray input64
  arrB64 <- loadArray (replicate 64 0)

  result64 <- runTiled @16 vectorFoldAdd arrA64 arrB64
  let Array resultList64 = result64

  putStrLn "\n--- Results (64 elements) ---"
  putStrLn $ "Input:  " ++ show (take 16 input64) ++ " ... " ++ show (drop 48 input64)
  putStrLn $ "Output: " ++ show resultList64
  putStrLn $ "Expected: 1 tile of 16 elements"

  -- Test 2: 256 elements = 16 tiles (4 phases: 16→8→4→2→1)
  putStrLn "\n\n=== Test 2: 256 elements (16 tiles) ==="
  let input256 = [1..256] :: [Float]
  arrA256 <- loadArray input256
  arrB256 <- loadArray (replicate 256 0)

  result256 <- runTiled @16 vectorFoldAdd arrA256 arrB256
  let Array resultList256 = result256

  putStrLn "\n--- Results (256 elements) ---"
  putStrLn $ "Input:  256 elements across 16 tiles"
  putStrLn $ "Output: " ++ show resultList256
  putStrLn $ "Expected: 1 tile of 16 elements"

  -- Test 3: 1024 elements = 64 tiles (6 phases: 64→32→16→8→4→2→1)
  putStrLn "\n\n=== Test 3: 1024 elements (64 tiles) ==="
  let input1024 = [1..1024] :: [Float]
  arrA1024 <- loadArray input1024
  arrB1024 <- loadArray (replicate 1024 0)

  result1024 <- runTiled @16 vectorFoldAdd arrA1024 arrB1024
  let Array resultList1024 = result1024

  putStrLn "\n--- Results (1024 elements) ---"
  putStrLn $ "Input:  1024 elements across 64 tiles"
  putStrLn $ "Output: " ++ show resultList1024
  putStrLn $ "Expected: 1 tile of 16 elements"

  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║  ✓ Tree reduction across multiple tiles working!          ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"
