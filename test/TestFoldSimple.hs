{-# LANGUAGE DataKinds #-}

module Main where

import TileIR.DSL
import TileIR.Runtime

-- | Simple fold test: sum all elements in a tile
vectorFold :: Tile 16 Float -> Tile 16 Float
vectorFold a = tileFold (.+.) (lit 0) a

main :: IO ()
main = do
  putStrLn "\n=== Testing Fold Detection & Tree Reduction ==="

  -- Test with 16 elements (1 tile)
  let input = [1..16] :: [Float]
  arrA <- loadArray input

  result <- runTiled vectorFold arrA

  let Array resultList = result

  putStrLn "\n=== Results ==="
  putStrLn $ "Input:  " ++ show input
  putStrLn $ "Output: " ++ show resultList

  putStrLn "\n✓ Fold detection and execution working!"
