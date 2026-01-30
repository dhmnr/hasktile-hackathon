{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import TileIR.DSL
import TileIR.Runtime

-- Vector addition kernel
vectorAdd :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorAdd a b = tileZipWith (.+.) a b

main :: IO ()
main = do

  -- Create test data (16 elements - single tile)
  let inputA = [0..15] :: [Float]
  let inputB = [100..115] :: [Float]
  let expected = zipWith (+) inputA inputB

  putStrLn "Input:"
  putStrLn $ "  a = " ++ show inputA
  putStrLn $ "  b = " ++ show inputB
  putStrLn $ "  expected = " ++ show expected

  arrA <- loadArray inputA
  arrB <- loadArray inputB

  result <- runTiled @16 vectorAdd arrA arrB

  putStrLn "\nResult:"
  print result

  let Array resultList = result
  let matches = resultList == expected
  putStrLn $ "\n Results match: " ++ show matches
