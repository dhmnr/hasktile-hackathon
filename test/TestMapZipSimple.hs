{-# LANGUAGE DataKinds #-}

module Main where

import TileIR.DSL
import TileIR.Runtime

-- | Test zipWith: element-wise multiplication
mulKernel :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
mulKernel a b = tileZipWith (\x y -> x .*. y) a b

main :: IO ()
main = do
  putStrLn "\n=== Test: ZipWith - Element-wise multiplication ==="
  let inputA = [1..16] :: [Float]
  let inputB = [2, 4..32] :: [Float]  -- Even numbers from 2 to 32
  arrA <- loadArray inputA
  arrB <- loadArray inputB
  result <- runTiled mulKernel (arrA, arrB)
  let Array outputList = result

  putStrLn $ "Input A:  " ++ show inputA
  putStrLn $ "Input B:  " ++ show inputB
  putStrLn $ "Output:   " ++ show outputList
  putStrLn $ "Expected: " ++ show (zipWith (*) inputA inputB)

  let correct = all (\(a, b) -> abs (a - b) < 0.01) (zip outputList (zipWith (*) inputA inputB))
  putStrLn $ if correct then "✓ ZipWith multiplication test passed!" else "✗ ZipWith multiplication test failed!"
