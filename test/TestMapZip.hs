{-# LANGUAGE DataKinds #-}

module Main where

import TileIR.DSL
import TileIR.Runtime

-- | Test map: square each element
squareKernel :: Tile 16 Float -> Tile 16 Float
squareKernel a = tileMap (\x -> x .*. x) a

-- | Test zipWith: element-wise addition
addKernel :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
addKernel a b = tileZipWith (\x y -> x .+. y) a b

-- | Test zipWith: element-wise multiplication
mulKernel :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
mulKernel a b = tileZipWith (\x y -> x .*. y) a b

-- | Test complex expression: (a + b) * c
complexKernel :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float -> Tile 16 Float
complexKernel a b c =
  let sum_ab = tileZipWith (\x y -> x .+. y) a b
  in tileZipWith (\s z -> s .*. z) sum_ab c

-- | Test map with multiple operations: 2*x + 1
linearKernel :: Tile 16 Float -> Tile 16 Float
linearKernel a = tileMap (\x -> lit 2.0 .*. x .+. lit 1.0) a

main :: IO ()
main = do
  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║           Map & ZipWith Operations Test                   ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"

  -- Test 1: Map (square)
  putStrLn "\n=== Test 1: Map - Square each element ==="
  let input1 = [1..16] :: [Float]
  arr1 <- loadArray input1
  result1 <- runTiled squareKernel arr1
  let Array outputList1 = result1

  putStrLn $ "Input:    " ++ show input1
  putStrLn $ "Output:   " ++ show outputList1
  putStrLn $ "Expected: " ++ show (map (\x -> x * x) input1)

  let correct1 = all (\(a, b) -> abs (a - b) < 0.01) (zip outputList1 (map (\x -> x * x) input1))
  putStrLn $ if correct1 then "✓ Map test passed!" else "✗ Map test failed!"

  -- Test 2: ZipWith (addition)
  putStrLn "\n\n=== Test 2: ZipWith - Element-wise addition ==="
  let inputA2 = [1..16] :: [Float]
  let inputB2 = [10, 20..160] :: [Float]
  arrA2 <- loadArray inputA2
  arrB2 <- loadArray inputB2
  result2 <- runTiled addKernel (arrA2, arrB2)
  let Array outputList2 = result2

  putStrLn $ "Input A:  " ++ show inputA2
  putStrLn $ "Input B:  " ++ show inputB2
  putStrLn $ "Output:   " ++ show outputList2
  putStrLn $ "Expected: " ++ show (zipWith (+) inputA2 inputB2)

  let correct2 = all (\(a, b) -> abs (a - b) < 0.01) (zip outputList2 (zipWith (+) inputA2 inputB2))
  putStrLn $ if correct2 then "✓ ZipWith addition test passed!" else "✗ ZipWith addition test failed!"

  -- Test 3: ZipWith (multiplication)
  putStrLn "\n\n=== Test 3: ZipWith - Element-wise multiplication ==="
  let inputA3 = [1..16] :: [Float]
  let inputB3 = [2, 4..32] :: [Float]  -- Even numbers from 2 to 32
  arrA3 <- loadArray inputA3
  arrB3 <- loadArray inputB3
  result3 <- runTiled mulKernel (arrA3, arrB3)
  let Array outputList3 = result3

  putStrLn $ "Input A:  " ++ show inputA3
  putStrLn $ "Input B:  " ++ show inputB3
  putStrLn $ "Output:   " ++ show outputList3
  putStrLn $ "Expected: " ++ show (zipWith (*) inputA3 inputB3)

  let correct3 = all (\(a, b) -> abs (a - b) < 0.01) (zip outputList3 (zipWith (*) inputA3 inputB3))
  putStrLn $ if correct3 then "✓ ZipWith multiplication test passed!" else "✗ ZipWith multiplication test failed!"

  -- Test 4: Complex expression
  putStrLn "\n\n=== Test 4: Complex - (a + b) * c ==="
  let inputA4 = [1..16] :: [Float]
  let inputB4 = [2, 2..32] :: [Float]
  let inputC4 = [0.5, 0.5..8] :: [Float]
  arrA4 <- loadArray inputA4
  arrB4 <- loadArray inputB4
  arrC4 <- loadArray inputC4
  result4 <- runTiled complexKernel (arrA4, arrB4, arrC4)
  let Array outputList4 = result4

  let expected4 = zipWith (*) (zipWith (+) inputA4 inputB4) inputC4
  putStrLn $ "Input A:  " ++ show inputA4
  putStrLn $ "Input B:  " ++ show inputB4
  putStrLn $ "Input C:  " ++ show inputC4
  putStrLn $ "Output:   " ++ show outputList4
  putStrLn $ "Expected: " ++ show expected4

  let correct4 = all (\(a, b) -> abs (a - b) < 0.01) (zip outputList4 expected4)
  putStrLn $ if correct4 then "✓ Complex expression test passed!" else "✗ Complex expression test failed!"

  -- Test 5: Map with linear transformation
  putStrLn "\n\n=== Test 5: Map - Linear transformation (2*x + 1) ==="
  let input5 = [0..15] :: [Float]
  arr5 <- loadArray input5
  result5 <- runTiled linearKernel arr5
  let Array outputList5 = result5

  let expected5 = map (\x -> 2*x + 1) input5
  putStrLn $ "Input:    " ++ show input5
  putStrLn $ "Output:   " ++ show outputList5
  putStrLn $ "Expected: " ++ show expected5

  let correct5 = all (\(a, b) -> abs (a - b) < 0.01) (zip outputList5 expected5)
  putStrLn $ if correct5 then "✓ Linear transformation test passed!" else "✗ Linear transformation test failed!"

  -- Summary
  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  let allPassed = correct1 && correct2 && correct3 && correct4 && correct5
  if allPassed
    then putStrLn "║  ✓ All map & zipWith tests passed!                        ║"
    else putStrLn "║  ✗ Some tests failed!                                     ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"
