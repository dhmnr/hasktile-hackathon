{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import TileIR.DSL
import TileIR.Runtime
import Control.Monad (when)
import Data.List (intercalate)

-- | Test configuration
data TestConfig = TestConfig
  { testName :: String
  , testKernel :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
  , testInputA :: [Float]
  , testInputB :: [Float]
  , testExpected :: [Float]
  }

-- ============================================================================
-- Test Kernels
-- ============================================================================

-- | Vector addition
vectorAdd :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorAdd a b = tileZipWith (.+.) a b

-- | Vector multiplication
vectorMul :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorMul a b = tileZipWith (.*.) a b

-- | Vector subtraction
vectorSub :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorSub a b = tileZipWith (.-.) a b

-- | Fused multiply-add: a * b + a
vectorFMA :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorFMA a b =
  let prod = tileZipWith (.*.) a b
  in tileZipWith (.+.) prod a

-- | Scale vector: a * 2
vectorScale :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorScale a _ = tileZipWith (.*.) a a  -- Just use a*a for now (no constants yet)

-- ============================================================================
-- Test Cases
-- ============================================================================

basicTests :: [TestConfig]
basicTests =
  [ TestConfig
      { testName = "Vector Addition"
      , testKernel = vectorAdd
      , testInputA = [0..15]
      , testInputB = [100..115]
      , testExpected = zipWith (+) [0..15] [100..115]
      }
  , TestConfig
      { testName = "Vector Multiplication"
      , testKernel = vectorMul
      , testInputA = [1..16]
      , testInputB = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
      , testExpected = map (*2) [1..16]
      }
  , TestConfig
      { testName = "Vector Subtraction"
      , testKernel = vectorSub
      , testInputA = [100..115]
      , testInputB = [0..15]
      , testExpected = zipWith (-) [100..115] [0..15]
      }
  , TestConfig
      { testName = "Zeros"
      , testKernel = vectorMul
      , testInputA = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      , testInputB = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
      , testExpected = replicate 16 0
      }
  , TestConfig
      { testName = "Negative Numbers"
      , testKernel = vectorAdd
      , testInputA = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
      , testInputB = [8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7]
      , testExpected = replicate 16 0
      }
  ]

-- ============================================================================
-- Test Runner
-- ============================================================================

runTest :: TestConfig -> IO Bool
runTest config = do
  putStrLn $ "\n┌─ " ++ testName config
  putStrLn "│"

  -- Load data to GPU
  arrA <- loadArray (testInputA config)
  arrB <- loadArray (testInputB config)

  -- Run kernel
  result <- runTiled @16 (testKernel config) arrA arrB

  let Array resultList = result
  let expected = testExpected config
  let matches = resultList == expected

  -- Print results
  if matches
    then do
      putStrLn $ "│  ✓ PASS"
      when (length resultList <= 16) $ do
        putStrLn $ "│    Input A:  " ++ formatList (testInputA config)
        putStrLn $ "│    Input B:  " ++ formatList (testInputB config)
        putStrLn $ "│    Result:   " ++ formatList resultList
        putStrLn $ "│    Expected: " ++ formatList expected
    else do
      putStrLn $ "│  ✗ FAIL"
      putStrLn $ "│    Expected: " ++ formatList expected
      putStrLn $ "│    Got:      " ++ formatList resultList

  putStrLn "└─"
  return matches

formatList :: Show a => [a] -> String
formatList xs = "[" ++ intercalate ", " (map show xs) ++ "]"

runTestSuite :: String -> [TestConfig] -> IO (Int, Int)
runTestSuite suiteName tests = do
  putStrLn $ "\n" ++ replicate 60 '='
  putStrLn $ "  " ++ suiteName
  putStrLn $ replicate 60 '='

  results <- mapM runTest tests

  let passed = length $ filter id results
  let total = length results

  putStrLn $ "\n" ++ replicate 60 '-'
  putStrLn $ "  " ++ show passed ++ "/" ++ show total ++ " tests passed"
  putStrLn $ replicate 60 '-'

  return (passed, total)

-- ============================================================================
-- Main
-- ============================================================================

main :: IO ()
main = do
  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║         HasKTile Test Suite                                ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"

  -- Run basic tests
  (totalPassed, totalTests) <- runTestSuite "Basic Operations" basicTests

  -- Summary
  putStrLn $ "\n" ++ replicate 60 '='
  putStrLn "  FINAL SUMMARY"
  putStrLn $ replicate 60 '='
  putStrLn $ "  Total: " ++ show totalPassed ++ "/" ++ show totalTests ++ " tests passed"

  if totalPassed == totalTests
    then do
      putStrLn "  Status: ✓ ALL TESTS PASSED"
      putStrLn $ replicate 60 '='
      putStrLn ""
    else do
      putStrLn "  Status: ✗ SOME TESTS FAILED"
      putStrLn $ replicate 60 '='
      putStrLn ""
      error "Test suite failed"
