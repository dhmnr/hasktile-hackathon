{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import TileIR.DSL
import TileIR.Runtime
import Control.Monad (when)
import Data.List (intercalate)

-- ============================================================================
-- Test Kernels
-- ============================================================================

-- | Sum reduction: compute sum and broadcast to all elements
vectorSum :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorSum a _ = tileFold (.+.) (lit 0) a

-- | Product reduction: compute product and broadcast
vectorProduct :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorProduct a _ = tileFold (.*.) (lit 1) a

-- | Prefix sum: cumulative sum
vectorPrefixSum :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorPrefixSum a _ = tileScan (.+.) (lit 0) a

-- | Prefix product: cumulative product
vectorPrefixProduct :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
vectorPrefixProduct a _ = tileScan (.*.) (lit 1) a

-- ============================================================================
-- Test Configuration
-- ============================================================================

data TestConfig = TestConfig
  { testName :: String
  , testKernel :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
  , testInputA :: [Float]
  , testInputB :: [Float]  -- Dummy input
  , testExpected :: [Float]
  }

tests :: [TestConfig]
tests =
  [ TestConfig
      { testName = "Sum Reduction (1..16)"
      , testKernel = vectorSum
      , testInputA = [1..16]
      , testInputB = replicate 16 0
      , testExpected = replicate 16 (sum [1..16])  -- All elements = 136
      }
  , TestConfig
      { testName = "Sum Reduction (all ones)"
      , testKernel = vectorSum
      , testInputA = replicate 16 1
      , testInputB = replicate 16 0
      , testExpected = replicate 16 16  -- All elements = 16
      }
  , TestConfig
      { testName = "Product Reduction (2,2,2...)"
      , testKernel = vectorProduct
      , testInputA = replicate 16 2
      , testInputB = replicate 16 0
      , testExpected = replicate 16 (2^16)  -- All elements = 65536
      }
  , TestConfig
      { testName = "Prefix Sum (1..16)"
      , testKernel = vectorPrefixSum
      , testInputA = [1..16]
      , testInputB = replicate 16 0
      , testExpected = [1,3,6,10,15,21,28,36,45,55,66,78,91,105,120,136]
      }
  , TestConfig
      { testName = "Prefix Sum (all ones)"
      , testKernel = vectorPrefixSum
      , testInputA = replicate 16 1
      , testInputB = replicate 16 0
      , testExpected = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
      }
  , TestConfig
      { testName = "Prefix Product (1..5, then 1s)"
      , testKernel = vectorPrefixProduct
      , testInputA = [1,2,3,4,5,1,1,1,1,1,1,1,1,1,1,1]
      , testInputB = replicate 16 0
      , testExpected = [1,2,6,24,120,120,120,120,120,120,120,120,120,120,120,120]
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
      putStrLn $ "│    Input:    " ++ formatList (testInputA config)
      putStrLn $ "│    Result:   " ++ formatList resultList
      putStrLn $ "│    Expected: " ++ formatList expected
    else do
      putStrLn $ "│  ✗ FAIL"
      putStrLn $ "│    Input:    " ++ formatList (testInputA config)
      putStrLn $ "│    Expected: " ++ formatList expected
      putStrLn $ "│    Got:      " ++ formatList resultList

  putStrLn "└─"
  return matches

formatList :: Show a => [a] -> String
formatList xs
  | length xs <= 8 = "[" ++ intercalate ", " (map show xs) ++ "]"
  | otherwise = "[" ++ intercalate ", " (map show $ take 4 xs) ++ ", ..., " ++
                       intercalate ", " (map show $ drop (length xs - 4) xs) ++ "]"

main :: IO ()
main = do
  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║      Reduction & Scan Operations Test Suite               ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"

  results <- mapM runTest tests

  let passed = length $ filter id results
  let total = length results

  putStrLn $ "\n" ++ replicate 60 '='
  putStrLn $ "  SUMMARY: " ++ show passed ++ "/" ++ show total ++ " tests passed"
  putStrLn $ replicate 60 '='

  if passed == total
    then putStrLn "  Status: ✓ ALL TESTS PASSED\n"
    else do
      putStrLn "  Status: ✗ SOME TESTS FAILED\n"
      error "Test suite failed"
