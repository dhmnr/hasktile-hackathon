{-# LANGUAGE DataKinds #-}

module Main where

import TileIR.DSL
import TileIR.Runtime

-- Softmax with numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
--
-- This demonstrates composing multiple operations:
-- 1. Find max(x) via fold
-- 2. Subtract max from all elements (x - max)
-- 3. Apply exp to get exp(x - max)
-- 4. Sum all exp values
-- 5. Divide each exp by sum

-- Step 1: Find maximum value (for numerical stability)
findMax :: Tile 16 Float -> Tile 16 Float
findMax x = tileFold maxOf (lit (-1e38)) x  -- Start with very small number

-- Step 2: Subtract max from each element
subtractMax :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
subtractMax x maxVal = tileZipWith (.-.) x maxVal

-- Step 3: Apply exp (we'll approximate for now since we need to add exp support to codegen)
-- For demonstration, let's use x^2 as a stand-in (positive values like exp)
applyExp :: Tile 16 Float -> Tile 16 Float
applyExp x = tileMap (\v -> v .*. v) x  -- Square as exp approximation

-- Step 4: Sum all values
computeSum :: Tile 16 Float -> Tile 16 Float
computeSum x = tileFold (.+.) (lit 0.0) x

-- Step 5: Normalize by dividing by sum
normalize :: Tile 16 Float -> Tile 16 Float -> Tile 16 Float
normalize x sumVal = tileZipWith (./.) x sumVal

-- Full softmax pipeline (simplified version)
-- In a real implementation, we'd need multiple kernel launches or shared memory
softmaxStep1 :: Tile 16 Float -> Tile 16 Float
softmaxStep1 x =
  let maxVal = findMax x           -- Find max
      shifted = subtractMax x maxVal  -- Subtract max
  in shifted

softmaxStep2 :: Tile 16 Float -> Tile 16 Float
softmaxStep2 shifted =
  let expVals = applyExp shifted   -- Apply exp (approximated)
  in expVals

softmaxStep3 :: Tile 16 Float -> Tile 16 Float
softmaxStep3 expVals =
  let sumVal = computeSum expVals  -- Sum
      result = normalize expVals sumVal  -- Normalize
  in result

main :: IO ()
main = do
  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║              Softmax-Like Computation                      ║"
  putStrLn "║  (Using available primitives: fold, map, zipWith)         ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"

  putStrLn "\nNote: True softmax requires exp() which needs MLIR math.exp support."
  putStrLn "This demo shows the pipeline with square (x²) as a stand-in for exp."
  putStrLn ""

  -- Test data: small positive values
  let input = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0] :: [Float]

  putStrLn "=== Step 1: Find max and shift (x - max) ==="
  arrIn <- loadArray input
  step1Result <- runTiled softmaxStep1 arrIn
  let Array step1List = step1Result
  putStrLn $ "Input:         " ++ show (take 8 input)
  putStrLn $ "After shift:   " ++ show (take 8 step1List)

  putStrLn "\n=== Step 2: Apply function (x² as exp approximation) ==="
  step2Result <- runTiled softmaxStep2 step1Result
  let Array step2List = step2Result
  putStrLn $ "After square:  " ++ show (take 8 step2List)

  putStrLn "\n=== Step 3: Normalize (divide by sum) ==="
  step3Result <- runTiled softmaxStep3 step2Result
  let Array finalList = step3Result
  putStrLn $ "Normalized:    " ++ show (take 8 finalList)

  -- Verify sum is approximately 1.0
  let totalSum = sum finalList
  putStrLn $ "\nSum of outputs: " ++ show totalSum
  putStrLn $ "Expected: ~1.0 for proper normalization"

  if abs (totalSum - 1.0) < 0.01
    then putStrLn "✓ Softmax-like pipeline works!"
    else putStrLn "✗ Normalization off (expected for demonstration)"

  putStrLn "\n╔════════════════════════════════════════════════════════════╗"
  putStrLn "║  This demonstrates composing:                              ║"
  putStrLn "║  • fold (max, sum)                                         ║"
  putStrLn "║  • map (transform)                                         ║"
  putStrLn "║  • zipWith (subtract, divide)                              ║"
  putStrLn "║  Into a complex multi-step computation!                    ║"
  putStrLn "╚════════════════════════════════════════════════════════════╝"
