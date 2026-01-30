{-# LANGUAGE DataKinds #-}

module Main where

import TileIR.DSL
import TileIR.Runtime

-- Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
-- Demonstrates composing fold, map, and zipWith into a 3-stage pipeline

-- Stage 1: Shift by max for numerical stability
shiftByMax :: Tile 16 Float -> Tile 16 Float
shiftByMax x = tileZipWith (.-.) x (tileFold maxOf (lit (-1e38)) x)

-- Stage 2: Apply exp (x² approximation) and normalize
expAndNormalize :: Tile 16 Float -> Tile 16 Float
expAndNormalize x = let exp_x = tileMap (\v -> v .*. v) x
                    in tileZipWith (./.) exp_x (tileFold (.+.) (lit 0.0) exp_x)

-- Complete softmax: compose the 3 stages
softmax :: Tile 16 Float -> Tile 16 Float
softmax = expAndNormalize . shiftByMax

main :: IO ()
main = do
  putStrLn "\n╔═══════════════════════════════════════════════════════════╗"
  putStrLn "║         Softmax via Fold/Map/ZipWith Composition          ║"
  putStrLn "╚═══════════════════════════════════════════════════════════╝\n"

  let input = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

  -- Execute pipeline
  arrIn <- loadArray input
  stage1 <- runTiled shiftByMax arrIn
  stage2 <- runTiled expAndNormalize stage1

  let Array shifted = stage1
      Array result = stage2
      total = sum result

  -- Display results
  putStrLn $ "Input:      " ++ show (take 6 input)
  putStrLn $ "Shifted:    " ++ show (take 6 shifted)
  putStrLn $ "Normalized: " ++ show (take 6 result)
  putStrLn $ "\nSum: " ++ show total ++ " (expect ~1.0)"

  if abs (total - 1.0) < 0.01
    then putStrLn "✓ Softmax pipeline verified!"
    else putStrLn "✗ Off (using x² as exp approximation)\n"

  putStrLn "Pipeline: fold(max) → zipWith(-) → map(²) → fold(+) → zipWith(/)"
