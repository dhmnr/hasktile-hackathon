{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module TileIR.DSL
  ( Tile
  , TileKernel
  , kernel1, kernel2, kernel3
  , tileMap
  , tileZipWith
  , tileFold
  , tileConst
  , tileLoad
  , tileStore
  -- Scalar operations
  , lit
  , (.+.), (.-.), (.*.)
  , (.<.), (.>.), (.==.)
  ) where

import TileIR.Types
import GHC.TypeLits (KnownNat)
import Data.Proxy

-- | Create a 1-argument tile kernel
kernel1 :: forall n a b. KnownNat n
        => String
        -> (Tile n a -> Tile n b)
        -> TileKernel
kernel1 = TileKernel1

-- | Create a 2-argument tile kernel
kernel2 :: forall n a b c. KnownNat n
        => String
        -> (Tile n a -> Tile n b -> Tile n c)
        -> TileKernel
kernel2 = TileKernel2

-- | Create a 3-argument tile kernel
kernel3 :: forall n a b c d. KnownNat n
        => String
        -> (Tile n a -> Tile n b -> Tile n c -> Tile n d)
        -> TileKernel
kernel3 = TileKernel3

-- | Map a function over tile elements
tileMap :: forall n a b. KnownNat n
        => (ScalarExpr a -> ScalarExpr b)
        -> Tile n a
        -> Tile n b
tileMap f (Tile _ expr) = Tile (Proxy @n) (TileMap f expr)

-- | Zip two tiles with a binary function
tileZipWith :: forall n a b c. KnownNat n
            => (ScalarExpr a -> ScalarExpr b -> ScalarExpr c)
            -> Tile n a
            -> Tile n b
            -> Tile n c
tileZipWith f (Tile _ exprA) (Tile _ exprB) =
  Tile (Proxy @n) (TileZipWith f exprA exprB)

-- | Fold/reduce a tile to a single value (returns tile with all same values)
tileFold :: forall n a. KnownNat n
         => (ScalarExpr a -> ScalarExpr a -> ScalarExpr a)
         -> ScalarExpr a
         -> Tile n a
         -> Tile n a
tileFold f init (Tile _ expr) = Tile (Proxy @n) (TileFold f init expr)

-- | Constant tile (all elements same value)
tileConst :: forall n a. KnownNat n => a -> Tile n a
tileConst val = Tile (Proxy @n) (TileConst val)

-- | Load tile from memory
tileLoad :: forall n a. KnownNat n => String -> Tile n a
tileLoad name = Tile (Proxy @n) (TileLoad name)

-- | Store tile to memory
tileStore :: forall n a. KnownNat n => String -> Tile n a -> Tile n ()
tileStore name (Tile _ expr) = Tile (Proxy @n) (TileStore name expr)

-- Scalar operations

-- | Literal scalar value
lit :: a -> ScalarExpr a
lit = ScalarLit

-- | Addition
(.+.) :: Num a => ScalarExpr a -> ScalarExpr a -> ScalarExpr a
(.+.) = Add

-- | Subtraction
(.-.) :: Num a => ScalarExpr a -> ScalarExpr a -> ScalarExpr a
(.-.) = Sub

-- | Multiplication
(.*.) :: Num a => ScalarExpr a -> ScalarExpr a -> ScalarExpr a
(.*.) = Mul

-- | Less than
(.<.) :: Ord a => ScalarExpr a -> ScalarExpr a -> ScalarExpr Bool
(.<.) = Lt

-- | Greater than
(.>.) :: Ord a => ScalarExpr a -> ScalarExpr a -> ScalarExpr Bool
(.>.) = Gt

-- | Equal
(.==.) :: Eq a => ScalarExpr a -> ScalarExpr a -> ScalarExpr Bool
(.==.) = Eq

infixl 6 .+., .-.
infixl 7 .*.
infix 4 .<., .>., .==.
