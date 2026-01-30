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
  , tileScan
  , tileConst
  -- NOTE: tileLoad/tileStore removed from public API
  -- The kernel parameters ARE the tiles - no need to load explicitly!
  -- Stride will be specified at GPU buffer level (see Runtime)
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

-- | Scan (prefix scan) a tile with an accumulator
-- Returns tile where each element is the accumulated result up to that point
-- Example: tileScan (.+.) (lit 0) [1,2,3,4] = [1,3,6,10]
tileScan :: forall n a. KnownNat n
         => (ScalarExpr a -> ScalarExpr a -> ScalarExpr a)
         -> ScalarExpr a  -- Initial accumulator value
         -> Tile n a
         -> Tile n a
tileScan f init (Tile _ expr) = Tile (Proxy @n) (TileScan f init expr)

-- | Constant tile (all elements same value)
tileConst :: forall n a. KnownNat n => a -> Tile n a
tileConst val = Tile (Proxy @n) (TileConst val)

-- | Load tile from memory (basic: offset=0, stride=1)
tileLoad :: forall n a. KnownNat n => String -> Tile n a
tileLoad name = Tile (Proxy @n) (TileLoad name)

-- | Load tile from memory with stride and offset
-- Example: tileLoadStrided "A" 64 2 loads elements [64, 66, 68, 70, ...]
tileLoadStrided :: forall n a. KnownNat n
                => String    -- Variable name
                -> Int       -- Offset (starting element index)
                -> Int       -- Stride (elements between loads)
                -> Tile n a
tileLoadStrided name offset stride =
  Tile (Proxy @n) (TileVarStrided name offset stride)

-- | Load tile with offset only (stride=1)
-- Example: tileLoadOffset "A" 100 loads elements [100, 101, 102, ...]
tileLoadOffset :: forall n a. KnownNat n => String -> Int -> Tile n a
tileLoadOffset name offset = tileLoadStrided name offset 1

-- | Store tile to memory (basic: offset=0, stride=1)
tileStore :: forall n a. KnownNat n => String -> Tile n a -> Tile n ()
tileStore name (Tile _ expr) = Tile (Proxy @n) (TileStore name expr)

-- | Store tile to memory with stride and offset
-- Example: tileStoreStrided "out" 64 2 tile stores to [64, 66, 68, 70, ...]
tileStoreStrided :: forall n a. KnownNat n
                 => String    -- Destination name
                 -> Int       -- Offset (starting element index)
                 -> Int       -- Stride (elements between stores)
                 -> Tile n a
                 -> Tile n ()
tileStoreStrided name offset stride (Tile _ expr) =
  Tile (Proxy @n) (TileStoreStrided name offset stride expr)

-- | Store tile with offset only (stride=1)
-- Example: tileStoreOffset "out" 100 tile stores to [100, 101, 102, ...]
tileStoreOffset :: forall n a. KnownNat n => String -> Int -> Tile n a -> Tile n ()
tileStoreOffset name offset tile = tileStoreStrided name offset 1 tile

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
