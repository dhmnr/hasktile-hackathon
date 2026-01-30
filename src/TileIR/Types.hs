{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module TileIR.Types where

import GHC.TypeLits (Nat, KnownNat)
import Data.Proxy

-- | Tile - sized array living in GPU registers
data Tile (n :: Nat) a = Tile
  { tileSize :: Proxy n
  , tileExpr :: TileExpr a
  }

deriving instance Show a => Show (Tile n a)

-- | Tile expression AST
data TileExpr a where
  -- Variable reference (basic, stride=1, offset=0)
  TileVar :: String -> TileExpr a

  -- Variable reference with stride and offset
  TileVarStrided :: String   -- variable name
                 -> Int      -- offset (starting element index)
                 -> Int      -- stride (elements between loads)
                 -> TileExpr a

  -- Constant tile (all elements same value)
  TileConst :: a -> TileExpr a

  -- Element-wise map
  TileMap :: (ScalarExpr a -> ScalarExpr b) -> TileExpr a -> TileExpr b

  -- Element-wise zip
  TileZipWith :: (ScalarExpr a -> ScalarExpr b -> ScalarExpr c)
              -> TileExpr a -> TileExpr b -> TileExpr c

  -- Fold/reduction (within a single tile - reduces to scalar broadcast to all elements)
  TileFold :: (ScalarExpr a -> ScalarExpr a -> ScalarExpr a)
           -> ScalarExpr a -> TileExpr a -> TileExpr a

  -- Scan (prefix scan within a single tile)
  TileScan :: (ScalarExpr a -> ScalarExpr a -> ScalarExpr a)
           -> ScalarExpr a  -- Initial value
           -> TileExpr a
           -> TileExpr a

  -- Load from memory (basic, stride=1, offset=0)
  TileLoad :: String -> TileExpr a

  -- Store to memory (basic, stride=1, offset=0)
  TileStore :: String -> TileExpr a -> TileExpr ()

  -- Store to memory with stride and offset
  TileStoreStrided :: String      -- destination name
                   -> Int         -- offset
                   -> Int         -- stride
                   -> TileExpr a
                   -> TileExpr ()

-- Manual Show instance (can't derive for functions)
instance Show (TileExpr a) where
  show (TileVar name) = "TileVar " ++ show name
  show (TileVarStrided name offset stride) =
    "TileVarStrided " ++ show name ++
    " offset=" ++ show offset ++
    " stride=" ++ show stride
  show (TileConst val) = "TileConst <value>"
  show (TileMap _ expr) = "TileMap <fn> " ++ show expr
  show (TileZipWith _ a b) = "TileZipWith <fn> " ++ show a ++ " " ++ show b
  show (TileFold _ init expr) = "TileFold <fn> <init> " ++ show expr
  show (TileScan _ init expr) = "TileScan <fn> <init> " ++ show expr
  show (TileLoad name) = "TileLoad " ++ show name
  show (TileStore name expr) = "TileStore " ++ show name ++ " " ++ show expr
  show (TileStoreStrided name offset stride expr) =
    "TileStoreStrided " ++ show name ++
    " offset=" ++ show offset ++
    " stride=" ++ show stride ++
    " " ++ show expr

-- | Scalar expression (operations on individual elements)
data ScalarExpr a where
  -- Literals
  ScalarLit :: a -> ScalarExpr a

  -- Variable
  ScalarVar :: String -> ScalarExpr a

  -- Arithmetic
  Add :: Num a => ScalarExpr a -> ScalarExpr a -> ScalarExpr a
  Sub :: Num a => ScalarExpr a -> ScalarExpr a -> ScalarExpr a
  Mul :: Num a => ScalarExpr a -> ScalarExpr a -> ScalarExpr a

  -- Comparison
  Lt :: Ord a => ScalarExpr a -> ScalarExpr a -> ScalarExpr Bool
  Gt :: Ord a => ScalarExpr a -> ScalarExpr a -> ScalarExpr Bool
  Eq :: Eq a => ScalarExpr a -> ScalarExpr a -> ScalarExpr Bool

-- Manual Show instance
instance Show (ScalarExpr a) where
  show (ScalarLit _) = "ScalarLit <value>"
  show (ScalarVar name) = "ScalarVar " ++ show name
  show (Add a b) = "Add (" ++ show a ++ ") (" ++ show b ++ ")"
  show (Sub a b) = "Sub (" ++ show a ++ ") (" ++ show b ++ ")"
  show (Mul a b) = "Mul (" ++ show a ++ ") (" ++ show b ++ ")"
  show (Lt a b) = "Lt (" ++ show a ++ ") (" ++ show b ++ ")"
  show (Gt a b) = "Gt (" ++ show a ++ ") (" ++ show b ++ ")"
  show (Eq a b) = "Eq (" ++ show a ++ ") (" ++ show b ++ ")"

-- | Tile kernel - function from tiles to tiles
data TileKernel where
  TileKernel1 :: KnownNat n
              => String                    -- kernel name
              -> (Tile n a -> Tile n b)    -- tile function
              -> TileKernel

  TileKernel2 :: KnownNat n
              => String                            -- kernel name
              -> (Tile n a -> Tile n b -> Tile n c) -- tile function
              -> TileKernel

  TileKernel3 :: KnownNat n
              => String                                    -- kernel name
              -> (Tile n a -> Tile n b -> Tile n c -> Tile n d) -- tile function
              -> TileKernel

-- | Check if an expression contains a fold operation
containsFold :: TileExpr a -> Bool
containsFold (TileVar _) = False
containsFold (TileVarStrided _ _ _) = False
containsFold (TileConst _) = False
containsFold (TileMap _ expr) = containsFold expr
containsFold (TileZipWith _ exprA exprB) = containsFold exprA || containsFold exprB
containsFold (TileFold _ _ _) = True
containsFold (TileScan _ _ _) = False  -- Scan doesn't need multi-phase (yet)
containsFold (TileLoad _) = False
containsFold (TileStore _ expr) = containsFold expr
containsFold (TileStoreStrided _ _ _ expr) = containsFold expr

-- | Check if an expression contains a scan operation
containsScan :: TileExpr a -> Bool
containsScan (TileVar _) = False
containsScan (TileVarStrided _ _ _) = False
containsScan (TileConst _) = False
containsScan (TileMap _ expr) = containsScan expr
containsScan (TileZipWith _ exprA exprB) = containsScan exprA || containsScan exprB
containsScan (TileFold _ _ _) = False
containsScan (TileScan _ _ _) = True
containsScan (TileLoad _) = False
containsScan (TileStore _ expr) = containsScan expr
containsScan (TileStoreStrided _ _ _ expr) = containsScan expr

-- Manual Show instance
instance Show TileKernel where
  show (TileKernel1 name _) = "TileKernel1 " ++ show name
  show (TileKernel2 name _) = "TileKernel2 " ++ show name
  show (TileKernel3 name _) = "TileKernel3 " ++ show name

-- | Runtime type information for code generation
data TileType = TF32 | TF64 | TI32 | TI64 | TBool
  deriving (Show, Eq)

-- | Get type info from Haskell type
class TileTypeable a where
  tileType :: Proxy a -> TileType

instance TileTypeable Float where tileType _ = TF32
instance TileTypeable Double where tileType _ = TF64
instance TileTypeable Int where tileType _ = TI32
instance TileTypeable Bool where tileType _ = TBool
