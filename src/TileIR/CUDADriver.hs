{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TileIR.CUDADriver where

import Foreign
import Foreign.C.Types
import Foreign.C.String
import Control.Exception (throwIO, Exception)
import Data.Typeable
import qualified Data.ByteString as BS
import System.IO

-- | CUDA result type
newtype CUresult = CUresult CInt deriving (Eq, Show, Storable)

-- | CUDA device handle
newtype CUdevice = CUdevice CInt deriving (Eq, Show, Storable)

-- | CUDA context (opaque pointer)
newtype CUcontext = CUcontext (Ptr ()) deriving (Eq, Show, Storable)

-- | CUDA module (opaque pointer)
newtype CUmodule = CUmodule (Ptr ()) deriving (Eq, Show, Storable)

-- | CUDA function (opaque pointer)
newtype CUfunction = CUfunction (Ptr ()) deriving (Eq, Show, Storable)

-- | CUDA device pointer
newtype CUdeviceptr = CUdeviceptr CULLong deriving (Eq, Show, Storable)

-- | Success code
pattern CUDA_SUCCESS :: CUresult
pattern CUDA_SUCCESS = CUresult 0

-- CUDA Driver API FFI bindings
foreign import ccall unsafe "cuInit"
  cuInit :: CUInt -> IO CUresult

foreign import ccall unsafe "cuDeviceGet"
  cuDeviceGet :: Ptr CUdevice -> CInt -> IO CUresult

foreign import ccall unsafe "cuDevicePrimaryCtxRetain"
  cuDevicePrimaryCtxRetain :: Ptr CUcontext -> CUdevice -> IO CUresult

foreign import ccall unsafe "cuCtxSetCurrent"
  cuCtxSetCurrent :: CUcontext -> IO CUresult

foreign import ccall unsafe "cuModuleLoadData"
  cuModuleLoadData :: Ptr CUmodule -> Ptr CChar -> IO CUresult

foreign import ccall unsafe "cuModuleGetFunction"
  cuModuleGetFunction :: Ptr CUfunction -> CUmodule -> CString -> IO CUresult

foreign import ccall unsafe "cuMemAlloc_v2"
  cuMemAlloc :: Ptr CUdeviceptr -> CSize -> IO CUresult

foreign import ccall unsafe "cuMemcpyHtoD_v2"
  cuMemcpyHtoD :: CUdeviceptr -> Ptr a -> CSize -> IO CUresult

foreign import ccall unsafe "cuMemcpyDtoH_v2"
  cuMemcpyDtoH :: Ptr a -> CUdeviceptr -> CSize -> IO CUresult

foreign import ccall unsafe "cuMemFree_v2"
  cuMemFree :: CUdeviceptr -> IO CUresult

foreign import ccall unsafe "cuLaunchKernel"
  cuLaunchKernel :: CUfunction
                 -> CUInt -> CUInt -> CUInt  -- grid dims
                 -> CUInt -> CUInt -> CUInt  -- block dims
                 -> CUInt                     -- shared mem bytes
                 -> Ptr ()                    -- stream
                 -> Ptr (Ptr ())              -- kernel params
                 -> Ptr (Ptr ())              -- extra
                 -> IO CUresult

foreign import ccall unsafe "cuCtxSynchronize"
  cuCtxSynchronize :: IO CUresult

-- Exception type for CUDA errors
data CUDAException = CUDAException String CUresult
  deriving (Show, Typeable)

instance Exception CUDAException

-- | Check CUDA result and throw exception on error
checkCUDA :: String -> CUresult -> IO ()
checkCUDA msg result
  | result == CUDA_SUCCESS = return ()
  | otherwise = throwIO $ CUDAException msg result

-- High-level helpers

-- | Initialize CUDA
initCUDA :: IO ()
initCUDA = do
  result <- cuInit 0
  checkCUDA "cuInit" result

-- | Get device handle
getDevice :: Int -> IO CUdevice
getDevice devNum =
  alloca $ \pDevice -> do
    result <- cuDeviceGet pDevice (fromIntegral devNum)
    checkCUDA "cuDeviceGet" result
    peek pDevice

-- | Create and set context
createContext :: CUdevice -> IO CUcontext
createContext device = do
  ctx <- alloca $ \pCtx -> do
    result <- cuDevicePrimaryCtxRetain pCtx device
    checkCUDA "cuDevicePrimaryCtxRetain" result
    peek pCtx
  result <- cuCtxSetCurrent ctx
  checkCUDA "cuCtxSetCurrent" result
  return ctx

-- | Load module from cubin file
loadModule :: FilePath -> IO CUmodule
loadModule cubinPath = do
  cubinData <- BS.readFile cubinPath
  BS.useAsCString cubinData $ \pData ->
    alloca $ \pModule -> do
      result <- cuModuleLoadData pModule pData
      checkCUDA "cuModuleLoadData" result
      peek pModule

-- | Get function from module
getFunction :: CUmodule -> String -> IO CUfunction
getFunction modul name =
  withCString name $ \pName ->
    alloca $ \pFunc -> do
      result <- cuModuleGetFunction pFunc modul pName
      checkCUDA ("cuModuleGetFunction: " ++ name) result
      peek pFunc

-- | Allocate device memory
allocDevice :: Int -> IO CUdeviceptr
allocDevice bytes =
  alloca $ \pPtr -> do
    result <- cuMemAlloc pPtr (fromIntegral bytes)
    checkCUDA "cuMemAlloc" result
    peek pPtr

-- | Copy host to device
copyHtoD :: Storable a => CUdeviceptr -> [a] -> IO ()
copyHtoD dptr xs = do
  let bytes = length xs * sizeOf (head xs)
  withArray xs $ \pHost -> do
    result <- cuMemcpyHtoD dptr pHost (fromIntegral bytes)
    checkCUDA "cuMemcpyHtoD" result

-- | Copy device to host
copyDtoH :: forall a. Storable a => CUdeviceptr -> Int -> IO [a]
copyDtoH dptr count = do
  allocaArray count $ \(pHost :: Ptr a) -> do
    let bytes = count * sizeOf (undefined :: a)
    result <- cuMemcpyDtoH pHost dptr (fromIntegral bytes)
    checkCUDA "cuMemcpyDtoH" result
    peekArray count pHost

-- | Free device memory
freeDevice :: CUdeviceptr -> IO ()
freeDevice dptr = do
  result <- cuMemFree dptr
  checkCUDA "cuMemFree" result

-- | Launch kernel with simple 1D grid
launchKernel1D :: CUfunction -> Int -> [Ptr ()] -> IO ()
launchKernel1D func gridSize params = do
  withArray params $ \pParams -> do
    result <- cuLaunchKernel func
              (fromIntegral gridSize) 1 1  -- grid: gridSize x 1 x 1
              1 1 1                          -- block: 1 x 1 x 1
              0                              -- shared mem
              nullPtr                        -- stream
              pParams                        -- kernel params
              nullPtr                        -- extra
    checkCUDA "cuLaunchKernel" result
  result <- cuCtxSynchronize
  checkCUDA "cuCtxSynchronize" result

-- | Launch kernel with 2D grid
launchKernel2D :: CUfunction -> Int -> Int -> [Ptr ()] -> IO ()
launchKernel2D func gridX gridY params = do
  withArray params $ \pParams -> do
    result <- cuLaunchKernel func
              (fromIntegral gridX) (fromIntegral gridY) 1  -- grid: gridX x gridY x 1
              1 1 1                                          -- block: 1 x 1 x 1
              0                                              -- shared mem
              nullPtr                                        -- stream
              pParams                                        -- kernel params
              nullPtr                                        -- extra
    checkCUDA "cuLaunchKernel" result
  result <- cuCtxSynchronize
  checkCUDA "cuCtxSynchronize" result
