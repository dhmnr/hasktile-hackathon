#!/usr/bin/env python3
"""
Dump MLIR from cutile-python
"""

import os

os.environ['CUDA_TILE_DUMP_TILEIR'] = '/tmp/cutile_dumps'

import cuda.tile as ct
import cupy as cp

TILE_SIZE = 16

@ct.kernel
def vector_add(a, b, result):
    """Vector addition - compare with Haskell version"""
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

@ct.kernel
def vector_mul(a, b, result):
    """Vector multiplication - compare with Haskell version"""
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile * b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

print("=== Compiling cutile-python kernels ===\n")

# Create test data and launch
a = cp.arange(32, dtype=cp.float32)
b = cp.arange(32, dtype=cp.float32)
result = cp.zeros(32, dtype=cp.float32)
grid = (ct.cdiv(32, TILE_SIZE), 1, 1)

print("1. Launching vector_add...")
ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (a, b, result))
print(f"   Result: {result[:4]} ... (should be [0, 2, 4, 6])")

print("\n2. Launching vector_mul...")
ct.launch(cp.cuda.get_current_stream(), grid, vector_mul, (a, b, result))
print(f"   Result: {result[:4]} ... (should be [0, 1, 4, 9])")

print("\n3. Generated MLIR files:")
import glob
mlir_files = glob.glob('/tmp/cutile_dumps/*.mlir')
for f in mlir_files:
    print(f"   {f}")
