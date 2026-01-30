#!/usr/bin/env python3
"""
Try different debug approaches for cutile
"""

import os
import sys

# Try various debug environment variables
os.environ['CUDA_TILE_DEBUG'] = '1'
os.environ['TILE_IR_VERBOSE'] = '1'
os.environ['NVVM_DEBUG'] = '1'

print("=== Testing with debug environment variables ===\n")

import cuda.tile as ct
import cupy as cp

TILE_SIZE = 16

@ct.kernel
def vector_add(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

print("Creating test data...")
a = cp.arange(32, dtype=cp.float32)
b = cp.arange(32, dtype=cp.float32)
result = cp.zeros(32, dtype=cp.float32)

print("\nLaunching kernel (watch for debug output)...")
grid = (ct.cdiv(32, TILE_SIZE), 1, 1)

try:
    ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (a, b, result))
    print(f"\n✅ Kernel executed successfully")
    print(f"Result sample: {result[:8]}")
except Exception as e:
    print(f"❌ Error: {e}")
