#!/usr/bin/env python3
"""
Dump bytecode from cutile-python, then convert to MLIR
"""

import os
import glob

# Enable bytecode dumping
os.environ['CUDA_TILE_DUMP_BYTECODE'] = '/tmp/cutile_bytecode/'

import cuda.tile as ct
import cupy as cp

TILE_SIZE = 16

@ct.kernel
def vector_add(a, b, result):
    """Vector addition"""
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

@ct.kernel
def vector_mul(a, b, result):
    """Vector multiplication"""
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile * b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

print("=== Dumping cutile-python bytecode ===\n")

# Create directory
os.makedirs('/tmp/cutile_bytecode/', exist_ok=True)

# Run kernels to trigger compilation
a = cp.arange(32, dtype=cp.float32)
b = cp.arange(32, dtype=cp.float32)
result = cp.zeros(32, dtype=cp.float32)
grid = (ct.cdiv(32, TILE_SIZE), 1, 1)

print("1. Compiling vector_add...")
ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (a, b, result))
print(f"   ✅ Result: {result[:4]}")

print("\n2. Compiling vector_mul...")
ct.launch(cp.cuda.get_current_stream(), grid, vector_mul, (a, b, result))
print(f"   ✅ Result: {result[:4]}")

print("\n3. Bytecode files generated:")
bc_files = glob.glob('/tmp/cutile_bytecode/*.tilebc')
for f in bc_files:
    size = os.path.getsize(f)
    print(f"   {os.path.basename(f)} ({size} bytes)")

print(f"\n✅ Found {len(bc_files)} bytecode files")
