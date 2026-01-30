#!/usr/bin/env python3
"""
Single-block vector add in cutile-python
Matches the Haskell EDSL pattern (no grid indexing)
"""

import os
import cupy as cp
import cuda.tile as ct

# Enable bytecode dumping
os.environ['CUDA_TILE_DUMP_BYTECODE'] = '/home/shadeform/workspace/hasktile/python/bytecode/'

TILE_SIZE = 16

@ct.kernel
def vector_add_single_block(a, b, result):
    """
    Single block vector add - processes exactly 16 elements
    No block ID, just like our Haskell EDSL
    """
    # Don't use block_id - just process tile at index 0
    a_tile = ct.load(a, index=(0,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(0,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(0,), tile=result_tile)

@ct.kernel
def vector_mul_single_block(a, b, result):
    """
    Single block vector multiply - processes exactly 16 elements
    """
    a_tile = ct.load(a, index=(0,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(0,), shape=(TILE_SIZE,))
    result_tile = a_tile * b_tile
    ct.store(result, index=(0,), tile=result_tile)

if __name__ == "__main__":
    print("=== Single Block Vector Operations (cutile-python) ===\n")

    # Create directory for bytecode
    os.makedirs('/home/shadeform/workspace/hasktile/python/bytecode/', exist_ok=True)

    # Create test data (only 16 elements - single tile)
    a = cp.arange(16, dtype=cp.float32)
    b = cp.arange(16, dtype=cp.float32) + 100
    result = cp.zeros(16, dtype=cp.float32)

    print("Input:")
    print(f"  a = {a}")
    print(f"  b = {b}")

    # Launch with single block
    grid = (1, 1, 1)  # Only 1 block!

    print("\n1. Testing vector_add_single_block...")
    ct.launch(cp.cuda.get_current_stream(), grid, vector_add_single_block, (a, b, result))
    print(f"   Result: {result}")
    print(f"   Expected: a + b = {a.get() + b.get()}")
    print(f"   ✅ Match: {cp.allclose(result, a + b)}")

    print("\n2. Testing vector_mul_single_block...")
    result = cp.zeros(16, dtype=cp.float32)
    ct.launch(cp.cuda.get_current_stream(), grid, vector_mul_single_block, (a, b, result))
    print(f"   Result: {result}")
    print(f"   Expected: a * b = {(a.get() * b.get())[:4]}...")
    print(f"   ✅ Match: {cp.allclose(result, a * b)}")

    print("\n3. Bytecode generated:")
    import glob
    bc_files = glob.glob('/home/shadeform/workspace/hasktile/python/bytecode/*.cutile')
    for f in bc_files:
        print(f"   {os.path.basename(f)}")
