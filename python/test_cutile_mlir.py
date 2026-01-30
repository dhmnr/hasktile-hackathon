#!/usr/bin/env python3
"""
Try to extract MLIR from cutile-python
"""

import cuda.tile as ct
import inspect

print("=== Exploring cuda.tile internals ===\n")

# Check what _ir is
print("1. What is _ir?")
print(f"   Type: {type(ct._ir)}")
print(f"   Dir: {[x for x in dir(ct._ir) if not x.startswith('__')]}")

# Try to create a simple kernel and inspect it
print("\n2. Creating a simple kernel...")

@ct.kernel
def simple_add(a, b, result):
    """Simple vector addition"""
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(16,))
    b_tile = ct.load(b, index=(block_id,), shape=(16,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

print(f"   Kernel type: {type(simple_add)}")
print(f"   Kernel attrs: {[x for x in dir(simple_add) if not x.startswith('__')]}")

# Check if we can access the IR
print("\n3. Looking for IR/MLIR...")
for attr in dir(simple_add):
    if any(x in attr.lower() for x in ['ir', 'mlir', 'code', 'compile', 'src']):
        try:
            val = getattr(simple_add, attr)
            print(f"   {attr}: {type(val)}")
            if callable(val):
                print(f"      (callable)")
            else:
                print(f"      Value: {str(val)[:100]}...")
        except:
            print(f"   {attr}: <error accessing>")

# Check the source
print("\n4. Kernel source:")
try:
    print(inspect.getsource(simple_add))
except:
    print("   Could not get source")

# Try to access compilation details
print("\n5. Checking for compilation artifacts...")
if hasattr(simple_add, '_compiled'):
    print(f"   _compiled: {simple_add._compiled}")
if hasattr(simple_add, '_func'):
    print(f"   _func: {type(simple_add._func)}")
    print(f"   _func dir: {[x for x in dir(simple_add._func) if not x.startswith('__')]}")
