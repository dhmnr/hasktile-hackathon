#!/usr/bin/env python3
"""
Validate and compile TileIR MLIR from Python
"""

from cuda_tile._mlir._mlir_libs._cuda_tile import (
    TileType,
    PointerType,
    writeBytecode,
    register_dialect,
)
from cuda_tile._mlir.ir import Context, Location, Module
from cuda_tile._mlir.extras import types as T
import tempfile
import os

def validate_mlir(mlir_text: str) -> bool:
    """Parse and validate MLIR"""
    try:
        with Context() as ctx:
            register_dialect(ctx, load=True)
            with Location.unknown(ctx):
                module = Module.parse(mlir_text)
            print("✅ MLIR is valid!")
            return True
    except Exception as e:
        print(f"❌ MLIR validation failed: {e}")
        return False

def compile_to_bytecode(mlir_text: str, output_path: str) -> bool:
    """Compile MLIR to TileIR bytecode"""
    try:
        with Context() as ctx:
            register_dialect(ctx, load=True)
            with Location.unknown(ctx):
                module = Module.parse(mlir_text)

            with open(output_path, 'wb') as f:
                result = writeBytecode(f, module.operation)
                if result:
                    print(f"✅ Bytecode written to {output_path}")
                    print(f"   Size: {os.path.getsize(output_path)} bytes")
                    return True
                else:
                    print("❌ Failed to write bytecode")
                    return False
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return False

def inspect_types():
    """Show how to work with TileIR types"""
    with Context() as ctx:
        register_dialect(ctx, load=True)

        # Create types programmatically
        tile_type = TileType.get([16], T.f32())
        ptr_type = PointerType.get(T.f32())

        print(f"Tile type: {tile_type}")      # !cuda_tile.tile<16xf32>
        print(f"Pointer type: {ptr_type}")    # !cuda_tile.ptr<f32>

if __name__ == "__main__":
    print("=== TileIR Python Bindings Demo ===\n")

    # Example MLIR from Haskell EDSL
    mlir_code = """
    module {
        cuda_tile.module @kernels {
            cuda_tile.entry @vector_add(
                %ptr_a: !cuda_tile.tile<ptr<f32>>,
                %ptr_b: !cuda_tile.tile<ptr<f32>>,
                %ptr_out: !cuda_tile.tile<ptr<f32>>
            ) {
                cuda_tile.return
            }
        }
    }
    """

    print("1. Validating MLIR...")
    validate_mlir(mlir_code)

    print("\n2. Compiling to bytecode...")
    compile_to_bytecode(mlir_code, "/tmp/test.tilebc")

    print("\n3. Type system demo...")
    inspect_types()
