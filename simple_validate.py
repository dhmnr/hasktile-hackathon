#!/usr/bin/env python3
"""
Simple MLIR validation using cuda-tile-translate
No Python bindings needed!
"""

import subprocess
import tempfile
import os

def validate_mlir(mlir_text: str) -> bool:
    """Validate MLIR by trying to compile it"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_text)
        mlir_path = f.name

    try:
        # Try to compile it
        result = subprocess.run(
            ['/home/shadeform/cuda-tile/build/bin/cuda-tile-translate',
             mlir_path,
             '--mlir-to-cudatilebc',
             '--no-implicit-module',
             '--bytecode-version=13.1',
             '-o', '/tmp/test.tilebc'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ MLIR is valid!")
            print(f"✅ Compiled to bytecode: {os.path.getsize('/tmp/test.tilebc')} bytes")
            return True
        else:
            print("❌ MLIR validation failed:")
            print(result.stderr)
            return False
    finally:
        os.unlink(mlir_path)

if __name__ == "__main__":
    # Test with our Haskell-generated MLIR
    mlir_code = """cuda_tile.module @kernels {
  entry @vector_mul(
    %ptr_a: !cuda_tile.tile<ptr<f32>>,
    %ptr_b: !cuda_tile.tile<ptr<f32>>,
    %ptr_out: !cuda_tile.tile<ptr<f32>>
  ) {
    %offset = iota : !cuda_tile.tile<16xi32>

    %a_base = reshape %ptr_a : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>
    %a_bcast = broadcast %a_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<16xptr<f32>>
    %a_ptrs = offset %a_bcast, %offset : !cuda_tile.tile<16xptr<f32>>, !cuda_tile.tile<16xi32> -> !cuda_tile.tile<16xptr<f32>>

    %a_val, %tok_a = load_ptr_tko weak %a_ptrs : !cuda_tile.tile<16xptr<f32>> -> !cuda_tile.tile<16xf32>, !cuda_tile.token

    %b_base = reshape %ptr_b : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>
    %b_bcast = broadcast %b_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<16xptr<f32>>
    %b_ptrs = offset %b_bcast, %offset : !cuda_tile.tile<16xptr<f32>>, !cuda_tile.tile<16xi32> -> !cuda_tile.tile<16xptr<f32>>

    %b_val, %tok_b = load_ptr_tko weak %b_ptrs : !cuda_tile.tile<16xptr<f32>> -> !cuda_tile.tile<16xf32>, !cuda_tile.token

    %out_base = reshape %ptr_out : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>
    %out_bcast = broadcast %out_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<16xptr<f32>>
    %out_ptrs = offset %out_bcast, %offset : !cuda_tile.tile<16xptr<f32>>, !cuda_tile.tile<16xi32> -> !cuda_tile.tile<16xptr<f32>>

    %v0 = mulf %a_val, %b_val rounding<nearest_even> : !cuda_tile.tile<16xf32>
    %tok_out = store_ptr_tko weak %out_ptrs, %v0 : !cuda_tile.tile<16xptr<f32>>, !cuda_tile.tile<16xf32> -> !cuda_tile.token

    return
  }
}"""

    print("=== Validating Haskell-generated MLIR ===\n")
    validate_mlir(mlir_code)
