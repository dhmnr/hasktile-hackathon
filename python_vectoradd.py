#!/usr/bin/env python3
"""
Python MLIR generator for TileIR vector add
Compare this with the Haskell EDSL approach
"""

def generate_vector_add_mlir(tile_size=16):
    """Generate TileIR MLIR for vector addition"""
    return f"""cuda_tile.module @kernels {{
  entry @vector_add(
    %ptr_a: !cuda_tile.tile<ptr<f32>>,
    %ptr_b: !cuda_tile.tile<ptr<f32>>,
    %ptr_out: !cuda_tile.tile<ptr<f32>>
  ) {{
    // Create offset tensor (0, 1, 2, ..., {tile_size-1})
    %offset = iota : !cuda_tile.tile<{tile_size}xi32>

    // Prepare pointer A
    %a_base = reshape %ptr_a : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>
    %a_bcast = broadcast %a_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<{tile_size}xptr<f32>>
    %a_final = offset %a_bcast, %offset : !cuda_tile.tile<{tile_size}xptr<f32>>, !cuda_tile.tile<{tile_size}xi32> -> !cuda_tile.tile<{tile_size}xptr<f32>>

    // Prepare pointer B
    %b_base = reshape %ptr_b : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>
    %b_bcast = broadcast %b_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<{tile_size}xptr<f32>>
    %b_final = offset %b_bcast, %offset : !cuda_tile.tile<{tile_size}xptr<f32>>, !cuda_tile.tile<{tile_size}xi32> -> !cuda_tile.tile<{tile_size}xptr<f32>>

    // Prepare pointer OUT
    %c_base = reshape %ptr_out : !cuda_tile.tile<ptr<f32>> -> !cuda_tile.tile<1xptr<f32>>
    %c_bcast = broadcast %c_base : !cuda_tile.tile<1xptr<f32>> -> !cuda_tile.tile<{tile_size}xptr<f32>>
    %c_final = offset %c_bcast, %offset : !cuda_tile.tile<{tile_size}xptr<f32>>, !cuda_tile.tile<{tile_size}xi32> -> !cuda_tile.tile<{tile_size}xptr<f32>>

    // Load, compute, store
    %a_val, %tok_a = load_ptr_tko weak %a_final : !cuda_tile.tile<{tile_size}xptr<f32>> -> !cuda_tile.tile<{tile_size}xf32>, !cuda_tile.token
    %b_val, %tok_b = load_ptr_tko weak %b_final : !cuda_tile.tile<{tile_size}xptr<f32>> -> !cuda_tile.tile<{tile_size}xf32>, !cuda_tile.token
    %result = addf %a_val, %b_val rounding<nearest_even> : !cuda_tile.tile<{tile_size}xf32>
    %tok_out = store_ptr_tko weak %c_final, %result : !cuda_tile.tile<{tile_size}xptr<f32>>, !cuda_tile.tile<{tile_size}xf32> -> !cuda_tile.token

    return
  }}
}}"""

if __name__ == "__main__":
    print("=== Python-generated TileIR MLIR ===\n")
    mlir = generate_vector_add_mlir(16)
    print(mlir)

    # Save to file
    with open("/home/shadeform/python_generated.mlir", "w") as f:
        f.write(mlir)

    print("\n=== Saved to python_generated.mlir ===")
    print("\nCompare with Haskell approach:")
    print("  Haskell: vectorAdd a b = tileZipWith (.+.) a b")
    print("  Python:  generate_vector_add_mlir(16)")
    print("\nBoth generate the same MLIR!")
