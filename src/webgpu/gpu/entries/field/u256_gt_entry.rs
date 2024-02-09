use crate::webgpu::gpu::{
    entries::entry_creator::batched_entry, u32_sizes::U256_SIZE, utils::GpuU32Inputs,
    wgsl::U256_WGSL,
};

async fn u256_gt(
    input1: GpuU32Inputs,
    input2: GpuU32Inputs,
    batch_size: Option<usize>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let shader_entry = format!(
        r#"@group(0) @binding(0)
var<storage, read> input1: array<u256>;
@group(0) @binding(1)
var<storage, read> input2: array<u256>;
@group(0) @binding(2)
var<storage, read_write> output: array<u256>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{
    var gt_result = gt(input1[global_id.x], input2[global_id.x]);
    var result_as_uint_256: u256 = u256(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0));
    if (gt_result) {{
        result_as_uint_256.components[7u] = 1u;
    }}
    output[global_id.x] = result_as_uint_256;
}}"#
    );

    let shader_modules = [U256_WGSL, &shader_entry].join("\n\n");

    batched_entry(
        vec![input1, input2],
        &shader_modules,
        U256_SIZE as usize,
        batch_size,
        None,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::webgpu::gpu::utils::{big_int_to_u32_array, u32_array_to_bigints};
    use num_bigint::BigInt;
    use tokio::test as async_test; // Adjust according to your async runtime if not using Tokio.

    #[async_test]
    async fn test_u256_gt() {
        let input1 = GpuU32Inputs {
            u32_inputs: big_int_to_u32_array(&BigInt::from(10)),
            individual_input_size: U256_SIZE as usize,
        };
        let input2 = GpuU32Inputs {
            u32_inputs: big_int_to_u32_array(&BigInt::from(5)),
            individual_input_size: U256_SIZE as usize,
        };

        let result = u256_gt(input1, input2, Some(1)).await.unwrap();
        let big_int_result = u32_array_to_bigints(&result);
        assert_eq!(big_int_result[0], BigInt::from(1));
    }
}
