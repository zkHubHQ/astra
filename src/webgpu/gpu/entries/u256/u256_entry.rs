use crate::webgpu::gpu::{
    entries::entry_creator::batched_entry, u32_sizes::U256_SIZE, utils::GpuU32Inputs,
    wgsl::U256_WGSL,
};

async fn u256_entry(
    wgsl_function: &str,
    inputs: Vec<GpuU32Inputs>,
    batch_size: Option<usize>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut input_bindings = String::new();
    let mut args = String::new();

    for i in 0..inputs.len() {
        input_bindings += &format!(
            r#"@group(0) @binding({i})
            var<storage, read> input{i}: array<u256>;
            "#,
        );
        args += &format!("input{i}[global_id.x],");
    }
    // Drop end comma from args
    let args = args.trim_end_matches(',');

    let output_bindings = format!(
        r#"@group(0) @binding({})
        var<storage, read_write> output: array<u256>;
        "#,
        inputs.len()
    );

    let shader_entry = format!(
        r#"{}{}
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{
            var result = {}({});
            output[global_id.x] = result;
        }}"#,
        input_bindings, output_bindings, wgsl_function, args
    );

    let shader_modules = [U256_WGSL, &shader_entry].join("\n\n");

    batched_entry(
        inputs,
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
    async fn test_u256_entry_add() {
        // Example test case, assuming a specific WGSL function and inputs
        let wgsl_function = "u256_add";
        let inputs = vec![
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(2)),
                individual_input_size: U256_SIZE as usize,
            },
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(4)),
                individual_input_size: U256_SIZE as usize,
            },
        ];

        let result = u256_entry(wgsl_function, inputs, Some(1)).await.unwrap();
        let big_int_result = u32_array_to_bigints(&result);
        assert_eq!(big_int_result[0], BigInt::from(6));
    }

    #[async_test]
    async fn test_u256_entry_double() {
        // Example test case, assuming a specific WGSL function and inputs
        let wgsl_function = "u256_double";
        let inputs = vec![GpuU32Inputs {
            u32_inputs: big_int_to_u32_array(&BigInt::from(2)),
            individual_input_size: U256_SIZE as usize,
        }];

        let result = u256_entry(wgsl_function, inputs, Some(1)).await.unwrap();
        let big_int_result = u32_array_to_bigints(&result);
        assert_eq!(big_int_result[0], BigInt::from(4));
    }

    #[async_test]
    async fn test_u256_entry_sub() {
        // Example test case, assuming a specific WGSL function and inputs
        let wgsl_function = "u256_sub";
        let inputs = vec![
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(6)),
                individual_input_size: U256_SIZE as usize,
            },
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(4)),
                individual_input_size: U256_SIZE as usize,
            },
        ];

        let result = u256_entry(wgsl_function, inputs, Some(1)).await.unwrap();
        let big_int_result = u32_array_to_bigints(&result);
        assert_eq!(big_int_result[0], BigInt::from(2));
    }

    #[async_test]
    async fn test_u256_entry_subw() {
        // Example test case, assuming a specific WGSL function and inputs
        let wgsl_function = "u256_subw";
        let inputs = vec![
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(6)),
                individual_input_size: U256_SIZE as usize,
            },
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(4)),
                individual_input_size: U256_SIZE as usize,
            },
        ];

        let result = u256_entry(wgsl_function, inputs, Some(1)).await.unwrap();
        let big_int_result = u32_array_to_bigints(&result);
        assert_eq!(big_int_result[0], BigInt::from(2));
    }
}
