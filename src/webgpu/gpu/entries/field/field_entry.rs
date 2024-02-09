use crate::webgpu::gpu::{
    curve_specific::{get_curve_params_wgsl, CurveType},
    entries::entry_creator::batched_entry,
    u32_sizes::FIELD_SIZE,
    utils::GpuU32Inputs,
    wgsl::{FIELD_MODULUS_WGSL, U256_WGSL},
};

async fn field_entry(
    wgsl_function: &str,
    curve: CurveType,
    inputs: Vec<GpuU32Inputs>,
    batch_size: Option<usize>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut input_bindings = String::new();
    let mut args = String::new();
    for i in 0..inputs.len() {
        input_bindings.push_str(&format!(
            "@group(0) @binding({i})\nvar<storage, read> input{i}: array<Field>;\n"
        ));
        args.push_str(&format!("input{i}[global_id.x],"));
    }
    // Drop end comma from args
    let args = args.trim_end_matches(',');

    let output_bindings = format!(
        "@group(0) @binding({})\nvar<storage, read_write> output: array<Field>;\n",
        inputs.len()
    );

    let shader_entry = format!(
        "{input_bindings}\n{output_bindings}\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{\nvar result = {wgsl_function}({args});\noutput[global_id.x] = result;\n}}",
    );

    let curve_params_wgsl = get_curve_params_wgsl(curve);
    let shader_modules = [
        U256_WGSL,
        curve_params_wgsl,
        FIELD_MODULUS_WGSL,
        shader_entry.as_str(),
    ]
    .join("\n");

    batched_entry(
        inputs,
        &shader_modules,
        FIELD_SIZE as usize,
        batch_size,
        None,
    )
    .await
}

#[cfg(test)]
mod tests {
    use crate::webgpu::gpu::utils::{big_int_to_u32_array, u32_array_to_bigints};

    use super::*;
    use num_bigint::BigInt;
    use tokio::test as async_test; // Use this line if you're using Tokio. Otherwise, adjust according to your async runtime.

    #[async_test]
    async fn test_u256_add() {
        let inputs = vec![
            // Convert your BigInt inputs to `GpuU32Inputs` here.
            // Assuming you have a utility function similar to bigIntToU32Array for the conversion:
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(2)),
                individual_input_size: FIELD_SIZE as usize,
            },
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(4)),
                individual_input_size: FIELD_SIZE as usize,
            },
        ];

        let wgsl_function = "field_add"; // Adjust based on your actual WGSL function
        let curve = CurveType::BLS12_377; // Example, adjust as per your curve

        let result = field_entry(wgsl_function, curve, inputs, Some(2))
            .await
            .unwrap();

        // Convert the result back to a BigInt for comparison
        let big_int_result = u32_array_to_bigints(&result); // Assuming a similar utility function exists

        // Now, assert the result matches your expected value
        assert_eq!(big_int_result[0], BigInt::from(6)); // Example assertion, adjust as per your test case
    }
}
