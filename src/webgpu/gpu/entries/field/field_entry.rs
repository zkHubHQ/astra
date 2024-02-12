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
            r#"@group(0) @binding({i})
            var<storage, read> input{i}: array<Field>;
            "#,
        ));
        args.push_str(&format!("input{i}[global_id.x],"));
    }
    // Drop end comma from args
    let args = args.trim_end_matches(',');

    let output_bindings = format!(
        r#"@group(0) @binding({})
        var<storage, read_write> output: array<Field>;
        "#,
        inputs.len()
    );

    let shader_entry = format!(
        r#"{input_bindings}{output_bindings}
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{
            var result = {wgsl_function}({args});
            output[global_id.x] = result;
        }}"#,
    );

    let curve_params_wgsl = get_curve_params_wgsl(curve);
    let shader_modules = [
        U256_WGSL,
        curve_params_wgsl,
        FIELD_MODULUS_WGSL,
        &shader_entry,
    ]
    .join("\n\n");

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
    use crate::{bn256::G1Affine, webgpu::gpu::utils::{big_int_to_u32_array, convert_bn256_scalar_to_u32_array, convert_u32_array_to_bn256_scalar, generate_random_scalar_point, u32_array_to_bigints}};

    use super::*;
    use group::prime::PrimeCurveAffine;
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
        let curve = CurveType::BN254; // Example, adjust as per your curve

        let result = field_entry(wgsl_function, curve, inputs, Some(2))
            .await
            .unwrap();

        // Convert the result back to a BigInt for comparison
        let big_int_result = u32_array_to_bigints(&result); // Assuming a similar utility function exists

        // Now, assert the result matches your expected value
        assert_eq!(big_int_result[0], BigInt::from(6)); // Example assertion, adjust as per your test case
    }

    #[async_test]
    async fn test_u256_add_actual_fields() {
        let scalar_point1 = generate_random_scalar_point::<G1Affine>();
        let scalar_point2 = generate_random_scalar_point::<G1Affine>();

        let scalar_point1_u32_array = convert_bn256_scalar_to_u32_array(&scalar_point1);
        let scalar_point2_u32_array = convert_bn256_scalar_to_u32_array(&scalar_point2);

        let expected_result = scalar_point1 + scalar_point2;

        let inputs = vec![
            GpuU32Inputs {
                u32_inputs: scalar_point1_u32_array.u32_array,
                individual_input_size: FIELD_SIZE as usize,
            },
            GpuU32Inputs {
                u32_inputs: scalar_point2_u32_array.u32_array,
                individual_input_size: FIELD_SIZE as usize,
            },
        ];

        let wgsl_function = "field_add";
        let curve = CurveType::BN254;

        let result = field_entry(wgsl_function, curve, inputs, Some(2))
            .await
            .unwrap();

        let actual_result = convert_u32_array_to_bn256_scalar(&result);

        assert_eq!(expected_result, actual_result);
    }

    #[async_test]
    async fn test_u256_mod_exp() {
        let inputs = vec![
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(2)),
                individual_input_size: FIELD_SIZE as usize,
            },
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(4)),
                individual_input_size: FIELD_SIZE as usize,
            },
        ];

        let wgsl_function = "field_pow";
        let curve = CurveType::BN254;

        let result = field_entry(wgsl_function, curve, inputs, Some(2))
            .await
            .unwrap();

        let big_int_result = u32_array_to_bigints(&result);

        assert_eq!(big_int_result[0], BigInt::from(16));
    }

    #[async_test]
    async fn test_u256_multiply() {
        let inputs = vec![
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(2)),
                individual_input_size: FIELD_SIZE as usize,
            },
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(4)),
                individual_input_size: FIELD_SIZE as usize,
            },
        ];

        let wgsl_function = "field_multiply";
        let curve = CurveType::BN254;

        let result = field_entry(wgsl_function, curve, inputs, Some(2))
            .await
            .unwrap();

        let big_int_result = u32_array_to_bigints(&result);

        assert_eq!(big_int_result[0], BigInt::from(8));
    }

    #[async_test]
    async fn test_u256_sqrt() {
        let inputs = vec![
            GpuU32Inputs {
                u32_inputs: big_int_to_u32_array(&BigInt::from(81)),
                individual_input_size: FIELD_SIZE as usize,
            },
        ];

        let wgsl_function = "field_sqrt";
        let curve = CurveType::BN254;

        let result = field_entry(wgsl_function, curve, inputs, Some(2))
            .await
            .unwrap();

        let big_int_result = u32_array_to_bigints(&result);

        assert_eq!(big_int_result[0], BigInt::from(9));
    }
}
