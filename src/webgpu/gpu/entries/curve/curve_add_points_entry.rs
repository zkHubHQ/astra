use crate::webgpu::gpu::{
    curve_specific::{get_curve_base_functions_wgsl, get_curve_params_wgsl, CurveType},
    entries::entry_creator::batched_entry,
    prune::prune,
    u32_sizes::FIELD_SIZE,
    utils::GpuU32Inputs,
    wgsl::{CURVE_WGSL, FIELD_MODULUS_WGSL, U256_WGSL},
};

async fn point_add(
    curve: CurveType,
    points_a: GpuU32Inputs,
    points_b: GpuU32Inputs,
    batch_size: Option<usize>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let shader_entry = format!(
        r#"@group(0) @binding(0)
        var<storage, read> input1: array<AffinePoint>;
        @group(0) @binding(1)
        var<storage, read> input2: array<AffinePoint>;
        @group(0) @binding(2)
        var<storage, read_write> output: array<Field>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{
            var p1 = input1[global_id.x];
            var p1_t = field_multiply(p1.x, p1.y);
            var p2 = input2[global_id.x];
            var p2_t = field_multiply(p2.x, p2.y);
            var z = U256_ONE;
            var ext_p1 = Point(p1.x, p1.y, p1_t, z);
            var ext_p2 = Point(p2.x, p2.y, p2_t, z);

            var added = add_points(ext_p1, ext_p2);
            var x_normalized = normalize_x(added.x, added.z);

            output[global_id.x] = x_normalized;
        }}"#
    );

    let curve_params = get_curve_params_wgsl(curve);
    let curve_base_functions = get_curve_base_functions_wgsl(curve);

    let shader_modules = [
        U256_WGSL,
        &curve_params,
        FIELD_MODULUS_WGSL,
        &curve_base_functions,
        CURVE_WGSL,
    ];
    let shader_code = prune(
        &shader_modules.join("\n"),
        &["field_multiply", "add_points", "normalize_x"],
    ) + &shader_entry;

    batched_entry(
        vec![points_a, points_b],
        &shader_code,
        FIELD_SIZE as usize,
        batch_size,
        None,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bn256::G1Affine,
        serde::SerdeObject,
        webgpu::gpu::{
            u32_sizes::AFFINE_POINT_SIZE,
            utils::{
                big_int_to_u32_array, concatenate_vectors, convert_bn256_curve_to_u32_array,
                convert_u32_array_to_bn256_curve, generate_random_affine_point,
                generate_random_scalars,
            },
        },
    };
    use ff::{PrimeField, PrimeFieldBits};
    use group::Curve;
    use num_bigint::BigInt;
    use pasta_curves::arithmetic::CurveAffine;
    use tokio::test as async_test;

    #[async_test]
    #[ignore]
    async fn test_point_add() {
        let input_affine_point1 = generate_random_affine_point::<G1Affine>();
        let input_affine_point2 = generate_random_affine_point::<G1Affine>();
        let expected_result = (input_affine_point1 + input_affine_point2).to_affine();

        let point1_u32_array = convert_bn256_curve_to_u32_array(&input_affine_point1);
        let point2_u32_array = convert_bn256_curve_to_u32_array(&input_affine_point2);

        let points_a = GpuU32Inputs {
            u32_inputs: concatenate_vectors(
                &point1_u32_array.x_u32_array,
                &point1_u32_array.y_u32_array,
            ),
            individual_input_size: AFFINE_POINT_SIZE as usize,
        };
        let points_b = GpuU32Inputs {
            u32_inputs: concatenate_vectors(
                &point2_u32_array.x_u32_array,
                &point2_u32_array.y_u32_array,
            ),
            individual_input_size: AFFINE_POINT_SIZE as usize,
        };
        let curve = CurveType::BN254;

        // Print the inputs
        println!("points_a: {:?}", points_a);
        println!("points_b: {:?}", points_b);

        println!("Starting point_add");
        let result_x = point_add(curve, points_a, points_b, Some(1)).await.unwrap();
        println!("result_x: {:?}", result_x);

        let mut x_bytes = [0u8; 32];
        for (i, chunk) in result_x.iter().rev().enumerate() {
            let chunk_bytes = &chunk.to_le_bytes();
            x_bytes[i * 4..(i + 1) * 4].copy_from_slice(chunk_bytes);
        }
        let actual_x_result = crate::bn256::Fq::from_repr(x_bytes).unwrap();
        assert_eq!(expected_result.x, actual_x_result);
    }

    #[async_test]
    async fn test_create_point() {
        let curve_affine_point = generate_random_affine_point::<G1Affine>();
        println!("curve point: {:?}", curve_affine_point);

        let curve_u32_array = convert_bn256_curve_to_u32_array(&curve_affine_point);
        println!("curve point 1 as u32 array: {:?}", curve_u32_array);

        let original_curve_point = convert_u32_array_to_bn256_curve(&curve_u32_array);
        println!("original curve point: {:?}", original_curve_point);
    }
}
