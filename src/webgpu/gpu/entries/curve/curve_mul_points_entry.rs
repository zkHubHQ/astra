use crate::webgpu::gpu::{
    curve_specific::{get_curve_base_functions_wgsl, get_curve_params_wgsl, CurveType},
    entries::entry_creator::batched_entry,
    prune::prune,
    u32_sizes::FIELD_SIZE,
    utils::{GpuU32Inputs},
    wgsl::{CURVE_WGSL, FIELD_MODULUS_WGSL, U256_WGSL},
};

async fn point_mul(
    curve: CurveType,
    points: GpuU32Inputs,
    scalars: GpuU32Inputs,
    batch_size: Option<usize>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let shader_entry = format!(
        r#"
        const ZERO_POINT = Point (U256_ZERO, U256_ONE, U256_ZERO, U256_ZERO);
        const ZERO_AFFINE = AffinePoint (U256_ZERO, U256_ONE);

        @group(0) @binding(0)
        var<storage, read> input1: array<AffinePoint>;
        @group(0) @binding(1)
        var<storage, read> input2: array<Field>;
        @group(0) @binding(2)
        var<storage, read_write> output: array<Field>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{
            var p1 = input1[global_id.x];
            var p1_t = field_multiply(p1.x, p1.y);
            var z = U256_ONE;
            var ext_p1 = Point(p1.x, p1.y, p1_t, z);

            var scalar = input2[global_id.x];

            var multiplied = mul_point(ext_p1, scalar);
            var x_normalized = normalize_x(multiplied.x, multiplied.z);

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
        &[
            "field_multiply",
            "mul_point",
            "field_inverse",
            "normalize_x",
        ],
    ) + &shader_entry;

    batched_entry(
        vec![points, scalars],
        &shader_code,
        FIELD_SIZE as usize,
        batch_size,
        None,
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::ops::Mul;

    use super::*;
    use crate::{
        bn256::{self, G1Affine, G1},
        webgpu::gpu::{
            u32_sizes::AFFINE_POINT_SIZE,
            utils::{
                concatenate_vectors, convert_bn256_fq_to_u32_array,
                convert_bn256_scalar_to_u32_array, convert_hex_string_to_bn256_fq,
                convert_hex_string_to_bn256_fr, generate_random_affine_point,
                generate_random_scalar_point,
            },
        },
    };
    use ff::PrimeField;
    use group::Group;
    use group::{cofactor::CofactorCurveAffine, Curve};
    use pasta_curves::arithmetic::CurveAffine;
    use tokio::test as async_test;

    #[async_test]
    #[ignore]
    async fn test_point_mul() {
        let random_affine_point = generate_random_affine_point::<G1Affine>();
        let random_scalar = generate_random_scalar_point::<G1Affine>();
        let expected_result_x = random_affine_point
            .to_curve()
            .mul(random_scalar)
            .to_affine()
            .x;
        // u32 points input is x and y Fq elements concatenated as a single u32 array
        let x_u32_vec = convert_bn256_fq_to_u32_array(&random_affine_point.x);
        let y_u32_vec = convert_bn256_fq_to_u32_array(&random_affine_point.y);
        let points = GpuU32Inputs {
            u32_inputs: concatenate_vectors(&x_u32_vec, &y_u32_vec),
            individual_input_size: AFFINE_POINT_SIZE as usize,
        };
        let fr_u32_vec = convert_bn256_scalar_to_u32_array(&random_scalar);
        let scalars = GpuU32Inputs {
            u32_inputs: fr_u32_vec.u32_array,
            individual_input_size: FIELD_SIZE as usize,
        };
        let curve = CurveType::BN254;

        let result = point_mul(curve, points, scalars, Some(1)).await.unwrap();
        println!("Result: {:?}", result);
    }

    #[async_test]
    async fn test_repr() {
        let fq_x = convert_hex_string_to_bn256_fq(
            "0x0d015c9f3dbc23ea97c59aca583a74d7bb8ba2bf6dc73d1a251cca6facd58fd3",
        );
        let fq_y = convert_hex_string_to_bn256_fq(
            "0x20bbc60135d38427f0f805fdf9ba60f0916fda28e470d4f93fe5ae9f005e222a",
        );
        let scalar = convert_hex_string_to_bn256_fr(
            "0x2272d81bf65542f08b87009f78424a384d057929d5b305b1509f7595b9a499db",
        );
        let point = G1Affine::from_xy(fq_x, fq_y).unwrap().to_curve();
        println!("Point: {:?}", point);
        println!("Scalar: {:?}", scalar);
        let expected_result = point.mul(scalar);

        println!("Expected result: {:?}", expected_result);

        let test_fr = bn256::Fr::from(5);

        let expected_mul = point.mul(test_fr);
        println!("Expected mul: {:?}", expected_mul);

        let small_mul = small_multiexp(&[test_fr], &[point.to_affine()]);
        println!("Small multiexp: {:?}", small_mul);
    }

    pub fn small_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
        let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
        let mut acc = C::Curve::identity();

        // for byte idx
        for byte_idx in (0..32).rev() {
            // for bit idx
            for bit_idx in (0..8).rev() {
                acc = acc.double();
                // for each coeff
                for coeff_idx in 0..coeffs.len() {
                    let byte = coeffs[coeff_idx].as_ref()[byte_idx];
                    if ((byte >> bit_idx) & 1) != 0 {
                        acc += bases[coeff_idx];
                    }
                }
            }
        }

        acc
    }
}
