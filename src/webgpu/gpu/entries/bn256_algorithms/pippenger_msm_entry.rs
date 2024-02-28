use group::Group;

use crate::{
    bn256::{self, Fq, Fr, G1},
    webgpu::gpu::{
        curve_specific::{
            get_ugly_curve_base_functions_wgsl, get_ugly_curve_params_wgsl, CurveType,
        },
        entries::entry_creator::{entry, GpuContext},
        prune::prune,
        u32_sizes::{EXT_POINT_SIZE, FIELD_SIZE},
        utils::{convert_bn256_fq_to_u32_array, convert_u32_array_to_bn256_fq_vec, GpuU32Inputs},
        wgsl::{UGLY_CURVE_WGSL, UGLY_FIELD_MODULUS_WGSL, UGLY_U256_WGSL},
    },
};
use std::{
    collections::HashMap,
    ops::{Add, Mul},
};

/// Breaks up a vector into separate vectors of size chunk_size
fn chunk_array<T: Clone>(input_array: Vec<T>, chunk_size: usize) -> Vec<Vec<T>> {
    input_array
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

// Function to convert Vec<u16> to Vec<u32>
fn convert_u16_vec_to_u32_vec(input: Vec<u16>) -> Vec<u32> {
    input.into_iter().map(|x| x as u32).collect()
}

// Convert Vec<Fq> to Vec<u32>
fn points_to_u32_array(points: Vec<Fq>) -> Vec<u32> {
    points
        .iter()
        .flat_map(|point| convert_bn256_fq_to_u32_array(point))
        .collect()
}

pub async fn pippenger_msm(
    points: Vec<G1>, // Assuming ExtPointType or ProjPointType or (BigInt, BigInt, BigInt) for x, y, z
    scalars: Vec<u16>,
    // field_math: Option<BLS12_377FieldMath>,
) -> Result<G1, String> {
    const SCALAR_CHUNK_WIDTH: usize = 16;

    // Dictionary setup
    let num_msms = 256 / SCALAR_CHUNK_WIDTH;
    let mut msms: Vec<HashMap<u16, G1>> = vec![HashMap::new(); num_msms];

    let start = std::time::Instant::now();
    // Bucket method
    let mut scalar_index = 0;
    let mut points_index = 0;
    while points_index < points.len() {
        let scalar = scalars[scalar_index];
        let point_to_add = &points[points_index];

        let msm_index = scalar_index % msms.len();

        let current_point = msms[msm_index].get(&scalar);
        if current_point.is_none() {
            msms[msm_index].insert(scalar, point_to_add.clone());
        } else {
            let current_point = msms[msm_index].get_mut(&scalar).unwrap();
            *current_point = current_point.add(point_to_add);
        }

        scalar_index += 1;
        if scalar_index % msms.len() == 0 {
            points_index += 1;
        }
    }
    println!("Time taken for bucket method: {:?}", start.elapsed());

    // GPU input setup & computation
    let mut points_concatenated = Vec::new();
    let mut scalars_concatenated = Vec::new();
    for msm in msms.iter() {
        for (scalar, point) in msm {
            let expanded_point = vec![point.x, point.y, point.x.mul(&point.y), point.z];
            points_concatenated.extend(expanded_point);
            scalars_concatenated.push(*scalar);
        }
    }

    // Handling GPU buffer and memory limits, chunking inputs
    let chunked_points = chunk_array(points_concatenated, 44_000);
    let chunked_scalars = chunk_array(scalars_concatenated, 11_000);

    let start = std::time::Instant::now();
    let mut gpu_results_as_fq_vec = Vec::new();
    let mut gpu_context = GpuContext::new().await.unwrap();
    for (chunked_point, chunked_scalar) in chunked_points.into_iter().zip(chunked_scalars.iter()) {
        let buffer_result = point_mul(
            &mut gpu_context,
            CurveType::BN254,
            GpuU32Inputs::new(points_to_u32_array(chunked_point), EXT_POINT_SIZE as usize),
            GpuU32Inputs::new(
                convert_u16_vec_to_u32_vec(chunked_scalar.clone()),
                FIELD_SIZE as usize,
            ),
        )
        .await
        .unwrap();
        gpu_results_as_fq_vec.extend(convert_u32_array_to_bn256_fq_vec(&buffer_result));
    }
    println!("Time taken for GPU computation: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    // Convert GPU results back to extended points
    let mut gpu_results_as_extended_points = Vec::new();
    for chunk in gpu_results_as_fq_vec.chunks(4) {
        let (projective_x, projective_y, _projective_t, projective_z) =
            (chunk[0], chunk[1], chunk[2], chunk[3]);
        let extended_point = bn256::G1::from_xyz(projective_x, projective_y, projective_z).unwrap();
        gpu_results_as_extended_points.push(extended_point);
    }
    println!(
        "Time taken for converting gpu results back: {:?}",
        start.elapsed()
    );

    let start = std::time::Instant::now();
    // Summation of scalar multiplications for each MSM
    let mut msm_results = Vec::new();
    let bucketing = msms.iter().map(|msm| msm.len());
    let mut prev_bucket_sum = 0;

    for bucket in bucketing {
        let mut current_sum = bn256::G1::identity();
        for i in 0..bucket {
            current_sum = current_sum.add(&gpu_results_as_extended_points[i + prev_bucket_sum]);
        }
        msm_results.push(current_sum);
        prev_bucket_sum += bucket;
    }
    println!("Time taken for summation: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    // Solve for original MSM
    let mut original_msm_result = msm_results[0].clone();
    let exponent_mul_term = Fr::from(2u64.pow(SCALAR_CHUNK_WIDTH as u32));
    for msm_result in msm_results.iter().skip(1) {
        original_msm_result = original_msm_result.mul(exponent_mul_term); // square
        original_msm_result = original_msm_result.add(msm_result);
    }
    println!("Time taken for solving original MSM: {:?}", start.elapsed());

    Ok(original_msm_result)
}

async fn point_mul(
    gpu_context: &mut GpuContext,
    curve: CurveType,
    input1: GpuU32Inputs,
    input2: GpuU32Inputs,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let shader_entry = format!(
        r#"
        const ZERO_POINT = UglyPoint (U256_ZERO, U256_ONE, U256_ZERO, U256_ZERO);
        const ZERO_AFFINE = UglyAffinePoint (U256_ZERO, U256_ONE);

        @group(0) @binding(0) var<storage, read> input1: array<UglyPoint>;
        @group(0) @binding(1) var<storage, read> input2: array<u32>;
        @group(0) @binding(2) var<storage, read_write> output: array<UglyPoint>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{
            var extended_point = input1[global_id.x];
            var scalar = input2[global_id.x];

            var result = mul_point_32_bit_scalar(extended_point, scalar);

            output[global_id.x] = result;
        }}
        "#
    );

    let shader_code = prune(
        &vec![
            UGLY_U256_WGSL,
            &get_ugly_curve_params_wgsl(curve),
            UGLY_FIELD_MODULUS_WGSL,
            UGLY_CURVE_WGSL,
            &get_ugly_curve_base_functions_wgsl(curve),
        ]
        .join("\n"),
        &vec!["mul_point_32_bit_scalar"],
    ) + &shader_entry;

    entry(
        gpu_context,
        vec![input1, input2],
        &shader_code,
        EXT_POINT_SIZE as usize,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bn256::G1Affine,
        msm::{best_multiexp, small_multiexp},
        serde::SerdeObject,
        webgpu::{
            gpu::utils::{
                convert_bn_256_scalars_to_u16_array, convert_hex_string_to_bn256_fq,
                convert_hex_string_to_bn256_fr,
            },
            utils::input_generator::point_scalar_generator,
        },
    };
    use ff::{Field, PrimeField};
    use group::{prime::PrimeCurveAffine, Curve, GroupEncoding};
    use pasta_curves::arithmetic::{CurveAffine, CurveExt};
    use tokio::test as async_test; // Adjust according to your async runtime if not using Tokio.

    #[async_test]
    async fn test_pippenger_msm_single_input() {
        // let point = generate_random_affine_point::<G1Affine>().to_curve();
        // let scalar = generate_random_scalar_point::<G1Affine>();
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
        let expected_result = point.mul(scalar);

        let test_cases = vec![(vec![point], vec![scalar], expected_result)];

        for (points, scalars, expected_result) in test_cases {
            let actual_result =
                pippenger_msm(points, convert_bn_256_scalars_to_u16_array(&scalars))
                    .await
                    .expect("MSM computation failed");
            println!("Actual result: {:?}", actual_result);

            assert_eq!(
                actual_result, expected_result,
                "MSM result did not match the expected value"
            );
        }
    }

    #[async_test]
    async fn test_pippenger_msm_multiple_inputs() {
        // time taken for input generation
        let start = std::time::Instant::now();
        let point_scalar_inputs = point_scalar_generator::<G1Affine>(100);
        println!("Time taken for input generation: {:?}", start.elapsed());

        // time taken for input preprocessing
        let start = std::time::Instant::now();
        let affine_points: Vec<G1Affine> = point_scalar_inputs.iter().map(|x| x.point).collect();
        let points: Vec<G1> = point_scalar_inputs
            .iter()
            .map(|x| x.point.to_curve())
            .collect();
        let scalars: Vec<Fr> = point_scalar_inputs.iter().map(|x| x.scalar).collect();
        println!("Time taken for input preprocessing: {:?}", start.elapsed());

        // Measure time taken for small CPU multiexp
        let start = std::time::Instant::now();
        let expected_result =
            small_multiexp::<G1Affine>(scalars.as_slice(), affine_points.as_slice());
        println!("Time taken for small CPU multiexp: {:?}", start.elapsed());

        // Measure time taken for best CPU multiexp
        let start = std::time::Instant::now();
        let best_cpu_result =
            best_multiexp::<G1Affine>(scalars.as_slice(), affine_points.as_slice());
        println!("Time taken for best CPU multiexp: {:?}", start.elapsed());

        assert_eq!(
            expected_result, best_cpu_result,
            "Small multiexp result did not match the best multiexp result"
        );

        // Perform pippenger_msm
        // measure time taken
        let start = std::time::Instant::now();
        let actual_result = pippenger_msm(points, convert_bn_256_scalars_to_u16_array(&scalars))
            .await
            .expect("MSM computation failed");
        println!("Time taken for pippenger_msm: {:?}", start.elapsed());

        println!("Expected result: {:?}", expected_result.to_affine());
        println!("Actual result: {:?}", actual_result.to_affine());
        assert_eq!(
            actual_result, expected_result,
            "MSM result did not match the expected value"
        );
    }

    #[async_test]
    async fn test_pippenger_only() {
        // time taken for input generation
        let start = std::time::Instant::now();
        let point_scalar_inputs = point_scalar_generator::<G1Affine>(100);
        println!("Time taken for input generation: {:?}", start.elapsed());

        // time taken for input preprocessing
        let points: Vec<G1> = point_scalar_inputs
            .iter()
            .map(|x| x.point.to_curve())
            .collect();
        let scalars: Vec<Fr> = point_scalar_inputs.iter().map(|x| x.scalar).collect();

        // print out the points and scalars
        println!("Points: {:?}", points);
        println!("Scalars: {:?}", scalars);

        // Perform pippenger_msm
        // measure time taken
        let start = std::time::Instant::now();
        let actual_result = pippenger_msm(points, convert_bn_256_scalars_to_u16_array(&scalars))
            .await
            .expect("MSM computation failed");
        println!("Time taken for pippenger_msm: {:?}", start.elapsed());

        println!("Actual result: {:?}", actual_result.to_affine());
    }

    #[test]
    fn test_point_serde() {
        let rng = &mut rand::thread_rng();
        let random_point = G1::random(rng);
        let rng = &mut rand::thread_rng();
        let random_scalar = Fr::random(rng);
        let mul_point = random_point.mul(random_scalar);
        println!("Mul point: {:?}", mul_point);

        // to bytes
        println!("Mul point to raw bytes: {:?}", mul_point.to_raw_bytes());

        // print to_repr
        println!("Mul point x to repr: {:?}", mul_point.x.to_repr());
        println!("Mul point y to repr: {:?}", mul_point.y.to_bytes());
        println!("Mul point z to repr: {:?}", mul_point.z.to_bytes());

        // hex encoded
        println!(
            "Mul point x to hex: {:?}",
            hex::encode(mul_point.x.to_bytes())
        );
        println!(
            "Mul point y to hex: {:?}",
            hex::encode(mul_point.y.to_bytes())
        );
        println!(
            "Mul point z to hex: {:?}",
            hex::encode(mul_point.z.to_bytes())
        );

        let point_from_coordinates = G1::from_xyz(mul_point.x, mul_point.y, mul_point.z).unwrap();

        println!("Point from coordinates: {:?}", point_from_coordinates);
    }
}
