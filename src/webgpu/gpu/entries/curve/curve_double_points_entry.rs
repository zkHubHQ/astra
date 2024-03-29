use crate::webgpu::gpu::{
    curve_specific::{get_curve_base_functions_wgsl, get_curve_params_wgsl, CurveType},
    entries::entry_creator::batched_entry,
    prune::prune,
    u32_sizes::FIELD_SIZE,
    utils::GpuU32Inputs,
    wgsl::{CURVE_WGSL, FIELD_MODULUS_WGSL, U256_WGSL},
};

async fn point_double(
    curve: CurveType,
    points: GpuU32Inputs,
    batch_size: Option<usize>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let shader_entry = format!(
        r#"@group(0) @binding(0)
        var<storage, read> input1: array<AffinePoint>;
        @group(0) @binding(1)
        var<storage, read_write> output: array<Field>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{
            var p1 = input1[global_id.x];
            var p1_t = field_multiply(p1.x, p1.y);
            var z = U256_ONE;
            var ext_p1 = Point(p1.x, p1.y, p1_t, z);

            var doubled = double_point(ext_p1);
            var x_normalized = normalize_x(doubled.x, doubled.z);

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
        &["field_multiply", "double_point", "normalize_x"],
    ) + &shader_entry;

    batched_entry(
        vec![points],
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
    use crate::webgpu::gpu::utils::big_int_to_u32_array;
    use num_bigint::BigInt;
    use tokio::test as async_test;

    #[async_test]
    #[ignore]
    async fn test_point_double() {
        let points = GpuU32Inputs {
            u32_inputs: big_int_to_u32_array(&BigInt::from(123)),
            individual_input_size: FIELD_SIZE as usize,
        };
        let curve = CurveType::BN254;

        let result = point_double(curve, points, Some(1)).await.unwrap();
    }
}
