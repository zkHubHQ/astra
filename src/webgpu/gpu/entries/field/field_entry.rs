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
