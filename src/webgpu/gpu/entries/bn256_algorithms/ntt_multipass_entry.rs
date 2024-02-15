use crate::{
    bn256::Fr,
    webgpu::gpu::{
        entries::entry_creator::{get_device, read_buffer},
        u32_sizes::FIELD_SIZE,
        utils::{convert_bn256_scalar_to_u32_array, convert_hex_string_to_u32_array, GpuU32Inputs},
    },
};
use ff::PrimeField;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct ShaderCode {
    code: String,
    entry_point: String,
}

#[derive(Debug)]
pub struct EntryInfo {
    num_inputs: u64,
    output_size: u64,
    num_inputs_for_workgroup: Option<u64>,
}

fn join_numbers(numbers: &Vec<u32>, separator: &str) -> String {
    numbers
        .iter()
        .map(|num| num.to_string())
        .collect::<Vec<String>>()
        .join(separator)
}

pub fn ntt_multipass_info(
    num_inputs: usize,
    roots: &HashMap<usize, Fr>,
    wn_modules: &str,
    butterfly_modules: &str,
) -> (Vec<ShaderCode>, EntryInfo) {
    let log_num_inputs = (num_inputs as f64).log2() as usize;
    let mut w_current = roots[&log_num_inputs];
    let mut wn_precomputed: Vec<Fr> = vec![w_current];
    const workgroup_size: u32 = 64;
    for _ in 0..log_num_inputs {
        w_current = w_current.square();
        wn_precomputed.push(w_current);
    }

    let mut shaders: Vec<ShaderCode> = Vec::new();
    let input_output_buffer_size =
        (num_inputs * (FIELD_SIZE as usize)) as u64 * std::mem::size_of::<u32>() as u64;
    let field_modulus_u32 = convert_hex_string_to_u32_array(Fr::MODULUS);

    // Steps 1 to log(n) - 1: Parallelized Cooley-Tukey FFT algorithm
    for i in 0..log_num_inputs {
        let half_len = 2usize.pow(i as u32);
        let wn = wn_precomputed[log_num_inputs - i - 1];
        let wn_entry = format!(
            r#"
            @group(0) @binding(0)
            var<storage, read_write> wN: array<Field>;

            @compute @workgroup_size({workgroup_size})
            fn main(
                @builtin(global_invocation_id)
                global_id : vec3<u32>
            ) {{
                let field_modulus = Field(array<u32, 8>({field_modulus_u32}u));
                let half_len: u32 = {half_len}u;
                let j: u32 = global_id.x % half_len;
                let wn: Field = Field(array<u32, 8>({wn_u32}u));
                wN[j] = gen_field_pow(wn, Field(array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, j)), field_modulus);
            }}
            "#,
            workgroup_size = workgroup_size,
            field_modulus_u32 = join_numbers(&field_modulus_u32, "u, "),
            half_len = half_len,
            wn_u32 = join_numbers(&convert_bn256_scalar_to_u32_array(&wn).u32_array, "u, "),
        );

        let wn_shader = ShaderCode {
            code: format!("{}\n{}", wn_modules, wn_entry),
            entry_point: "main".to_string(),
        };
        shaders.push(wn_shader);

        let log_pass_entry = format!(
            r#"
            @group(0) @binding(0)
            var<storage, read_write> coeffs: array<Field>;

            @group(0) @binding(1)
            var<storage, read_write> wi_precomp: array<Field>;

            @compute @workgroup_size({workgroup_size})
            fn main(
                @builtin(global_invocation_id)
                global_id : vec3<u32>
            ) {{
                let len: u32 = {len}u;
                let half_len: u32 = {half_len}u;
                let group_id: u32 = global_id.x / half_len;
                let field_modulus = Field(array<u32, 8>({field_modulus_u32}u));

                let i: u32 = group_id * len;
                let j: u32 = global_id.x % half_len;
                let w_i = wi_precomp[j];
                let u: Field = coeffs[i + j];
                let v: Field = gen_field_multiply(w_i, coeffs[i + j + half_len], field_modulus);
                coeffs[i + j] = gen_field_add(u, v, field_modulus);
                coeffs[i + j + half_len] = gen_field_sub(u, v, field_modulus);
            }}
            "#,
            workgroup_size = workgroup_size,
            len = 2 * half_len,
            half_len = half_len,
            field_modulus_u32 = join_numbers(&field_modulus_u32, "u, "),
        );

        let log_pass_shader = ShaderCode {
            code: format!("{}\n{}", butterfly_modules, log_pass_entry),
            entry_point: "main".to_string(),
        };
        shaders.push(log_pass_shader);
    }

    let entry_info = EntryInfo {
        num_inputs: num_inputs as u64,
        output_size: input_output_buffer_size,
        num_inputs_for_workgroup: Some((num_inputs / 2) as u64),
    };

    (shaders, entry_info)
}

pub async fn ntt_multipass(
    polynomial_coefficients: GpuU32Inputs,
    roots: HashMap<usize, Fr>,
    wn_modules: &str,
    butterfly_modules: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let (device, queue) = get_device().await?;
    let num_inputs =
        polynomial_coefficients.u32_inputs.len() / polynomial_coefficients.individual_input_size;
    let input_output_buffer_size =
        (num_inputs * (FIELD_SIZE as usize)) as u64 * std::mem::size_of::<u32>() as u64;

    let w_n_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wN Buffer"),
        size: input_output_buffer_size / 2,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Polynomial Coefficients Buffer"),
        contents: bytemuck::cast_slice(&polynomial_coefficients.u32_inputs),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let (shaders, entry_info) =
        ntt_multipass_info(num_inputs, &roots, wn_modules, butterfly_modules);

    for i in 0..shaders.len() / 2 {
        let wn_shader = &shaders[2 * i];
        let wn_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wn Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let wn_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &wn_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(w_n_buffer.as_entire_buffer_binding()),
            }],
            label: Some("wn Bind Group"),
        });
        let wn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wn Compute Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("wn Pipeline Layout"),
                    bind_group_layouts: &[&wn_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wn Shader Module"),
                source: wgpu::ShaderSource::Wgsl(wn_shader.code.clone().into()),
            }),
            entry_point: &wn_shader.entry_point,
        });
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("wn Compute Pass"),
                    timestamp_writes: None,
                });
            compute_pass.set_pipeline(&wn_pipeline);
            compute_pass.set_bind_group(0, &wn_bind_group, &[]);
            println!("Dispatching workgroups: {}", (2usize.pow(i as u32) + 63 / 64 as usize) as u32);
            compute_pass.dispatch_workgroups(
                (2usize.pow(i as u32) + 63 / 64 as usize) as u32, // 64 is the workgroup size
                1,
                1,
            );
        }
        queue.submit(Some(command_encoder.finish()));

        let shader = &shaders[2 * i + 1];
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(w_n_buffer.as_entire_buffer_binding()),
                },
            ],
            label: Some("Bind Group"),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader Module"),
                source: wgpu::ShaderSource::Wgsl(shader.code.clone().into()),
            }),
            entry_point: &shader.entry_point,
        });
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                ((entry_info
                    .num_inputs_for_workgroup
                    .unwrap_or(entry_info.num_inputs)
                    + 63 / 64 as u64) as usize) as u32, // 64 is the workgroup size
                1,
                1,
            );
        }
        queue.submit(Some(command_encoder.finish()));
    }

    let gpu_read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU Read Buffer"),
        size: entry_info.output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });
    command_encoder.copy_buffer_to_buffer(&buffer, 0, &gpu_read_buffer, 0, entry_info.output_size);
    queue.submit(Some(command_encoder.finish()));

    // Map the result buffer to read its contents
    let output_data = read_buffer(&device, &gpu_read_buffer).await?;

    Ok(output_data)
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use super::*;
    use crate::fft::best_fft;
    use crate::webgpu::gpu::curve_specific::get_curve_params_wgsl;
    use crate::webgpu::gpu::utils::convert_bn_256_scalars_to_u32_array;
    use crate::webgpu::gpu::utils::convert_u32_array_to_bn256_fr_vec;
    use crate::webgpu::gpu::wgsl::FIELD_MODULUS_WGSL;
    use crate::webgpu::gpu::wgsl::U256_WGSL;
    use crate::webgpu::gpu::{prune::prune, utils::convert_hex_string_to_bn256_fr, CurveType};
    use ff::Field;
    use rand_core::OsRng;
    use tokio::test as async_test;

    #[async_test]
    async fn test_ntt_multipass_basic() {
        let mut polynomial_coeff_fr = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(0),
            Fr::from(0),
        ];
        let polynomial_length = polynomial_coeff_fr.len();
        let polynomial_coeff_vec =
            convert_bn_256_scalars_to_u32_array(&polynomial_coeff_fr.clone());
        println!("Polynomial coefficients: {:?}", polynomial_coeff_vec);
        let individual_input_size = 8;
        let omega = convert_hex_string_to_bn256_fr(
            "0x30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636",
            // "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
        );

        // Expected result using best_fft
        best_fft(
            &mut polynomial_coeff_fr[0..polynomial_length],
            omega,
            // log of the length of the polynomial
            (polynomial_length as f64).log2() as u32,
        );
        println!("Expected result: {:?}", polynomial_coeff_fr);

        let polynomial_coefficients = GpuU32Inputs {
            u32_inputs: polynomial_coeff_vec, // Example polynomial coefficients
            individual_input_size: individual_input_size,
        };
        let roots = HashMap::from([(2, omega)]); // Example roots of unity

        let base_modules = vec![
            U256_WGSL,
            &get_curve_params_wgsl(CurveType::BN254),
            FIELD_MODULUS_WGSL,
        ];
        let wn_modules = prune(&base_modules.join("\n"), &vec!["gen_field_pow"]);
        let butterfly_modules = prune(
            &base_modules.join("\n"),
            &vec!["gen_field_add", "gen_field_sub", "gen_field_multiply"],
        );

        let result = ntt_multipass(
            polynomial_coefficients,
            roots,
            wn_modules.as_str(),
            butterfly_modules.as_str(),
        )
        .await
        .unwrap();

        let result_fr: Vec<Fr> = convert_u32_array_to_bn256_fr_vec(&result);

        println!("Actual result: {:?}", result_fr);
        // Add your assertions here
    }

    fn generate_data(k: u32) -> Vec<Fr> {
        let n = 1 << k;
        let timer = SystemTime::now();
        println!("\n\nGenerating 2^{k} = {n} values..",);
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let end = timer.elapsed().unwrap();
        println!(
            "Generating 2^{k} = {n} values took: {} sec.\n\n",
            end.as_secs()
        );
        data
    }

    #[async_test]
    async fn test_ntt_multipass_medium() {
        let k = 4;
        let mut polynomial_coeff_fr = generate_data(k);

        let polynomial_length = polynomial_coeff_fr.len();
        let polynomial_coeff_vec =
            convert_bn_256_scalars_to_u32_array(&polynomial_coeff_fr.clone());
        println!("Polynomial coefficients: {:?}", polynomial_coeff_vec);
        let individual_input_size = 8;
        let omega: Fr = convert_hex_string_to_bn256_fr(
            "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b",
        );

        // Expected result using best_fft
        best_fft(&mut polynomial_coeff_fr[0..polynomial_length], omega, k);
        println!("Expected result: {:?}", polynomial_coeff_fr);

        let polynomial_coefficients = GpuU32Inputs {
            u32_inputs: polynomial_coeff_vec, // Example polynomial coefficients
            individual_input_size,
        };
        let roots = HashMap::from([(4, omega)]); // Build this omega locally

        let base_modules = vec![
            U256_WGSL,
            &get_curve_params_wgsl(CurveType::BN254),
            FIELD_MODULUS_WGSL,
        ];
        let wn_modules = prune(&base_modules.join("\n"), &vec!["gen_field_pow"]);
        let butterfly_modules = prune(
            &base_modules.join("\n"),
            &vec!["gen_field_add", "gen_field_sub", "gen_field_multiply"],
        );

        let result = ntt_multipass(
            polynomial_coefficients,
            roots,
            wn_modules.as_str(),
            butterfly_modules.as_str(),
        )
        .await
        .unwrap();

        let result_fr: Vec<Fr> = convert_u32_array_to_bn256_fr_vec(&result);

        println!("Actual result: {:?}", result_fr);
        // Add your assertions here
    }
}
