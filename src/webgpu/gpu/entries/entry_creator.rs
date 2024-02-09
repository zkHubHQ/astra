use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, Device,
    PipelineLayoutDescriptor, Queue, ShaderModule, ShaderStages,
};

use crate::webgpu::gpu::utils::{chunk_gpu_inputs, GpuU32Inputs};
// use futures::executor::block_on;

async fn get_device() -> Result<(Device, Queue), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok_or("Failed to find an appropriate adapter")?;

    adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(Into::into)
}

fn create_shader_module(device: &Device, shader_code: &str) -> ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    })
}

fn create_buffer(device: &Device, data: &[u32], usage: wgpu::BufferUsages) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer"),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

fn create_bind_group_layout(device: &Device, num_inputs: usize) -> BindGroupLayout {
    let mut bind_group_layout_entries = (0..num_inputs)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect::<Vec<_>>();
    bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
        binding: num_inputs as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("BindGroupLayout"),
        entries: &bind_group_layout_entries,
    })
}

fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    buffers: &[Buffer],
    result_buffer: &Buffer,
) -> BindGroup {
    let mut entries = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();

    entries.push(wgpu::BindGroupEntry {
        binding: buffers.len() as u32,
        resource: result_buffer.as_entire_binding(),
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BindGroup"),
        layout,
        entries: &entries,
    })
}

pub async fn batched_entry(
    input_data: Vec<GpuU32Inputs>,
    shader_code: &str,
    u32_size_per_output: usize,
    batch_size: Option<usize>,
    inputs_to_batch: Option<Vec<usize>>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let u32_size_per_first_input = input_data[0].individual_input_size;
    let total_inputs = input_data[0].u32_inputs.len() / u32_size_per_first_input;
    let total_expected_outputs = total_inputs;
    let batch_size = batch_size.unwrap_or(total_inputs);
    let inputs_to_batch = inputs_to_batch.unwrap_or_else(Vec::new); // default to batching all inputs

    let chunked_inputs = if batch_size < total_inputs {
        chunk_gpu_inputs(&input_data, batch_size, &inputs_to_batch)
    } else {
        vec![input_data]
    };

    let mut output_result = Vec::with_capacity(total_expected_outputs * u32_size_per_output);

    for chunk in chunked_inputs {
        let batch_result = entry(chunk, shader_code, u32_size_per_output).await?;
        output_result.extend(batch_result);
    }

    Ok(output_result)
}

pub async fn entry(
    input_data: Vec<GpuU32Inputs>,
    shader_code: &str,
    u32_size_per_output: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let (device, queue) = get_device().await?;
    let shader_module = create_shader_module(&device, shader_code);

    let num_elements = input_data
        .iter()
        .map(|input| input.u32_inputs.len())
        .sum::<usize>();
    let result_size = num_elements * u32_size_per_output;

    let input_data_flat = input_data
        .iter()
        .flat_map(|input| &input.u32_inputs)
        .cloned()
        .collect::<Vec<u32>>();

    let input_buffer = create_buffer(&device, &input_data_flat, BufferUsages::STORAGE);
    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Output Buffer"),
        size: (result_size * std::mem::size_of::<u32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
    });

    let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((num_elements as u32 + 63) / 64, 1, 1);
        // Adjust based on your workgroup size and needs
    }

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Read Buffer"),
        size: output_buffer.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    command_encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        output_buffer.size(),
    );
    queue.submit(Some(command_encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    receiver.recv_async().await.unwrap().unwrap();
    let mut output = vec![];
    output.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()[..]));
    staging_buffer.unmap(); // Clean up

    Ok(output)
}

#[cfg_attr(test, allow(dead_code))]
fn main() {
    // Example shader code. Replace with your actual WGSL shader code.
    let shader_code = r#"
        @group(0) @binding(0) var<storage, read> input_data: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx < arrayLength(&input_data)) {
                output_data[idx] = input_data[idx] * 2; // Example shader operation
            }
        }
    "#;

    // Example input data. Replace with your actual data.
    let input_data = vec![
        GpuU32Inputs {
            u32_inputs: vec![1, 2, 3, 4], // Example data
            individual_input_size: 1,
        },
        // Add more GpuU32Inputs structs as needed.
    ];

    // Parameters for batched_entry call.
    let u32_size_per_output = 1; // Adjust as needed.
    let batch_size = Some(2); // Optional: adjust your batch size as needed.
    let inputs_to_batch = Some(vec![]); // Optional: specify indices of inputs to batch.

    // Running the batched_entry function synchronously using pollster::block_on.
    let result = pollster::block_on(batched_entry(
        input_data,
        shader_code,
        u32_size_per_output,
        batch_size,
        inputs_to_batch,
    ));

    // Handling the result of the batched_entry call.
    match result {
        Ok(output) => {
            println!("Output: {:?}", output);
            // Process output as needed.
        }
        Err(e) => eprintln!("Error executing batched_entry: {}", e),
    }
}
