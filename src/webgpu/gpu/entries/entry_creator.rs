use wgpu::util::DeviceExt;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    Device, Queue, ShaderModule,
};

use crate::webgpu::gpu::utils::{chunk_gpu_inputs, GpuU32Inputs};
// use futures::executor::block_on;

pub async fn get_device() -> Result<(Device, Queue), Box<dyn std::error::Error>> {
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

// Function to create the result buffer
fn create_result_buffer(device: &Device, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("Result Buffer"),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// Function to create an input buffer from u32 data
fn create_u32_array_input_buffer(device: &Device, uint32s: &[u32]) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(uint32s),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

// Function to setup bind group layout and bind group
fn setup_bind_group(
    device: &Device,
    gpu_buffer_inputs: &[Buffer],
    result_buffer: &Buffer,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    // Create a vector to hold all bind group layout entries
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = gpu_buffer_inputs
        .iter()
        .enumerate()
        .map(|(i, _)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    // Add the result buffer layout entry
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: gpu_buffer_inputs.len() as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &entries,
        label: Some("Bind Group Layout"),
    });

    // Correctly wrap BufferBinding in BindingResource
    let bind_group_entries = gpu_buffer_inputs
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()), // Correctly use as_entire_buffer_binding()
        })
        .chain(std::iter::once(wgpu::BindGroupEntry {
            binding: gpu_buffer_inputs.len() as u32,
            resource: wgpu::BindingResource::Buffer(result_buffer.as_entire_buffer_binding()),
        }))
        .collect::<Vec<_>>();

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &bind_group_entries,
        label: Some("Bind Group"),
    });

    (bind_group_layout, bind_group)
}

// Function to setup the compute pipeline
fn setup_compute_pipeline(
    device: &Device,
    shader_module: &ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: shader_module,
        entry_point: "main",
    })
}

// Function to submit compute pass
fn submit_compute_pass(
    device: &Device,
    queue: &Queue,
    compute_pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    num_inputs: usize,
    result_buffer: &Buffer,
    result_buffer_size: u64,
) -> Buffer {
    let gpu_read_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("GPU Read Buffer"),
        size: result_buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        let workgroup_count = (num_inputs + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    }

    // Copy from result_buffer to gpu_read_buffer
    command_encoder.copy_buffer_to_buffer(
        result_buffer,
        0,
        &gpu_read_buffer,
        0,
        result_buffer_size,
    );

    queue.submit(Some(command_encoder.finish()));

    gpu_read_buffer
}

// Function to read data from a buffer
pub async fn read_buffer(
    device: &Device,
    gpu_read_buffer: &Buffer,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let buffer_slice = gpu_read_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    receiver.recv_async().await.unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    gpu_read_buffer.unmap(); // Clean up
    Ok(result)
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
    let num_inputs = input_data[0].u32_inputs.len() / input_data[0].individual_input_size;
    let result_buffer_size =
        (num_inputs * u32_size_per_output) as u64 * std::mem::size_of::<u32>() as u64;

    let result_buffer = create_result_buffer(&device, result_buffer_size);
    let gpu_buffer_inputs = input_data
        .iter()
        .map(|data| create_u32_array_input_buffer(&device, &data.u32_inputs))
        .collect::<Vec<_>>();

    let (bind_group_layout, bind_group) =
        setup_bind_group(&device, &gpu_buffer_inputs, &result_buffer);
    let compute_pipeline = setup_compute_pipeline(&device, &shader_module, &bind_group_layout);

    let gpu_read_buffer = submit_compute_pass(
        &device,
        &queue,
        &compute_pipeline,
        &bind_group,
        num_inputs,
        &result_buffer,
        result_buffer_size,
    );

    // Map the result buffer to read its contents
    let output_data = read_buffer(&device, &gpu_read_buffer).await?;

    Ok(output_data)
}
