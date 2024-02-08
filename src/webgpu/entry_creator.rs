use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, Device, Queue, ShaderModule};
// use futures::executor::block_on;

async fn get_device() -> (Device, Queue) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap()
}

fn create_shader_module(device: &Device, shader_code: &str) -> ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
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

async fn run_shader(
    device: &Device,
    queue: &Queue,
    shader_code: &str,
    inputs: Vec<Vec<u32>>,
    output_size: usize,
) -> Vec<u32> {
    let shader_module = create_shader_module(device, shader_code);

    // Assuming inputs is non-empty and all inputs have the same length
    let num_inputs = inputs[0].len();
    let input_buffers: Vec<Buffer> = inputs
        .into_iter()
        .map(|input| create_buffer(device, &input, wgpu::BufferUsages::STORAGE))
        .collect();

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (output_size * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = create_bind_group_layout(device, input_buffers.len());
    let bind_group = create_bind_group(device, &bind_group_layout, &input_buffers, &output_buffer);

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: &shader_module,
        entry_point: "main",
    });

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((num_inputs as u32 + 63) / 64, 1, 1);
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (output_size * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    command_encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (output_size * std::mem::size_of::<u32>()) as u64,
    );
    queue.submit(Some(command_encoder.finish()));

    get_data(staging_buffer, device).await
}

// Note: In a real application, consider using `futures` and async/await to prevent blocking
async fn get_data(staging_buffer: Buffer, device: &Device) -> Vec<u32> {
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    receiver.recv_async().await.unwrap().unwrap();
    let mut output = vec![];
    output.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()[..]));
    staging_buffer.unmap();
    output
}

#[cfg_attr(test, allow(dead_code))]
async fn run() {
    let (device, queue) = get_device().await;
    // Example usage
    let shader_code = "..."; // Your WGSL shader code here
    let inputs: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4]]; // Example input
    let output_size = 4; // Number of u32s expected in the output

    let result = run_shader(&device, &queue, shader_code, inputs, output_size).await;
    println!("{:?}", result);
}

pub fn main() {
    pollster::block_on(run());
}
