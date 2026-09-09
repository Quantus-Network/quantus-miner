//! Compare full-width GPU arithmetic against Rust u128, including lazy residues.
use rand::{RngCore, SeedableRng};
use wgpu::util::DeviceExt;

fn main() {
    tokio::runtime::Runtime::new().unwrap().block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SHADER_INT64,
            ..Default::default()
        })
        .await
        .unwrap();
    let p = 0xffff_ffff_0000_0001u64;
    let edges = [0, 1, 0xffff_ffff, 0x1_0000_0000, p - 1, p, p + 1, u64::MAX];
    let mut inputs = Vec::new();
    for a in edges {
        for b in edges {
            inputs.extend([a, b]);
        }
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x20260909);
    for _ in 0..4096 {
        inputs.extend([rng.next_u64(), rng.next_u64()]);
    }
    let count = inputs.len() / 2;
    let source = format!(
        r#"{}
@group(0) @binding(5) var<storage, read> pairs: array<u64>;
@group(0) @binding(6) var<storage, read_write> answer: array<u64>;
@compute @workgroup_size(64)
fn arithmetic_edges(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= {}u) {{ return; }}
    let a = pairs[id.x * 2u];
    let b = pairs[id.x * 2u + 1u];
    let v = mul_wide(a, b);
    answer[id.x * 5u] = v.lo;
    answer[id.x * 5u + 1u] = v.hi;
    answer[id.x * 5u + 2u] = gf64_canon(gf64_mul(a, b));
    answer[id.x * 5u + 3u] = gf64_canon(gf64_sqr(a));
    answer[id.x * 5u + 4u] = gf64_canon(gf64_reduce(U128(a,b)));
}}
"#,
        engine_gpu::Kernel::Apple.source(),
        count
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("arithmetic_edges"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let size = (count * 5 * 8) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 5,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    let answers: &[u64] = bytemuck::cast_slice(&mapped);
    for i in 0..count {
        let a = u128::from(inputs[i * 2]);
        let b = u128::from(inputs[i * 2 + 1]);
        let product = a * b;
        assert_eq!(
            &answers[i * 5..i * 5 + 5],
            &[
                product as u64,
                (product >> 64) as u64,
                (product % u128::from(p)) as u64,
                ((a * a) % u128::from(p)) as u64,
                (((b << 64) | a) % u128::from(p)) as u64
            ],
            "pair {i}"
        );
    }
    drop(mapped);
    staging.unmap();
    println!("ARITHMETIC OK: {count} operand pairs, wide product / modular multiply / square / arbitrary u128 reduction");
}
