use pow_core::{hash_from_nonce, JobContext};
use primitive_types::U512;
use rand::Rng;
use rand::SeedableRng;
use wgpu::util::DeviceExt;

pub async fn test_end_to_end_mining(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running End-to-End Mining Test...");

    // 1. Setup a job context
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);
    let mut header = [0u8; 32];
    rng.fill(&mut header);

    // Use difficulty 1 so target is MAX. Any hash should pass.
    let difficulty = U512::from(1u64);
    let ctx = JobContext::new(header, difficulty);

    // 2. Pick a nonce
    let nonce_val = U512::from(123456789u64);

    // 3. Compute expected hash using CPU (pow-core)
    let expected_hash = hash_from_nonce(&ctx, nonce_val);
    println!("CPU Expected Hash: {:x}", expected_hash);

    // 4. Run GPU Mining for this specific nonce

    // Header Buffer
    let mut header_u32s = [0u32; 8];
    for i in 0..8 {
        let chunk = &ctx.header[i * 4..(i + 1) * 4];
        header_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Header Buffer"),
        contents: bytemuck::cast_slice(&header_u32s),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Target Buffer
    let target_bytes = ctx.target.to_little_endian();
    let mut target_u32s = [0u32; 16];
    for i in 0..16 {
        let chunk = &target_bytes[i * 4..(i + 1) * 4];
        target_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Target Buffer"),
        contents: bytemuck::cast_slice(&target_u32s),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Start Nonce Buffer
    // We set start_nonce = nonce_val. Thread 0 will check start_nonce + 0 = nonce_val.
    let start_nonce_bytes = nonce_val.to_little_endian();
    let mut start_nonce_u32s = [0u32; 16];
    for i in 0..16 {
        let chunk = &start_nonce_bytes[i * 4..(i + 1) * 4];
        start_nonce_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let start_nonce_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Start Nonce Buffer"),
        contents: bytemuck::cast_slice(&start_nonce_u32s),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Results Buffer
    let results_size = (1 + 16 + 16) * 4;
    let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Results Buffer"),
        size: results_size as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Clear results buffer
    let zeros = vec![0u8; results_size];
    queue.write_buffer(&results_buffer, 0, &zeros);

    // Dispatch config buffer: [total_threads, nonces_per_thread, work_per_batch, threads_per_workgroup]
    let dispatch_config_data: [u32; 4] = [256, 1, 1, 256];
    let dispatch_config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dispatch Config Buffer"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &dispatch_config_buffer,
        0,
        bytemuck::cast_slice(&dispatch_config_data),
    );

    // Load Shader
    let shader_source = include_str!("mining.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mining Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mining Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("mining_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mining Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: results_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: header_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: start_nonce_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: target_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dispatch_config_buffer.as_entire_binding(),
            },
        ],
    });

    // Run Compute Pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mining Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Mining Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1); // 1 workgroup, 256 threads. Thread 0 will check nonce_val.
    }

    // Read Results
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: results_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, results_size as u64);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();

    if let Ok(Ok(())) = receiver.await {
        let data = buffer_slice.get_mapped_range();
        let result_u32s: &[u32] = bytemuck::cast_slice(&data);

        if result_u32s[0] != 0 {
            println!("GPU found solution!");

            // Parse nonce
            let mut nonce_bytes = [0u8; 64];
            for i in 0..16 {
                let bytes = result_u32s[1 + i].to_le_bytes();
                nonce_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }
            let found_nonce = U512::from_little_endian(&nonce_bytes);
            println!("GPU Nonce: {}", found_nonce);

            assert_eq!(found_nonce, nonce_val, "Nonce mismatch!");

            // Parse hash
            let mut hash_bytes = [0u8; 64];
            for i in 0..16 {
                let bytes = result_u32s[17 + i].to_le_bytes();
                hash_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }
            let found_hash = U512::from_little_endian(&hash_bytes);
            println!("GPU Hash: {:x}", found_hash);

            assert_eq!(found_hash, expected_hash, "Hash mismatch!");
            println!("✅ End-to-End Test Passed!");
        } else {
            println!("❌ GPU did not find solution (should have passed with MAX target)");
            return Err("GPU did not find solution".into());
        }
    } else {
        return Err("Failed to map buffer".into());
    }

    Ok(())
}
