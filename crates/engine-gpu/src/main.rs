use bytemuck;
use futures::executor::block_on;
use wgpu::{self, util::DeviceExt};

fn main() {
    block_on(run()).unwrap();
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Instance and adapter
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL, // Force Metal on Apple
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    // Create simplified WGSL mining shader (basic hash function for testing)
    let shader_source = r#"
        // Storage buffers
        @group(0) @binding(0) var<storage, read_write> results: array<u32>;
        @group(0) @binding(1) var<storage, read> header: array<u32, 8>;     // 32 bytes
        @group(0) @binding(2) var<storage, read> start_nonce: array<u32, 16>; // 64 bytes
        @group(0) @binding(3) var<storage, read> difficulty_target: array<u32, 16>;    // 64 bytes (U512 target)

        // Simple hash function (placeholder for now)
        fn simple_hash(input: array<u32, 24>) -> array<u32, 16> {
            var result: array<u32, 16>;

            // Very simple hash: XOR all input values and spread to output
            var hash_val = 0u;
            for (var i = 0u; i < 24u; i++) {
                hash_val = hash_val ^ input[i];
                hash_val = hash_val * 1103515245u + 12345u; // Simple LCG
            }

            // Fill output array with variations of hash
            for (var i = 0u; i < 16u; i++) {
                hash_val = hash_val * 1103515245u + 12345u;
                result[i] = hash_val;
            }

            return result;
        }

        // Double hash
        fn double_hash(input: array<u32, 24>) -> array<u32, 16> {
            let first_hash = simple_hash(input);
            // Convert back to input format for second hash
            var second_input: array<u32, 24>;
            for (var i = 0u; i < 16u; i++) {
                second_input[i] = first_hash[i];
            }
            // Pad remaining with zeros
            for (var i = 16u; i < 24u; i++) {
                second_input[i] = 0u;
            }
            return simple_hash(second_input);
        }

        // Check if hash < target (U512 comparison)
        fn is_below_target(hash: array<u32, 16>, difficulty_tgt: array<u32, 16>) -> bool {
            // Compare from most significant to least significant
            for (var i = 0u; i < 16u; i++) {
                if (hash[15u - i] < difficulty_tgt[15u - i]) {
                    return true;
                } else if (hash[15u - i] > difficulty_tgt[15u - i]) {
                    return false;
                }
            }
            return false; // Equal, not below
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let thread_id = id.x;

            // Prepare mining input: header (32 bytes) + nonce (64 bytes)
            var mining_input: array<u32, 24>; // 96 bytes total

            // Copy header (32 bytes = 8 u32s)
            for (var i = 0u; i < 8u; i++) {
                mining_input[i] = header[i];
            }

            // Calculate nonce for this thread
            var nonce: array<u32, 16> = start_nonce;
            // Add thread_id to the nonce (simplified increment)
            var carry = thread_id;
            for (var i = 0u; i < 16u && carry > 0u; i++) {
                let old_val = nonce[i];
                nonce[i] = old_val + carry;
                // Check for overflow
                if (nonce[i] < old_val) {
                    carry = 1u;
                } else {
                    carry = 0u;
                }
            }

            // Copy nonce (64 bytes = 16 u32s)
            for (var i = 0u; i < 16u; i++) {
                mining_input[8u + i] = nonce[i];
            }

            // Compute double Poseidon2 hash
            let hash_result = double_hash(mining_input);

            // Check if hash meets difficulty target
            if (is_below_target(hash_result, difficulty_target)) {
                // Found a valid nonce! Store it in results
                results[0] = 1u; // Success flag
                for (var i = 0u; i < 16u; i++) {
                    results[i + 1u] = nonce[i]; // Store winning nonce
                }
                for (var i = 0u; i < 16u; i++) {
                    results[i + 17u] = hash_result[i]; // Store winning hash
                }
            }
        }
    "#;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
    });

    // Buffer setup for mining
    // Results buffer: [success_flag, nonce[16], hash[16]] = 33 u32s
    let results_data = vec![0u32; 33];
    let results_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Results Buffer"),
        contents: bytemuck::cast_slice(&results_data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // Header buffer (32 bytes = 8 u32s)
    let header_data = [1u32, 2, 3, 4, 5, 6, 7, 8]; // Example header
    let header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Header Buffer"),
        contents: bytemuck::cast_slice(&header_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Start nonce buffer (64 bytes = 16 u32s)
    let start_nonce_data = [0u32; 16]; // Start from nonce 0
    let start_nonce_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Start Nonce Buffer"),
        contents: bytemuck::cast_slice(&start_nonce_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Target buffer (64 bytes = 16 u32s) - very easy target for testing
    let target_data = [0xFFFFFFFFu32; 16]; // Maximum target (very easy)
    let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Target Buffer"),
        contents: bytemuck::cast_slice(&target_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Pipeline with multiple buffers
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mining Bind Group Layout"),
        entries: &[
            // Results buffer (read-write)
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
            // Header buffer (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Start nonce buffer (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Target buffer (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Bind group with all buffers
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
        ],
    });

    // Dispatch mining workgroups
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mining Command Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Mining Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // Launch 1024 threads to test different nonces
        pass.dispatch_workgroups(16, 1, 1); // 16 * 64 = 1024 threads
    }
    queue.submit(Some(encoder.finish()));

    // Read back results
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: results_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Command Encoder"),
    });
    encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging, 0, results_buffer.size());
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let mapped = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();

    // Check mining results
    if result[0] == 1 {
        println!("🎉 Found valid nonce!");
        println!("Nonce: {:?}", &result[1..17]);
        println!("Hash: {:?}", &result[17..33]);
    } else {
        println!(
            "No valid nonce found in this batch. Result[0] = {}",
            result[0]
        );
        println!("First few results: {:?}", &result[0..10]);
    }

    Ok(())
}
