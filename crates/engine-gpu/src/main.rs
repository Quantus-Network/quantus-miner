use bytemuck;
use futures::executor::block_on;

use wgpu::{self, util::DeviceExt};

mod tests;

// Extract Poseidon2 constants and generate WGSL code
fn generate_wgsl_constants() {
    println!("Extracting Poseidon2 constants for WGSL...");

    // Internal round constants (22 values)
    println!("// Internal round constants");
    println!("const INTERNAL_CONSTANTS: array<array<u32, 2>, 22> = array<array<u32, 2>, 22>(");
    for (i, &constant) in qp_poseidon_constants::POSEIDON2_INTERNAL_CONSTANTS_RAW
        .iter()
        .enumerate()
    {
        let low = (constant & 0xFFFFFFFF) as u32;
        let high = (constant >> 32) as u32;
        if i < 21 {
            println!("    array<u32, 2>({}, {}),", low, high);
        } else {
            println!("    array<u32, 2>({}, {})", low, high);
        }
    }
    println!(");");

    // Initial external round constants (4 rounds x 12 elements each)
    println!("\n// Initial external round constants");
    println!("const INITIAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(");
    for (round_idx, round) in qp_poseidon_constants::POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW
        .iter()
        .enumerate()
    {
        println!("    array<array<u32, 2>, 12>(");
        for (i, &constant) in round.iter().enumerate() {
            let low = (constant & 0xFFFFFFFF) as u32;
            let high = (constant >> 32) as u32;
            if i < 11 {
                println!("        array<u32, 2>({}, {}),", low, high);
            } else {
                println!("        array<u32, 2>({}, {})", low, high);
            }
        }
        if round_idx < 3 {
            println!("    ),");
        } else {
            println!("    )");
        }
    }
    println!(");");

    // Terminal external round constants (4 rounds x 12 elements each)
    println!("\n// Terminal external round constants");
    println!("const TERMINAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(");
    for (round_idx, round) in qp_poseidon_constants::POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW
        .iter()
        .enumerate()
    {
        println!("    array<array<u32, 2>, 12>(");
        for (i, &constant) in round.iter().enumerate() {
            let low = (constant & 0xFFFFFFFF) as u32;
            let high = (constant >> 32) as u32;
            if i < 11 {
                println!("        array<u32, 2>({}, {}),", low, high);
            } else {
                println!("        array<u32, 2>({}, {})", low, high);
            }
        }
        if round_idx < 3 {
            println!("    ),");
        } else {
            println!("    )");
        }
    }
    println!(");");
}

// Generate simple debug test vectors
fn generate_debug_test_vectors() -> Vec<([u8; 96], [u8; 64])> {
    println!("Generating simplified debug test vectors...");

    let mut test_vectors = Vec::new();

    // Test vector 1: All zeros
    let input1 = [0u8; 96];
    let expected1 = qp_poseidon_core::hash_squeeze_twice(&input1);
    test_vectors.push((input1, expected1));
    println!(
        "Test vector 1 (zeros): {:?} -> {:?}",
        &input1[..8],
        &expected1[..8]
    );

    // Test vector 2: Simple sequential pattern
    let mut input2 = [0u8; 96];
    for i in 0..16 {
        input2[i] = (i + 1) as u8; // [1,2,3...16,0,0,...]
    }
    let expected2 = qp_poseidon_core::hash_squeeze_twice(&input2);
    test_vectors.push((input2, expected2));
    println!(
        "Test vector 2 (sequential): {:?} -> {:?}",
        &input2[..8],
        &expected2[..8]
    );

    test_vectors
}

// Test vector generation using the CPU implementation
fn generate_test_vectors() -> Vec<([u8; 96], [u8; 64])> {
    println!("Generating test vectors using CPU Poseidon2...");

    let mut test_vectors = Vec::new();

    // Test vector 1: All zeros
    let input1 = [0u8; 96];
    println!("=== CPU DEBUG for Test Vector 1 (all zeros) ===");
    println!("Input bytes: {:?}", &input1[..24]);

    // Manual CPU implementation to see what it produces
    let expected1 = qp_poseidon_core::hash_squeeze_twice(&input1);

    // Also get debug version to see intermediate states
    let felts = qp_poseidon_core::serialization::injective_bytes_to_felts::<
        qp_poseidon_core::serialization::p2_backend::GF,
    >(&input1);
    println!("CPU converted to {} field elements:", felts.len());
    for (i, felt) in felts.iter().enumerate() {
        println!("  CPU felt[{}] = {} (0x{:016x})", i, felt.0, felt.0);
    }

    test_vectors.push((input1, expected1));
    println!("CPU Test vector 1 (zeros): expected={:?}", &expected1[..8]);

    // Test vector 2: Simple 4-byte test to debug conversion
    let input2 = [1u8, 2u8, 3u8, 4u8]; // Just 4 bytes
    let felts2 = qp_poseidon_core::serialization::injective_bytes_to_felts::<
        qp_poseidon_core::serialization::p2_backend::GF,
    >(&input2);
    println!("=== CPU DEBUG for 4-byte test ===");
    println!("Input: {:?}", input2);
    println!("Expected field elements after injective padding:");
    for (i, felt) in felts2.iter().enumerate() {
        println!("  felt[{}] = {} (0x{:08x})", i, felt.0, felt.0);
    }
    println!(
        "Should be: [0x04030201, 0x00000001] -> [{}, 1]",
        u32::from_le_bytes([1, 2, 3, 4])
    );

    // Test vector 3: Incremental pattern
    let mut input3 = [0u8; 96];
    for i in 0..96 {
        input3[i] = (i % 256) as u8;
    }
    let expected3 = qp_poseidon_core::hash_squeeze_twice(&input3);
    test_vectors.push((input3, expected3));
    println!(
        "Test vector 3 (incremental): input={:?}, expected={:?}",
        &input3[..8],
        &expected3[..8]
    );

    // Test vector 4: Example mining input (header + nonce)
    let mut input4 = [0u8; 96];
    // Header (32 bytes)
    input4[..32].copy_from_slice(&[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    ]);
    // Nonce (64 bytes) - start with simple pattern
    for i in 32..96 {
        input4[i] = ((i - 32) % 256) as u8;
    }
    let expected4 = qp_poseidon_core::hash_squeeze_twice(&input4);
    test_vectors.push((input4, expected4));
    println!(
        "Test vector 4 (mining): input={:?}, expected={:?}",
        &input4[..8],
        &expected4[..8]
    );

    test_vectors
}

fn u8_array_to_u32_array(bytes: &[u8; 96]) -> [u32; 24] {
    let mut result = [0u32; 24];
    for i in 0..24 {
        let idx = i * 4;
        result[i] =
            u32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
    }
    result
}

fn u8_array_to_u32_array_64(bytes: &[u8; 64]) -> [u32; 16] {
    let mut result = [0u32; 16];
    for i in 0..16 {
        let idx = i * 4;
        result[i] =
            u32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
    }
    result
}

async fn test_gpu_with_vectors(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bind_group_layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
    test_vectors: &[([u8; 96], [u8; 64])],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nTesting GPU implementation with known vectors...");

    for (i, (input_bytes, expected_bytes)) in test_vectors.iter().enumerate() {
        println!("\n--- Testing Vector {} ---", i + 1);

        let input_u32 = u8_array_to_u32_array(input_bytes);
        let expected_u32 = u8_array_to_u32_array_64(expected_bytes);

        // Create test buffers
        let test_results_data = vec![0u32; 33];
        let test_results_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Results Buffer"),
            contents: bytemuck::cast_slice(&test_results_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Create debug buffer for intermediate state logging (250 u32s should be enough)
        let debug_data = vec![0u32; 250];
        let debug_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Debug Buffer"),
            contents: bytemuck::cast_slice(&debug_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Split input into header and nonce parts
        let header_data = [
            input_u32[0],
            input_u32[1],
            input_u32[2],
            input_u32[3],
            input_u32[4],
            input_u32[5],
            input_u32[6],
            input_u32[7],
        ];
        let nonce_data = [
            input_u32[8],
            input_u32[9],
            input_u32[10],
            input_u32[11],
            input_u32[12],
            input_u32[13],
            input_u32[14],
            input_u32[15],
            input_u32[16],
            input_u32[17],
            input_u32[18],
            input_u32[19],
            input_u32[20],
            input_u32[21],
            input_u32[22],
            input_u32[23],
        ];

        let test_header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Header Buffer"),
            contents: bytemuck::cast_slice(&header_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let test_nonce_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Nonce Buffer"),
            contents: bytemuck::cast_slice(&nonce_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Use max target so any hash will be "valid"
        let max_target_data = [0xFFFFFFFFu32; 16];
        let test_target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Target Buffer"),
            contents: bytemuck::cast_slice(&max_target_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let test_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: test_results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: test_header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: test_nonce_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: test_target_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: debug_buffer.as_entire_binding(),
                },
            ],
        });

        // Run single thread (thread 0 will use exact nonce)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Test Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &test_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // Just 1 workgroup = 64 threads, thread 0 uses exact nonce
        }
        queue.submit(Some(encoder.finish()));

        // Read results
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Staging Buffer"),
            size: test_results_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Copy Command Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &test_results_buffer,
            0,
            &staging,
            0,
            test_results_buffer.size(),
        );
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        let mapped = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();

        if result[0] == 1 {
            let gpu_hash = &result[17..33];
            println!("GPU result: {:?}", &gpu_hash[..4]);
            println!("Expected:   {:?}", &expected_u32[..4]);

            let matches = gpu_hash
                .iter()
                .zip(expected_u32.iter())
                .all(|(a, b)| a == b);
            if matches {
                println!("✅ Test vector {} PASSED", i + 1);
            } else {
                println!("❌ Test vector {} FAILED", i + 1);
                println!("GPU hash: {:?}", gpu_hash);
                println!("Expected: {:?}", expected_u32);

                // Read debug buffer to compare intermediate states
                let debug_staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Debug Staging Buffer"),
                    size: debug_buffer.size(),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut debug_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Debug Copy Command Encoder"),
                    });
                debug_encoder.copy_buffer_to_buffer(
                    &debug_buffer,
                    0,
                    &debug_staging,
                    0,
                    debug_buffer.size(),
                );
                queue.submit(Some(debug_encoder.finish()));

                let debug_slice = debug_staging.slice(..);
                debug_slice.map_async(wgpu::MapMode::Read, |_| ());
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .unwrap();
                let debug_mapped = debug_slice.get_mapped_range();
                let debug_result: Vec<u32> = bytemuck::cast_slice(&debug_mapped).to_vec();
                drop(debug_mapped);
                debug_staging.unmap();

                println!("=== SIMPLIFIED DEBUG TEST ===");
                println!("Global marker (should be 9999): {}", debug_result[0]);
                println!("Thread ID: {}", debug_result[1]);

                if debug_result.len() > 8 {
                    println!("Thread 0 tests:");
                    println!("  Thread 0 marker (should be 1111): {}", debug_result[2]);
                    println!("  Simple value (should be 2): {}", debug_result[3]);
                    println!("  Simple arithmetic 1+1 (should be 2): {}", debug_result[4]);
                    println!(
                        "  gf_one() result: ({}, {})",
                        debug_result[5], debug_result[6]
                    );
                    println!(
                        "  gf_add(one, one) result: ({}, {})",
                        debug_result[7], debug_result[8]
                    );
                }

                if debug_result.len() > 12 {
                    println!("S-box and permutation tests:");
                    println!("  S-box(2): ({}, {})", debug_result[9], debug_result[10]);
                    println!("  S-box(1): ({}, {})", debug_result[11], debug_result[12]);
                }

                if debug_result.len() > 98 {
                    println!("MDS matrix test:");
                    println!("  Initial state (first 4 elements):");
                    for i in 0..4 {
                        println!(
                            "    Element {}: ({}, {})",
                            i,
                            debug_result[80 + i * 2],
                            debug_result[80 + i * 2 + 1]
                        );
                    }
                    println!("  After MDS matrix (first 4 elements):");
                    for i in 0..4 {
                        println!(
                            "    Element {}: ({}, {})",
                            i,
                            debug_result[90 + i * 2],
                            debug_result[90 + i * 2 + 1]
                        );
                    }
                }

                if debug_result.len() > 101 {
                    println!("Round constant addition test:");
                    println!(
                        "  1 + INTERNAL_CONSTANTS[0]: ({}, {})",
                        debug_result[100], debug_result[101]
                    );
                }

                if debug_result.len() > 148 {
                    println!("Step-by-step Poseidon2 permutation test:");
                    println!("  Initial state:");
                    for i in 0..4 {
                        println!(
                            "    Element {}: ({}, {})",
                            i,
                            debug_result[110 + i * 2],
                            debug_result[110 + i * 2 + 1]
                        );
                    }
                    println!("  After adding round constants:");
                    for i in 0..4 {
                        println!(
                            "    Element {}: ({}, {})",
                            i,
                            debug_result[120 + i * 2],
                            debug_result[120 + i * 2 + 1]
                        );
                    }
                    println!("  After S-box:");
                    for i in 0..4 {
                        println!(
                            "    Element {}: ({}, {})",
                            i,
                            debug_result[130 + i * 2],
                            debug_result[130 + i * 2 + 1]
                        );
                    }
                    println!("  After linear layer:");
                    for i in 0..4 {
                        println!(
                            "    Element {}: ({}, {})",
                            i,
                            debug_result[140 + i * 2],
                            debug_result[140 + i * 2 + 1]
                        );
                    }
                }

                if debug_result.len() > 68 {
                    println!("Input conversion debug:");
                    println!("  Raw input bytes: {:?}", &debug_result[50..58]);
                    println!(
                        "  Field element 0: ({}, {})",
                        debug_result[60], debug_result[61]
                    );
                    println!(
                        "  Field element 1: ({}, {})",
                        debug_result[62], debug_result[63]
                    );
                    println!(
                        "  Field element 2: ({}, {})",
                        debug_result[64], debug_result[65]
                    );
                    println!(
                        "  Field element 3: ({}, {})",
                        debug_result[66], debug_result[67]
                    );
                }

                if debug_result.len() > 26 {
                    println!(
                        "  gf_add inputs: a=({}, {}), b=({}, {})",
                        debug_result[20], debug_result[21], debug_result[22], debug_result[23]
                    );
                    println!(
                        "  gf_add outputs: sum=({}, {}), carry={}",
                        debug_result[24], debug_result[25], debug_result[26]
                    );
                }

                // Print debug information
                println!("=== GPU DEBUG STATES ===");
                println!("Initial state: {:?}", &debug_result[0..8]);
                println!("Input felts: {:?}", &debug_result[24..32]);
                println!(
                    "bytes_to_field_elements returned: {:?}",
                    &debug_result[150..174]
                );

                // Manual byte conversion verification
                println!("=== MANUAL BYTE CONVERSION VERIFICATION ===");
                println!("Simple 4-byte test [1,2,3,4]:");
                println!("  GPU felt[0] = {} (should be 67305985)", debug_result[13]);
                println!("  GPU felt[1] = {} (should be 1)", debug_result[14]);

                println!("96-byte zeros test:");
                println!("  GPU felt[23] = {} (should be 0)", debug_result[15]);
                println!("  GPU felt[24] = {} (should be 1)", debug_result[16]);

                println!(
                    "After 1st chunk absorption+permutation: {:?}",
                    &debug_result[48..56]
                );
                println!(
                    "After 2nd chunk absorption+permutation: {:?}",
                    &debug_result[72..80]
                );
                println!(
                    "After final absorption (with padding=1): {:?}",
                    &debug_result[96..104]
                );
                println!("After final permutation: {:?}", &debug_result[120..128]);
            }
        } else {
            println!("❌ Test vector {} FAILED - no result computed", i + 1);
        }
    }

    Ok(())
}

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

    // Generate simplified debug test vectors
    let debug_test_vectors = generate_debug_test_vectors();

    // Also generate regular test vectors for comparison
    let test_vectors = generate_test_vectors();

    // Load WGSL shader from external file
    let shader_source = include_str!("mining.wgsl");

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
    // Very easy target for testing
    let target_data = [0xFFFFFFFFu32; 16]; // Maximum target (very easy)
    let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Target Buffer"),
        contents: bytemuck::cast_slice(&target_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Debug buffer for main mining (even though we won't use it much here)
    let main_debug_data = vec![0u32; 250];
    let main_debug_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Main Debug Buffer"),
        contents: bytemuck::cast_slice(&main_debug_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
            // Debug buffer (read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: main_debug_buffer.as_entire_binding(),
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

    // Test with debug vectors first (more detailed logging)
    test_gpu_with_vectors(
        &device,
        &queue,
        &bind_group_layout,
        &pipeline,
        &debug_test_vectors,
    )
    .await?;

    println!("\n=== Running regular test vectors for comparison ===");

    // Test with known vectors
    test_gpu_with_vectors(
        &device,
        &queue,
        &bind_group_layout,
        &pipeline,
        &test_vectors,
    )
    .await?;

    // Run gf_mul tests
    if let Err(e) = tests::test_gf_mul(&device, &queue).await {
        eprintln!("gf_mul tests failed: {}", e);
    }

    Ok(())
}
