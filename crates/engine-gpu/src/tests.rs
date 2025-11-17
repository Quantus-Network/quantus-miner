use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64, PrimeField64};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use wgpu::util::DeviceExt;

// A simple struct to hold a test case for gf_mul
struct GfMulTestCase {
    a: GoldilocksField,
    b: GoldilocksField,
    expected: GoldilocksField,
}

// Represents the GoldilocksField in a WGSL-compatible format (two u32s)
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GfWgls {
    low: u32,
    high: u32,
}

impl From<GoldilocksField> for GfWgls {
    fn from(gf: GoldilocksField) -> Self {
        let val = gf.0;
        Self {
            low: val as u32,
            high: (val >> 32) as u32,
        }
    }
}

fn generate_gf_mul_test_vectors() -> Vec<GfMulTestCase> {
    println!("Generating gf_mul test vectors...");
    let mut vectors = Vec::new();

    // Basic edge cases
    println!("Adding basic edge cases...");

    // Case 1: 0 * x = 0
    let a1 = GoldilocksField::ZERO;
    let b1 = GoldilocksField::from_canonical_u64(123456789);
    vectors.push(GfMulTestCase {
        a: a1,
        b: b1,
        expected: a1 * b1,
    });

    // Case 2: 1 * x = x
    let a2 = GoldilocksField::ONE;
    let b2 = GoldilocksField::from_canonical_u64(987654321);
    vectors.push(GfMulTestCase {
        a: a2,
        b: b2,
        expected: a2 * b2,
    });

    // Case 3: x * 1 = x (commutative check)
    let a3 = GoldilocksField::from_noncanonical_u64(0xDEADBEEFCAFEBABE);
    let b3 = GoldilocksField::ONE;
    vectors.push(GfMulTestCase {
        a: a3,
        b: b3,
        expected: a3 * b3,
    });

    // Case 4: Small numbers (no overflow in u32*u32)
    let a4 = GoldilocksField::from_canonical_u64(100);
    let b4 = GoldilocksField::from_canonical_u64(200);
    vectors.push(GfMulTestCase {
        a: a4,
        b: b4,
        expected: a4 * b4,
    });

    // Boundary cases
    println!("Adding boundary cases...");

    // Case 5: u32::MAX
    let a5 = GoldilocksField::from_canonical_u64(u32::MAX as u64);
    let b5 = GoldilocksField::from_canonical_u64(u32::MAX as u64);
    vectors.push(GfMulTestCase {
        a: a5,
        b: b5,
        expected: a5 * b5,
    });

    // Case 6: Just above u32::MAX
    let a6 = GoldilocksField::from_canonical_u64(u32::MAX as u64 + 1);
    let b6 = GoldilocksField::from_canonical_u64(2);
    vectors.push(GfMulTestCase {
        a: a6,
        b: b6,
        expected: a6 * b6,
    });

    // Case 7: Powers of 2
    let a7 = GoldilocksField::from_canonical_u64(1u64 << 32);
    let b7 = GoldilocksField::from_canonical_u64(1u64 << 31);
    vectors.push(GfMulTestCase {
        a: a7,
        b: b7,
        expected: a7 * b7,
    });

    // Case 8: Large powers of 2
    let a8 = GoldilocksField::from_canonical_u64(1u64 << 62);
    let b8 = GoldilocksField::from_canonical_u64(1u64 << 1);
    vectors.push(GfMulTestCase {
        a: a8,
        b: b8,
        expected: a8 * b8,
    });

    // Field modulus edge cases
    println!("Adding field modulus edge cases...");

    // Case 9: Field modulus - 1
    let a9 = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - 1);
    let b9 = GoldilocksField::from_canonical_u64(2);
    vectors.push(GfMulTestCase {
        a: a9,
        b: b9,
        expected: a9 * b9,
    });

    // Case 10: Field modulus - 1 squared
    let a10 = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - 1);
    let b10 = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - 1);
    vectors.push(GfMulTestCase {
        a: a10,
        b: b10,
        expected: a10 * b10,
    });

    // Case 11: Large numbers that will cause reduction
    let a11 = GoldilocksField::from_noncanonical_u64(0xABCDEF1234567890);
    let b11 = GoldilocksField::from_noncanonical_u64(0x1122334455667788);
    vectors.push(GfMulTestCase {
        a: a11,
        b: b11,
        expected: a11 * b11,
    });

    // Mixed size cases
    println!("Adding mixed size cases...");

    // Case 12: Small * Large
    let a12 = GoldilocksField::from_canonical_u64(3);
    let b12 = GoldilocksField::from_noncanonical_u64(0xFFFFFFFFFFFFFFF0);
    vectors.push(GfMulTestCase {
        a: a12,
        b: b12,
        expected: a12 * b12,
    });

    // Case 13: Large * Small
    let a13 = GoldilocksField::from_noncanonical_u64(0xFFFFFFFFFFFFFFF0);
    let b13 = GoldilocksField::from_canonical_u64(7);
    vectors.push(GfMulTestCase {
        a: a13,
        b: b13,
        expected: a13 * b13,
    });

    // Random test cases
    println!("Adding random test cases...");

    // Use a fixed seed for reproducible tests
    let mut rng = ChaCha8Rng::seed_from_u64(0x123456789ABCDEF0);

    // Generate 50 random test cases
    for _ in 0..50 {
        // Use from_noncanonical_u64 to handle any u64 value safely
        let a_val = rng.gen::<u64>();
        let b_val = rng.gen::<u64>();

        let a = GoldilocksField::from_noncanonical_u64(a_val);
        let b = GoldilocksField::from_noncanonical_u64(b_val);

        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Special random cases focusing on problematic ranges
    println!("Adding focused random cases...");

    // Cases where one operand has high=0, other has high!=0
    for _ in 0..10 {
        let small_val = rng.gen::<u32>() as u64;
        // Generate a large value and use from_noncanonical_u64 to handle safely
        let large_val = rng.gen_range((1u64 << 32)..u64::MAX);

        let a = GoldilocksField::from_canonical_u64(small_val);
        let b = GoldilocksField::from_noncanonical_u64(large_val);

        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Cases where both operands are large (both high!=0)
    for _ in 0..10 {
        let a_val = rng.gen_range((1u64 << 32)..u64::MAX);
        let b_val = rng.gen_range((1u64 << 32)..u64::MAX);

        let a = GoldilocksField::from_noncanonical_u64(a_val);
        let b = GoldilocksField::from_noncanonical_u64(b_val);

        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Cases near field boundaries
    for _ in 0..5 {
        let offset = rng.gen_range(1..1000);
        let near_max = GoldilocksField::ORDER - offset;
        let other = rng.gen_range(2..100);

        let a = GoldilocksField::from_canonical_u64(near_max);
        let b = GoldilocksField::from_canonical_u64(other);

        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    println!("Generated {} total test vectors.", vectors.len());
    vectors
}

pub async fn test_gf_mul(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Running gf_mul tests ---");

    let test_vectors = generate_gf_mul_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    println!("Running {} test cases...", total_tests);

    // Load the full mining shader code
    let shader_source = include_str!("gf_mul_test.wgsl");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gf_mul test shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gf_mul test pipeline"),
        layout: None, // Let wgpu infer the layout
        module: &shader,
        entry_point: Some("gf_mul_test"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    for (i, vector) in test_vectors.iter().enumerate() {
        let a_wgls: GfWgls = vector.a.into();
        let b_wgls: GfWgls = vector.b.into();

        let input_a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input A Buffer"),
            contents: bytemuck::cast_slice(&[a_wgls]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let input_b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input B Buffer"),
            contents: bytemuck::cast_slice(&[b_wgls]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: std::mem::size_of::<GfWgls>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let data = slice.get_mapped_range();
        let result_wgls: GfWgls = bytemuck::from_bytes::<GfWgls>(&data).clone();
        drop(data);
        staging_buffer.unmap();

        let gpu_result_u64 = (result_wgls.high as u64) << 32 | (result_wgls.low as u64);
        let gpu_result = GoldilocksField(gpu_result_u64);

        let expected_wgls: GfWgls = vector.expected.into();

        // Progress indicator every 10 tests
        if i % 10 == 0 || i < 20 {
            println!(
                "Progress: {}/{} ({:.1}%)",
                i + 1,
                total_tests,
                (i + 1) as f32 / total_tests as f32 * 100.0
            );
        }

        // The GPU result might not be canonical, so we need to canonicalize it before comparing.
        if gpu_result.to_canonical_u64() == vector.expected.to_canonical_u64() {
            passed_tests += 1;

            // Only show details for first few tests or if verbose mode
            if i < 5 {
                println!("Test case {} ✅ PASSED", i + 1);
                println!(
                    "  a: 0x{:016x}, b: 0x{:016x} = 0x{:016x}",
                    vector.a.0, vector.b.0, vector.expected.0
                );
            }
        } else {
            failed_tests.push(i + 1);
            println!("Test case {} ❌ FAILED", i + 1);
            println!(
                "  a: 0x{:016x} ({}, {})",
                vector.a.0, a_wgls.low, a_wgls.high
            );
            println!(
                "  b: 0x{:016x} ({}, {})",
                vector.b.0, b_wgls.low, b_wgls.high
            );
            println!(
                "  CPU expected: 0x{:016x} ({}, {})",
                vector.expected.0, expected_wgls.low, expected_wgls.high
            );
            println!(
                "  GPU result:   0x{:016x} ({}, {})",
                gpu_result.0, result_wgls.low, result_wgls.high
            );
        }
    }

    // Print final summary
    println!("\n=== TEST SUMMARY ===");
    println!("Total tests: {}", total_tests);
    println!(
        "Passed: {} ({:.1}%)",
        passed_tests,
        passed_tests as f32 / total_tests as f32 * 100.0
    );

    if failed_tests.is_empty() {
        println!("🎉 All tests PASSED!");
    } else {
        println!(
            "Failed: {} ({:.1}%)",
            failed_tests.len(),
            failed_tests.len() as f32 / total_tests as f32 * 100.0
        );
        println!("Failed test cases: {:?}", failed_tests);
        return Err(format!("{} out of {} tests failed", failed_tests.len(), total_tests).into());
    }

    Ok(())
}
