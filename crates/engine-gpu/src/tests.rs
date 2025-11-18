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

// Represents the GoldilocksField in a WGSL-compatible format (four u16 limbs stored as u32s)
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GfWgls {
    limb0: u32,
    limb1: u32,
    limb2: u32,
    limb3: u32,
}

impl From<GoldilocksField> for GfWgls {
    fn from(gf: GoldilocksField) -> Self {
        let val = gf.0;
        Self {
            limb0: (val & 0xFFFF) as u32,
            limb1: ((val >> 16) & 0xFFFF) as u32,
            limb2: ((val >> 32) & 0xFFFF) as u32,
            limb3: ((val >> 48) & 0xFFFF) as u32,
        }
    }
}

fn generate_gf_mul_test_vectors() -> Vec<GfMulTestCase> {
    println!("Generating comprehensive test vectors...");
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);

    // Add the specific failing cases from the log that we fixed
    println!("Adding previously failing cases for regression testing...");

    // Test case 1: 0x19a071bdddc16e57 * 0x9f34cc3172d67f12
    let a1 = GoldilocksField::from_noncanonical_u64(0x19a071bdddc16e57);
    let b1 = GoldilocksField::from_noncanonical_u64(0x9f34cc3172d67f12);
    vectors.push(GfMulTestCase {
        a: a1,
        b: b1,
        expected: a1 * b1,
    });

    // Test case 2: 0x7e92900e25221b88 * 0x967d2999df3fa192
    let a2 = GoldilocksField::from_noncanonical_u64(0x7e92900e25221b88);
    let b2 = GoldilocksField::from_noncanonical_u64(0x967d2999df3fa192);
    vectors.push(GfMulTestCase {
        a: a2,
        b: b2,
        expected: a2 * b2,
    });

    // Test case 3: 0xd258f723b9f7cbc2 * 0x5ddb844a50f2bd61
    let a3 = GoldilocksField::from_noncanonical_u64(0xd258f723b9f7cbc2);
    let b3 = GoldilocksField::from_noncanonical_u64(0x5ddb844a50f2bd61);
    vectors.push(GfMulTestCase {
        a: a3,
        b: b3,
        expected: a3 * b3,
    });

    // Test case 4: 0x679ad356dc569f8a * 0x7a0ef9078a5b4fc5
    let a4 = GoldilocksField::from_noncanonical_u64(0x679ad356dc569f8a);
    let b4 = GoldilocksField::from_noncanonical_u64(0x7a0ef9078a5b4fc5);
    vectors.push(GfMulTestCase {
        a: a4,
        b: b4,
        expected: a4 * b4,
    });

    // Add special cases
    println!("Adding special cases...");

    // Zero cases
    vectors.push(GfMulTestCase {
        a: GoldilocksField::ZERO,
        b: GoldilocksField::from_canonical_u64(123456789),
        expected: GoldilocksField::ZERO,
    });

    // One cases
    let val = GoldilocksField::from_canonical_u64(987654321);
    vectors.push(GfMulTestCase {
        a: GoldilocksField::ONE,
        b: val,
        expected: val,
    });

    // Small values
    let small_a = GoldilocksField::from_canonical_u64(u32::MAX as u64);
    let small_b = GoldilocksField::from_canonical_u64(u32::MAX as u64);
    vectors.push(GfMulTestCase {
        a: small_a,
        b: small_b,
        expected: small_a * small_b,
    });

    // Add random test cases
    println!("Adding random test cases...");

    // Random small values (both operands < 2^32)
    for _ in 0..20 {
        let a = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Random mixed cases (one small, one large)
    for _ in 0..20 {
        let a = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u64>());
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Random large values (both operands >= 2^32)
    for _ in 0..30 {
        let a = GoldilocksField::from_canonical_u64(rng.gen::<u64>() | 0x100000000u64);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u64>() | 0x100000000u64);
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Random values near the field modulus
    for _ in 0..10 {
        let offset = rng.gen::<u32>() as u64;
        let a = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - offset);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u64>());
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    println!("Generated {} comprehensive test vectors.", vectors.len());
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

    // Use the mining shader but create a simple test wrapper
    let shader_source = format!(
        "
{}

// Simple test entry point that just multiplies two field elements
@group(0) @binding(0) var<storage, read> input_a: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read> input_b: array<GoldilocksField>;
@group(0) @binding(2) var<storage, read_write> output: array<GoldilocksField>;

@compute @workgroup_size(1)
fn gf_mul_test(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i = global_id.x;
    output[i] = gf_mul(input_a[i], input_b[i]);
}}
",
        include_str!("mining.wgsl")
    );

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

        let gpu_result_u64 = (result_wgls.limb0 as u64)
            | ((result_wgls.limb1 as u64) << 16)
            | ((result_wgls.limb2 as u64) << 32)
            | ((result_wgls.limb3 as u64) << 48);
        let gpu_result = GoldilocksField(gpu_result_u64);

        let expected_wgls: GfWgls = vector.expected.into();

        // The GPU result might not be canonical, so we need to canonicalize it before comparing.
        if gpu_result.to_canonical_u64() == vector.expected.to_canonical_u64() {
            passed_tests += 1;
        } else {
            failed_tests.push(i + 1);

            // Only show detailed output for failures
            println!(
                "\nTest case {} FAILED: 0x{:016x} * 0x{:016x}",
                i + 1,
                vector.a.0,
                vector.b.0
            );

            // Analyze the operands
            let a_limbs = [
                (vector.a.0 & 0xFFFF) as u32,
                ((vector.a.0 >> 16) & 0xFFFF) as u32,
                ((vector.a.0 >> 32) & 0xFFFF) as u32,
                ((vector.a.0 >> 48) & 0xFFFF) as u32,
            ];
            let b_limbs = [
                (vector.b.0 & 0xFFFF) as u32,
                ((vector.b.0 >> 16) & 0xFFFF) as u32,
                ((vector.b.0 >> 32) & 0xFFFF) as u32,
                ((vector.b.0 >> 48) & 0xFFFF) as u32,
            ];

            println!(
                "  a: [{}, {}, {}, {}]",
                a_limbs[0], a_limbs[1], a_limbs[2], a_limbs[3]
            );
            println!(
                "  b: [{}, {}, {}, {}]",
                b_limbs[0], b_limbs[1], b_limbs[2], b_limbs[3]
            );

            // Determine which multiplication path this should take
            let a_high_nonzero = a_limbs[2] != 0 || a_limbs[3] != 0;
            let b_high_nonzero = b_limbs[2] != 0 || b_limbs[3] != 0;
            let path = if !a_high_nonzero && !b_high_nonzero {
                "Step 1: Both small (high limbs=0)"
            } else if !a_high_nonzero || !b_high_nonzero {
                "Step 2: Mixed case (one has high limbs=0)"
            } else {
                "Step 3: Both large (both have high limbs≠0)"
            };
            println!("  Expected path: {}", path);

            println!(
                "  CPU expected: 0x{:016x} [{}, {}, {}, {}]",
                vector.expected.0,
                expected_wgls.limb0,
                expected_wgls.limb1,
                expected_wgls.limb2,
                expected_wgls.limb3
            );
            println!(
                "  GPU result:   0x{:016x} [{}, {}, {}, {}]",
                gpu_result.0,
                result_wgls.limb0,
                result_wgls.limb1,
                result_wgls.limb2,
                result_wgls.limb3
            );

            // Show the exact differences
            let diff_limb0 = result_wgls.limb0 as i64 - expected_wgls.limb0 as i64;
            let diff_limb1 = result_wgls.limb1 as i64 - expected_wgls.limb1 as i64;
            let diff_limb2 = result_wgls.limb2 as i64 - expected_wgls.limb2 as i64;
            let diff_limb3 = result_wgls.limb3 as i64 - expected_wgls.limb3 as i64;
            println!(
                "  Differences: limb0={:+}, limb1={:+}, limb2={:+}, limb3={:+}",
                diff_limb0, diff_limb1, diff_limb2, diff_limb3
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
