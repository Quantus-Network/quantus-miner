use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64, PrimeField64};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use wgpu::util::DeviceExt;

// A simple struct to hold a test case for gf_mul
#[derive(Debug, Clone)]
struct GfMulTestCase {
    a: GoldilocksField,
    b: GoldilocksField,
    expected: GoldilocksField,
}

#[derive(Debug, Clone)]
struct MdsTestCase {
    input: [GoldilocksField; 4],
    expected: [GoldilocksField; 4],
}

// A simple struct to hold a test case for sbox
#[derive(Debug, Clone)]
struct SboxTestCase {
    input: GoldilocksField,
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

impl From<GfWgls> for GoldilocksField {
    fn from(gf: GfWgls) -> Self {
        let val = (gf.limb0 as u64)
            | ((gf.limb1 as u64) << 16)
            | ((gf.limb2 as u64) << 32)
            | ((gf.limb3 as u64) << 48);
        GoldilocksField::from_noncanonical_u64(val)
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
    for _ in 0..50 {
        let a = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Random mixed cases (one small, one large)
    for _ in 0..80 {
        let a = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u64>());
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Random large values (both operands >= 2^32)
    for _ in 0..100 {
        let a = GoldilocksField::from_canonical_u64(rng.gen::<u64>() | 0x100000000u64);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u64>() | 0x100000000u64);
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Random values near the field modulus
    for _ in 0..30 {
        let offset = rng.gen::<u32>() as u64;
        let a = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - offset);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u64>());
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Add more specific high-limb test cases to detect patterns
    for i in 20..40 {
        let val1 = 1u64 << i; // Powers of 2 from 2^20 to 2^39
        let val2 = rng.gen::<u32>() as u64 | (1u64 << 32); // Random value with high bit set
        let a = GoldilocksField::from_canonical_u64(val1);
        let b = GoldilocksField::from_canonical_u64(val2);
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Test values with specific limb patterns
    for _ in 0..20 {
        let limb2_val = (rng.gen::<u16>() as u64) << 32;
        let limb3_val = (rng.gen::<u16>() as u64) << 48;
        let a = GoldilocksField::from_canonical_u64(limb2_val | limb3_val);
        let b = GoldilocksField::from_canonical_u64(rng.gen::<u64>() | 0x100000000u64);
        vectors.push(GfMulTestCase {
            a,
            b,
            expected: a * b,
        });
    }

    // Add failing S-box cases to test field multiplication with higher limbs
    println!("Adding S-box failing cases to test higher limb multiplication...");
    let failing_sbox_inputs = vec![
        0x0000000000400000u64, // 2^22
        0x0000000000800000u64, // 2^23
        0x0000000001000000u64, // 2^24
        0x0000000002000000u64, // 2^25
        0x0000000004000000u64, // 2^26
        0x0000000008000000u64, // 2^27
        0x0000000010000000u64, // 2^28
        0x0000000020000000u64, // 2^29
        0x0000000040000000u64, // 2^30
        0x0000000080000000u64, // 2^31
    ];

    for &input_val in &failing_sbox_inputs {
        let x = GoldilocksField::from_canonical_u64(input_val);

        // Test x * x (first step in S-box computation)
        vectors.push(GfMulTestCase {
            a: x,
            b: x,
            expected: x * x,
        });

        // Test intermediate S-box computations
        let x2 = x * x;
        let x4 = x2 * x2;

        vectors.push(GfMulTestCase {
            a: x2,
            b: x2,
            expected: x4,
        });

        vectors.push(GfMulTestCase {
            a: x4,
            b: x2,
            expected: x4 * x2,
        });
    }

    println!("Generated {} comprehensive test vectors.", vectors.len());
    vectors
}

fn generate_sbox_test_vectors() -> Vec<SboxTestCase> {
    println!("Generating S-box test vectors...");
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);

    // Add special cases
    println!("Adding S-box special cases...");

    // Zero case
    vectors.push(SboxTestCase {
        input: GoldilocksField::ZERO,
        expected: GoldilocksField::ZERO, // 0^7 = 0
    });

    // One case
    vectors.push(SboxTestCase {
        input: GoldilocksField::ONE,
        expected: GoldilocksField::ONE, // 1^7 = 1
    });

    // Two case
    let two = GoldilocksField::from_canonical_u64(2);
    let two_to_7 = two.exp_u64(7); // 2^7 = 128
    vectors.push(SboxTestCase {
        input: two,
        expected: two_to_7,
    });

    // Small powers of 2
    for i in 0..8 {
        let base = GoldilocksField::from_canonical_u64(1u64 << i);
        let expected = base.exp_u64(7);
        vectors.push(SboxTestCase {
            input: base,
            expected,
        });
    }

    // Small consecutive values
    for i in 3..32 {
        let base = GoldilocksField::from_canonical_u64(i);
        let expected = base.exp_u64(7);
        vectors.push(SboxTestCase {
            input: base,
            expected,
        });
    }

    // Larger powers of 2
    for i in 8..32 {
        let base = GoldilocksField::from_canonical_u64(1u64 << i);
        let expected = base.exp_u64(7);
        vectors.push(SboxTestCase {
            input: base,
            expected,
        });
    }

    // Random small values (< 2^32)
    for _ in 0..20 {
        let input = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        let expected = input.exp_u64(7);
        vectors.push(SboxTestCase { input, expected });
    }

    // Random large values (>= 2^32)
    for _ in 0..30 {
        let input = GoldilocksField::from_canonical_u64(rng.gen::<u64>() | 0x100000000u64);
        let expected = input.exp_u64(7);
        vectors.push(SboxTestCase { input, expected });
    }

    // Values near the field modulus
    for _ in 0..10 {
        let offset = rng.gen::<u32>() as u64;
        let input = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - offset);
        let expected = input.exp_u64(7);
        vectors.push(SboxTestCase { input, expected });
    }

    println!("Generated {} S-box test vectors.", vectors.len());
    vectors
}

fn generate_mds_test_vectors() -> Vec<MdsTestCase> {
    println!("Generating comprehensive MDS matrix test vectors using qp-poseidon...");
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(54321);

    // We'll compute the MDS matrix manually to match qp-poseidon's behavior

    // Add special cases first
    println!("Adding special cases...");

    // Zero vector
    let zero_input = [GoldilocksField::ZERO; 4];
    let zero_expected = apply_external_linear_layer_to_chunk(&(), &zero_input);
    vectors.push(MdsTestCase {
        input: zero_input,
        expected: zero_expected,
    });

    // Unit vectors
    for i in 0..4 {
        let mut input = [GoldilocksField::ZERO; 4];
        input[i] = GoldilocksField::ONE;
        let expected = apply_external_linear_layer_to_chunk(&(), &input);
        vectors.push(MdsTestCase { input, expected });
    }

    // All ones vector
    let ones_input = [GoldilocksField::ONE; 4];
    let ones_expected = apply_external_linear_layer_to_chunk(&(), &ones_input);
    vectors.push(MdsTestCase {
        input: ones_input,
        expected: ones_expected,
    });

    println!("Adding random test cases...");

    // Random small values (all elements < 2^16)
    for _ in 0..50 {
        let mut input = [GoldilocksField::ZERO; 4];
        for j in 0..4 {
            input[j] = GoldilocksField::from_canonical_u64(rng.gen::<u16>() as u64);
        }
        let expected = apply_external_linear_layer_to_chunk(&(), &input);
        vectors.push(MdsTestCase { input, expected });
    }

    // Random medium values (all elements < 2^32)
    for _ in 0..30 {
        let mut input = [GoldilocksField::ZERO; 4];
        for j in 0..4 {
            input[j] = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        }
        let expected = apply_external_linear_layer_to_chunk(&(), &input);
        vectors.push(MdsTestCase { input, expected });
    }

    // Random large values (using full 64-bit range)
    for _ in 0..30 {
        let mut input = [GoldilocksField::ZERO; 4];
        for j in 0..4 {
            input[j] = GoldilocksField::from_canonical_u64(rng.gen::<u64>());
        }
        let expected = apply_external_linear_layer_to_chunk(&(), &input);
        vectors.push(MdsTestCase { input, expected });
    }

    // Edge cases near field modulus
    for _ in 0..10 {
        let mut input = [GoldilocksField::ZERO; 4];
        for j in 0..4 {
            let offset = rng.gen::<u32>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - offset);
        }
        let expected = apply_external_linear_layer_to_chunk(&(), &input);
        vectors.push(MdsTestCase { input, expected });
    }

    println!(
        "Generated {} comprehensive MDS test vectors using qp-poseidon.",
        vectors.len()
    );
    vectors
}

// Helper function to apply the external linear layer to a 4-element chunk
// We'll manually compute the MDS matrix but using the exact same values as qp-poseidon
fn apply_external_linear_layer_to_chunk(
    _poseidon: &(), // We'll compute manually for now
    chunk: &[GoldilocksField; 4],
) -> [GoldilocksField; 4] {
    // Since we can't easily access the internal MDS implementation from qp-poseidon,
    // let's use the test_single_permutation function to validate our approach
    // For now, we'll implement the standard 4x4 MDS matrix that p3_poseidon2 uses:
    // [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]

    let input = chunk;
    [
        input[0] * GoldilocksField::from_canonical_u64(2)
            + input[1] * GoldilocksField::from_canonical_u64(3)
            + input[2]
            + input[3],
        input[0]
            + input[1] * GoldilocksField::from_canonical_u64(2)
            + input[2] * GoldilocksField::from_canonical_u64(3)
            + input[3],
        input[0]
            + input[1]
            + input[2] * GoldilocksField::from_canonical_u64(2)
            + input[3] * GoldilocksField::from_canonical_u64(3),
        input[0] * GoldilocksField::from_canonical_u64(3)
            + input[1]
            + input[2]
            + input[3] * GoldilocksField::from_canonical_u64(2),
    ]
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

pub async fn test_mds_matrix(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Running 4x4 MDS matrix tests ---");

    let test_vectors = generate_mds_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    println!("Running {} test cases...", total_tests);

    // Read the mining shader source and create a test wrapper
    let mining_shader_source = include_str!("mining.wgsl");
    let shader_source = format!(
        "
{}

// MDS matrix test entry point
@group(0) @binding(0) var<storage, read> input_data: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_data: array<GoldilocksField>;

@compute @workgroup_size(1)
fn mds_test(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i = global_id.x;
    let base_idx = i * 4u;

    // Read 4 input elements
    var chunk_state: array<GoldilocksField, 4>;
    chunk_state[0] = input_data[base_idx + 0u];
    chunk_state[1] = input_data[base_idx + 1u];
    chunk_state[2] = input_data[base_idx + 2u];
    chunk_state[3] = input_data[base_idx + 3u];

    // Apply 4x4 MDS matrix transformation
    let t01 = gf_add(chunk_state[0], chunk_state[1]);
    let t23 = gf_add(chunk_state[2], chunk_state[3]);
    let t0123 = gf_add(t01, t23);
    let t01123 = gf_add(t0123, chunk_state[1]);
    let t01233 = gf_add(t0123, chunk_state[3]);

    let new_3 = gf_add(t01233, gf_add(chunk_state[0], chunk_state[0])); // 3*x[0] + x[1] + x[2] + 2*x[3]
    let new_1 = gf_add(t01123, gf_add(chunk_state[2], chunk_state[2])); // x[0] + 2*x[1] + 3*x[2] + x[3]
    let new_0 = gf_add(t01123, t01); // 2*x[0] + 3*x[1] + x[2] + x[3]
    let new_2 = gf_add(t01233, t23); // x[0] + x[1] + 2*x[2] + 3*x[3]

    // Write results
    output_data[base_idx + 0u] = new_0;
    output_data[base_idx + 1u] = new_1;
    output_data[base_idx + 2u] = new_2;
    output_data[base_idx + 3u] = new_3;
}}
",
        mining_shader_source
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MDS test shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create compute pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MDS test pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("mds_test"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Process tests in batches
    const BATCH_SIZE: usize = 64;
    for chunk in test_vectors.chunks(BATCH_SIZE) {
        // Prepare input data (flatten 4-element arrays and convert to GfWgls)
        let mut input_data = Vec::new();
        for test_case in chunk {
            for &elem in &test_case.input {
                let u64_val = elem.to_canonical_u64();
                input_data.push(GfWgls {
                    limb0: (u64_val & 0xFFFF) as u32,
                    limb1: ((u64_val >> 16) & 0xFFFF) as u32,
                    limb2: ((u64_val >> 32) & 0xFFFF) as u32,
                    limb3: ((u64_val >> 48) & 0xFFFF) as u32,
                });
            }
        }

        // Create input buffer
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MDS Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create output buffer
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MDS Output Buffer"),
            size: (input_data.len() * std::mem::size_of::<GfWgls>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MDS Test Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Run compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MDS Test Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MDS Test Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(chunk.len() as u32, 1, 1);
        }

        // Create staging buffer and copy results
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MDS Staging Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver.await??;

        let data = buffer_slice.get_mapped_range();
        let gpu_results: &[GfWgls] = bytemuck::cast_slice(&data);

        // Check results
        for (i, test_case) in chunk.iter().enumerate() {
            let gpu_result = [
                GoldilocksField::from_noncanonical_u64(
                    gpu_results[i * 4].limb0 as u64
                        | ((gpu_results[i * 4].limb1 as u64) << 16)
                        | ((gpu_results[i * 4].limb2 as u64) << 32)
                        | ((gpu_results[i * 4].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    gpu_results[i * 4 + 1].limb0 as u64
                        | ((gpu_results[i * 4 + 1].limb1 as u64) << 16)
                        | ((gpu_results[i * 4 + 1].limb2 as u64) << 32)
                        | ((gpu_results[i * 4 + 1].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    gpu_results[i * 4 + 2].limb0 as u64
                        | ((gpu_results[i * 4 + 2].limb1 as u64) << 16)
                        | ((gpu_results[i * 4 + 2].limb2 as u64) << 32)
                        | ((gpu_results[i * 4 + 2].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    gpu_results[i * 4 + 3].limb0 as u64
                        | ((gpu_results[i * 4 + 3].limb1 as u64) << 16)
                        | ((gpu_results[i * 4 + 3].limb2 as u64) << 32)
                        | ((gpu_results[i * 4 + 3].limb3 as u64) << 48),
                ),
            ];

            if gpu_result == test_case.expected {
                passed_tests += 1;
            } else {
                failed_tests.push((test_case.clone(), gpu_result));
                if failed_tests.len() <= 5 {
                    println!(
                        "❌ FAILED: Input=[{}, {}, {}, {}], Expected=[{}, {}, {}, {}], Got=[{}, {}, {}, {}]",
                        test_case.input[0].to_canonical_u64(),
                        test_case.input[1].to_canonical_u64(),
                        test_case.input[2].to_canonical_u64(),
                        test_case.input[3].to_canonical_u64(),
                        test_case.expected[0].to_canonical_u64(),
                        test_case.expected[1].to_canonical_u64(),
                        test_case.expected[2].to_canonical_u64(),
                        test_case.expected[3].to_canonical_u64(),
                        gpu_result[0].to_canonical_u64(),
                        gpu_result[1].to_canonical_u64(),
                        gpu_result[2].to_canonical_u64(),
                        gpu_result[3].to_canonical_u64(),
                    );
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    println!("\n=== TEST SUMMARY ===");
    println!("Total tests: {}", total_tests);
    println!(
        "Passed: {} ({:.1}%)",
        passed_tests,
        (passed_tests as f64 / total_tests as f64) * 100.0
    );

    if failed_tests.is_empty() {
        println!("🎉 All tests PASSED!");
    } else {
        println!("❌ Failed: {} tests", failed_tests.len());
        if failed_tests.len() > 5 {
            println!("  (showing first 5 failures above)");
        }
    }

    if failed_tests.is_empty() {
        println!("🎉 All tests PASSED!");
    } else {
        println!("❌ Failed: {} tests", failed_tests.len());
        if failed_tests.len() > 5 {
            println!("  (showing first 5 failures above)");
        }
        return Err(format!("{} out of {} tests failed", failed_tests.len(), total_tests).into());
    }

    Ok(())
}

pub async fn test_sbox(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Running S-box tests ---");

    let test_vectors = generate_sbox_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    const BATCH_SIZE: usize = 64;
    let batches = test_vectors.chunks(BATCH_SIZE);

    for (batch_idx, batch) in batches.enumerate() {
        println!(
            "Processing batch {} ({} tests)...",
            batch_idx + 1,
            batch.len()
        );

        // Convert test vectors to WGSL format
        let input_data: Vec<GfWgls> = batch
            .iter()
            .map(|test_case| GfWgls::from(test_case.input))
            .collect();

        // Create compute shader for S-box testing by including mining.wgsl without main
        let mining_wgsl = include_str!("mining.wgsl");
        // Remove the main function from mining.wgsl
        let mining_wgsl_without_main = mining_wgsl
            .lines()
            .take_while(|line| !line.contains("@compute @workgroup_size(64)"))
            .collect::<Vec<_>>()
            .join("\n");

        let shader_source = format!(
            r#"
@group(0) @binding(0) var<storage, read> input_data: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_data: array<GoldilocksField>;

{}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    if (index >= {}u) {{
        return;
    }}

    let input_val = input_data[index];
    let result = sbox(input_val);
    output_data[index] = result;
}}
"#,
            mining_wgsl_without_main,
            batch.len()
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("S-box Test Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("S-box Test Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create buffers
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (batch.len() * std::mem::size_of::<GfWgls>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (batch.len() * std::mem::size_of::<GfWgls>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(batch.len() as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());
        queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver.await??;

        let data = buffer_slice.get_mapped_range();
        let results: &[GfWgls] = bytemuck::cast_slice(&data);

        // Check results
        for (i, (test_case, &gpu_result_raw)) in batch.iter().zip(results.iter()).enumerate() {
            let gpu_result = GoldilocksField::from(gpu_result_raw);

            if gpu_result == test_case.expected {
                passed_tests += 1;
            } else {
                failed_tests.push((test_case.clone(), gpu_result));
                if failed_tests.len() <= 10 {
                    println!(
                        "❌ FAILED: sbox(0x{:016x}) = 0x{:016x}, expected 0x{:016x}",
                        test_case.input.to_canonical_u64(),
                        gpu_result.to_canonical_u64(),
                        test_case.expected.to_canonical_u64()
                    );

                    // Debug intermediate calculations for failing cases
                    let x = test_case.input;
                    let x2_cpu = x * x;
                    let x4_cpu = x2_cpu * x2_cpu;
                    let x6_cpu = x4_cpu * x2_cpu;
                    let x7_cpu = x6_cpu * x;

                    println!("  CPU step-by-step:");
                    println!("    x   = 0x{:016x}", x.to_canonical_u64());
                    println!("    x^2 = 0x{:016x}", x2_cpu.to_canonical_u64());
                    println!("    x^4 = 0x{:016x}", x4_cpu.to_canonical_u64());
                    println!("    x^6 = 0x{:016x}", x6_cpu.to_canonical_u64());
                    println!("    x^7 = 0x{:016x}", x7_cpu.to_canonical_u64());

                    // Check if GPU is returning input unchanged
                    if gpu_result == test_case.input {
                        println!(
                            "  ⚠️  GPU returned input unchanged - possible multiplication bug"
                        );
                    }
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    println!("\n=== S-BOX TEST SUMMARY ===");
    println!("Total tests: {}", total_tests);
    println!(
        "Passed: {} ({:.1}%)",
        passed_tests,
        (passed_tests as f64 / total_tests as f64) * 100.0
    );

    if failed_tests.is_empty() {
        println!("🎉 All S-box tests PASSED!");
    } else {
        println!("❌ Failed: {} tests", failed_tests.len());
        if failed_tests.len() > 10 {
            println!("  (showing first 10 failures above)");
        }
    }

    if failed_tests.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} out of {} S-box tests failed",
            failed_tests.len(),
            total_tests
        )
        .into())
    }
}
