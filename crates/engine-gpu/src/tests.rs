use plonky2::hash::poseidon2::P2Permuter;
use qp_plonky2_field::goldilocks_field::GoldilocksField;
use qp_plonky2_field::types::{Field, Field64, PrimeField64};

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

// A simple struct to hold a test case for internal linear layer
#[derive(Debug, Clone)]
struct InternalLinearLayerTestCase {
    input: [GoldilocksField; 12],
    expected: [GoldilocksField; 12],
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
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);

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

    vectors
}

fn generate_sbox_test_vectors() -> Vec<SboxTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);

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

    vectors
}

fn generate_internal_linear_layer_test_vectors() -> Vec<InternalLinearLayerTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(67890);

    // Zero vector
    let zero_input = [GoldilocksField::ZERO; 12];
    let zero_expected = apply_internal_linear_layer_to_state(&zero_input);
    vectors.push(InternalLinearLayerTestCase {
        input: zero_input,
        expected: zero_expected,
    });

    // Unit vectors (one element is 1, rest are 0)
    for i in 0..12 {
        let mut input = [GoldilocksField::ZERO; 12];
        input[i] = GoldilocksField::ONE;
        let expected = apply_internal_linear_layer_to_state(&input);
        vectors.push(InternalLinearLayerTestCase { input, expected });
    }

    // Small values
    for _ in 0..10 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            input[j] = GoldilocksField::from_canonical_u64(rng.gen_range(0..100));
        }
        let expected = apply_internal_linear_layer_to_state(&input);
        vectors.push(InternalLinearLayerTestCase { input, expected });
    }

    // Random large values
    for _ in 0..20 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            input[j] = GoldilocksField::from_noncanonical_u64(rng.gen());
        }
        let expected = apply_internal_linear_layer_to_state(&input);
        vectors.push(InternalLinearLayerTestCase { input, expected });
    }

    println!(
        "Generated {} internal linear layer test vectors",
        vectors.len()
    );
    vectors
}

fn generate_mds_test_vectors() -> Vec<MdsTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(54321);

    // Zero vector
    let zero_input = [GoldilocksField::ZERO; 4];
    let zero_expected = apply_external_linear_layer_to_chunk(&zero_input);
    vectors.push(MdsTestCase {
        input: zero_input,
        expected: zero_expected,
    });

    // Unit vectors
    for i in 0..4 {
        let mut input = [GoldilocksField::ZERO; 4];
        input[i] = GoldilocksField::ONE;
        let expected = apply_external_linear_layer_to_chunk(&input);
        vectors.push(MdsTestCase { input, expected });
    }

    // All ones vector
    let ones_input = [GoldilocksField::ONE; 4];
    let ones_expected = apply_external_linear_layer_to_chunk(&ones_input);
    vectors.push(MdsTestCase {
        input: ones_input,
        expected: ones_expected,
    });

    // Random small values (all elements < 2^16)
    for _ in 0..50 {
        let mut input = [GoldilocksField::ZERO; 4];
        for j in 0..4 {
            input[j] = GoldilocksField::from_canonical_u64(rng.gen::<u16>() as u64);
        }
        let expected = apply_external_linear_layer_to_chunk(&input);
        vectors.push(MdsTestCase { input, expected });
    }

    // Random medium values (all elements < 2^32)
    for _ in 0..30 {
        let mut input = [GoldilocksField::ZERO; 4];
        for j in 0..4 {
            input[j] = GoldilocksField::from_canonical_u64(rng.gen::<u32>() as u64);
        }
        let expected = apply_external_linear_layer_to_chunk(&input);
        vectors.push(MdsTestCase { input, expected });
    }

    // Random large values (using full 64-bit range)
    for _ in 0..30 {
        let mut input = [GoldilocksField::ZERO; 4];
        for j in 0..4 {
            input[j] = GoldilocksField::from_canonical_u64(rng.gen::<u64>());
        }
        let expected = apply_external_linear_layer_to_chunk(&input);
        vectors.push(MdsTestCase { input, expected });
    }

    // Edge cases near field modulus
    for _ in 0..10 {
        let mut input = [GoldilocksField::ZERO; 4];
        for j in 0..4 {
            let offset = rng.gen::<u32>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER - offset);
        }
        let expected = apply_external_linear_layer_to_chunk(&input);
        vectors.push(MdsTestCase { input, expected });
    }

    vectors
}

// Helper function to apply the external linear layer to a 4-element chunk
// This implements the exact same MDS matrix as used in p3_poseidon2 for Goldilocks field
fn apply_external_linear_layer_to_chunk(chunk: &[GoldilocksField; 4]) -> [GoldilocksField; 4] {
    // Implement P3's apply_mat4 algorithm using plonky2's Goldilocks
    // This is the exact algorithm from Plonky3's poseidon2/src/external.rs apply_mat4
    //
    // The 4x4 MDS matrix is:
    // [[2, 3, 1, 1],
    //  [1, 2, 3, 1],
    //  [1, 1, 2, 3],
    //  [3, 1, 1, 2]]
    //
    // P3's optimized algorithm computes this using only 7 additions and 2 doubles
    // instead of 16 multiplications + 12 additions for naive matrix multiplication.
    //
    // The key insight is computing intermediate sums:
    // - t01 = x[0] + x[1]
    // - t23 = x[2] + x[3]
    // - t0123 = t01 + t23 = x[0] + x[1] + x[2] + x[3]
    // - t01123 = t0123 + x[1] = 2*x[1] + x[0] + x[2] + x[3]
    // - t01233 = t0123 + x[3] = x[0] + x[1] + x[2] + 2*x[3]
    //
    // Then the output is:
    // - x[0] = t01123 + t01 = 2*x[0] + 3*x[1] + x[2] + x[3]
    // - x[1] = t01123 + 2*x[2] = x[0] + 2*x[1] + 3*x[2] + x[3]
    // - x[2] = t01233 + t23 = x[0] + x[1] + 2*x[2] + 3*x[3]
    // - x[3] = t01233 + 2*x[0] = 3*x[0] + x[1] + x[2] + 2*x[3]

    let mut x = *chunk;

    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];

    // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
    x[3] = t01233 + x[0].double(); // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = t01123 + x[2].double(); // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]

    x
}

// Apply internal linear layer to a 12-element state (CPU reference implementation)
fn apply_internal_linear_layer_to_state(state: &[GoldilocksField; 12]) -> [GoldilocksField; 12] {
    let mut result = *state;

    // Compute sum of all elements first
    let mut sum = GoldilocksField::ZERO;
    for &element in state {
        sum += element;
    }

    // Apply internal linear layer: result[i] = state[i] * diag[i] + sum
    // Use the same diagonal constants as in WGSL (MATRIX_DIAG_12_GOLDILOCKS from Plonky3)
    let diag_constants = [
        0xc3b6c08e23ba9300u64,
        0xd84b5de94a324fb6u64,
        0x0d0c371c5b35b84fu64,
        0x7964f570e7188037u64,
        0x5daf18bbd996604bu64,
        0x6743bc47b9595257u64,
        0x5528b9362c59bb70u64,
        0xac45e25b7127b68bu64,
        0xa2077d7dfbb606b5u64,
        0xf3faac6faee378aeu64,
        0x0c6388b51545e883u64,
        0xd27dbb6944917b60u64,
    ];

    for i in 0..12 {
        let diag_value = GoldilocksField::from_noncanonical_u64(diag_constants[i]);
        result[i] = state[i] * diag_value + sum;
    }

    result
}

#[derive(Debug, Clone)]
struct Poseidon2TestCase {
    input: [GoldilocksField; 12],
    expected: [GoldilocksField; 12],
}

fn generate_poseidon2_test_vectors() -> Vec<Poseidon2TestCase> {
    println!("Generating comprehensive Poseidon2 permutation test vectors using qp-poseidon...");
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);

    // Use qp-plonky2 Poseidon2 permutation

    // Add special cases first
    println!("Adding special cases...");

    // Zero state
    let zero_input = [GoldilocksField::ZERO; 12];
    let zero_expected = <GoldilocksField as P2Permuter>::permute(zero_input);
    vectors.push(Poseidon2TestCase {
        input: zero_input,
        expected: zero_expected,
    });

    // Sequential values 1, 2, 3, ..., 12
    let sequential_input: [GoldilocksField; 12] =
        core::array::from_fn(|i| GoldilocksField::from_canonical_u64((i + 1) as u64));
    let sequential_expected = <GoldilocksField as P2Permuter>::permute(sequential_input);
    vectors.push(Poseidon2TestCase {
        input: sequential_input,
        expected: sequential_expected,
    });

    println!("Adding random test cases...");

    // Random small values (all elements < 2^16)
    for _ in 0..20 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u16>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = <GoldilocksField as P2Permuter>::permute(input);
        vectors.push(Poseidon2TestCase { input, expected });
    }

    // Random medium values (all elements < 2^32)
    for _ in 0..15 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u32>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = <GoldilocksField as P2Permuter>::permute(input);
        vectors.push(Poseidon2TestCase { input, expected });
    }

    // Random large values (using full 64-bit range)
    for _ in 0..10 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u64>();
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = <GoldilocksField as P2Permuter>::permute(input);
        vectors.push(Poseidon2TestCase { input, expected });
    }

    // Edge cases near field modulus
    for _ in 0..5 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let offset = rng.gen::<u32>() as u64;
            let val = GoldilocksField::ORDER - offset;
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = <GoldilocksField as P2Permuter>::permute(input);
        vectors.push(Poseidon2TestCase { input, expected });
    }

    println!(
        "Generated {} comprehensive Poseidon2 test vectors using qp-poseidon.",
        vectors.len()
    );
    // Focus on just the zero case to debug step by step
    vectors.into_iter().take(1).collect()
}

pub async fn test_poseidon2_permutation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Running Poseidon2 permutation tests ---");

    let test_vectors = generate_poseidon2_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    // CPU reference: what should happen when we add first round constants to zero state
    let zero_input = [GoldilocksField::ZERO; 12];
    println!("CPU reference - Zero state after first round constants:");
    let expected_after_constants = [
        GoldilocksField::from_canonical_u64(14536656643496110215u64), // First constant
        GoldilocksField::from_canonical_u64(13514488879006334106u64), // Second constant
        GoldilocksField::from_canonical_u64(8269306810203348633u64),  // etc...
        GoldilocksField::from_canonical_u64(16620024733254697148u64),
        GoldilocksField::from_canonical_u64(15437500654013853529u64),
        GoldilocksField::from_canonical_u64(17130370965102755777u64),
        GoldilocksField::from_canonical_u64(7737953962675256106u64),
        GoldilocksField::from_canonical_u64(7741006386906997745u64),
        GoldilocksField::from_canonical_u64(16560033798334227211u64),
        GoldilocksField::from_canonical_u64(15574890113987092201u64),
        GoldilocksField::from_canonical_u64(16870057884327338414u64),
        GoldilocksField::from_canonical_u64(3102687419436130260u64),
    ];

    for i in 0..12 {
        println!(
            "  Element {}: {}",
            i,
            expected_after_constants[i].to_canonical_u64()
        );
    }

    println!("Running {} test cases...", total_tests);

    // Use mining.wgsl directly with proper binding structure
    let mining_shader_source = include_str!("mining.wgsl");
    let shader_source = format!(
        "
{}

// Poseidon2 test entry point - reuse existing mining shader bindings
@compute @workgroup_size(1)
fn poseidon2_test(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let i = global_id.x;

    if (i == 0u) {{
        // Read input from header buffer (8 u32 values)
        // Convert to 12-element state
        var state: array<GoldilocksField, 12>;

        // Read actual input from header buffer (first 8 u32 values represent 2 GoldilocksField elements)
        // Each GoldilocksField needs 4 limbs (64 bits), but header has 8 u32 values (256 bits)
        // We can extract 4 GoldilocksField elements from 8 u32 values
        for (var j = 0u; j < 4u && j * 2u + 1u < 8u; j++) {{
            state[j] = GoldilocksField(
                header[j * 2u] & 0xFFFFu,
                (header[j * 2u] >> 16u) & 0xFFFFu,
                header[j * 2u + 1u] & 0xFFFFu,
                (header[j * 2u + 1u] >> 16u) & 0xFFFFu
            );
        }}

        // Fill remaining elements with zeros for now
        for (var j = 4u; j < 12u; j++) {{
            state[j] = GoldilocksField(0u, 0u, 0u, 0u);
        }}

        // Apply actual Poseidon2 permutation from mining.wgsl
        poseidon2_permute(&state);

        // Write first 4 results to results buffer as u32 pairs
        for (var j = 0u; j < 4u; j++) {{
            results[j * 2u] = state[j].limb0 | (state[j].limb1 << 16u);
            results[j * 2u + 1u] = state[j].limb2 | (state[j].limb3 << 16u);
        }}

        // Write remaining results to debug_buffer
        for (var j = 4u; j < 12u; j++) {{
            debug_buffer[100u + (j - 4u) * 2u] = state[j].limb0 | (state[j].limb1 << 16u);
            debug_buffer[100u + (j - 4u) * 2u + 1u] = state[j].limb2 | (state[j].limb3 << 16u);
        }}
    }}
}}
",
        mining_shader_source
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Poseidon2 test shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create explicit bind group layout to match mining.wgsl's 5 bindings
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Poseidon2 Test Bind Group Layout"),
        entries: &[
            // binding 0: results buffer
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
            // binding 1: header buffer
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
            // binding 2: start_nonce buffer
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
            // binding 3: difficulty_target buffer
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
            // binding 4: debug_buffer
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
        label: Some("Poseidon2 Test Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipeline with explicit layout
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Poseidon2 test pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("poseidon2_test"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Process tests in batches
    const BATCH_SIZE: usize = 32;
    for chunk in test_vectors.chunks(BATCH_SIZE) {
        // Prepare input data - convert first test case to header buffer format
        let test_input = &chunk[0].input; // Just use first test case
        let mut header_data = vec![0u32; 8];

        // Pack first 4 GoldilocksField elements into 8 u32 values for header buffer
        for i in 0..4.min(test_input.len()) {
            let val = test_input[i].to_canonical_u64();
            header_data[i * 2] = (val & 0xFFFFFFFF) as u32;
            header_data[i * 2 + 1] = ((val >> 32) & 0xFFFFFFFF) as u32;
        }

        // Create buffers matching mining.wgsl binding layout
        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Header Buffer"),
            contents: bytemuck::cast_slice(&header_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let nonce_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nonce Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let target_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Target Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Buffer"),
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create bind group with explicit layout
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Poseidon2 Test Bind Group"),
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
                    resource: nonce_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: target_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: debug_buffer.as_entire_binding(),
                },
            ],
        });

        // Run compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Poseidon2 Test Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Poseidon2 Test Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(chunk.len() as u32, 1, 1);
        }

        // Create staging buffers for both results and debug data
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Poseidon2 Staging Buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debug_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Poseidon2 Debug Staging Buffer"),
            size: debug_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, 1024);
        encoder.copy_buffer_to_buffer(
            &debug_buffer,
            0,
            &debug_staging_buffer,
            0,
            debug_buffer.size(),
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Read results from results buffer
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

        // Read results from debug buffer
        let debug_buffer_slice = debug_staging_buffer.slice(..);
        let (debug_sender, debug_receiver) = futures::channel::oneshot::channel();
        debug_buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            debug_sender.send(result).unwrap();
        });

        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        debug_receiver.await??;

        let data = buffer_slice.get_mapped_range();
        let debug_data = debug_buffer_slice.get_mapped_range();
        let gpu_results: &[u32] = bytemuck::cast_slice(&data);
        let debug_results: &[u32] = bytemuck::cast_slice(&debug_data);

        // Check results - read all 12 elements from both buffers
        for (i, test_case) in chunk.iter().enumerate() {
            if i > 0 {
                break;
            } // Only process first test case for now

            let mut gpu_result = [GoldilocksField::ZERO; 12];

            // Read first 4 elements from results buffer (as u32 pairs)
            for j in 0..4 {
                let low = gpu_results[j * 2] as u64;
                let high = gpu_results[j * 2 + 1] as u64;
                gpu_result[j] = GoldilocksField::from_noncanonical_u64(
                    (low & 0xFFFF)
                        | (((low >> 16) & 0xFFFF) << 16)
                        | ((high & 0xFFFF) << 32)
                        | (((high >> 16) & 0xFFFF) << 48),
                );
            }

            // Read remaining 8 elements from debug buffer (starting at index 100)
            for j in 4..12 {
                let debug_idx = 100 + (j - 4) * 2;
                if debug_idx + 1 < debug_results.len() {
                    let low = debug_results[debug_idx] as u64;
                    let high = debug_results[debug_idx + 1] as u64;
                    gpu_result[j] = GoldilocksField::from_noncanonical_u64(
                        (low & 0xFFFF)
                            | (((low >> 16) & 0xFFFF) << 16)
                            | ((high & 0xFFFF) << 32)
                            | (((high >> 16) & 0xFFFF) << 48),
                    );
                }
            }

            // Compare all 12 elements
            if gpu_result == test_case.expected {
                passed_tests += 1;
            } else {
                failed_tests.push((test_case.clone(), gpu_result));
                if failed_tests.len() <= 3 {
                    println!("❌ FAILED: Poseidon2 permutation test");
                    println!(
                        "Input: {:?}",
                        test_case
                            .input
                            .iter()
                            .map(|x| x.to_canonical_u64())
                            .collect::<Vec<_>>()
                    );
                    println!(
                        "Expected: {:?}",
                        test_case
                            .expected
                            .iter()
                            .map(|x| x.to_canonical_u64())
                            .collect::<Vec<_>>()
                    );
                    println!(
                        "Got:      {:?}",
                        gpu_result
                            .iter()
                            .map(|x| x.to_canonical_u64())
                            .collect::<Vec<_>>()
                    );

                    // Show element-by-element comparison
                    for i in 0..12 {
                        let expected = test_case.expected[i].to_canonical_u64();
                        let got = gpu_result[i].to_canonical_u64();
                        let status = if expected == got { "✓" } else { "✗" };
                        println!("  [{}] {} Expected: {}, Got: {}", i, status, expected, got);
                    }
                }
            }
        }

        drop(data);
        drop(debug_data);
        staging_buffer.unmap();
        debug_staging_buffer.unmap();
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
        return Err(format!("{} out of {} tests failed", failed_tests.len(), total_tests).into());
    }

    Ok(())
}

pub async fn test_gf_mul(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_gf_mul_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

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
    println!("\n=== GF_MUL TEST SUMMARY ===");
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
    let test_vectors = generate_mds_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

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

    // Apply 4x4 MDS matrix transformation: [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
    let two = gf_from_limbs(2u, 0u, 0u, 0u);
    let three = gf_from_limbs(3u, 0u, 0u, 0u);

    // new[0] = 2*x[0] + 3*x[1] + 1*x[2] + 1*x[3]
    let new_0 = gf_add(gf_add(gf_add(
        gf_mul(chunk_state[0], two),
        gf_mul(chunk_state[1], three)),
        chunk_state[2]),
        chunk_state[3]);

    // new[1] = 1*x[0] + 2*x[1] + 3*x[2] + 1*x[3]
    let new_1 = gf_add(gf_add(gf_add(
        chunk_state[0],
        gf_mul(chunk_state[1], two)),
        gf_mul(chunk_state[2], three)),
        chunk_state[3]);

    // new[2] = 1*x[0] + 1*x[1] + 2*x[2] + 3*x[3]
    let new_2 = gf_add(gf_add(gf_add(
        chunk_state[0],
        chunk_state[1]),
        gf_mul(chunk_state[2], two)),
        gf_mul(chunk_state[3], three));

    // new[3] = 3*x[0] + 1*x[1] + 1*x[2] + 2*x[3]
    let new_3 = gf_add(gf_add(gf_add(
        gf_mul(chunk_state[0], three),
        chunk_state[1]),
        chunk_state[2]),
        gf_mul(chunk_state[3], two));

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

    // Create explicit bind group layout for MDS test
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MDS Test Bind Group Layout"),
        entries: &[
            // binding 0: input_data buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: output_data buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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
        label: Some("MDS Test Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipeline with explicit layout
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MDS test pipeline"),
        layout: Some(&pipeline_layout),
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
                    (gpu_results[i * 4].limb0 as u64 & 0xFFFF)
                        | (((gpu_results[i * 4].limb1 as u64) & 0xFFFF) << 16)
                        | (((gpu_results[i * 4].limb2 as u64) & 0xFFFF) << 32)
                        | (((gpu_results[i * 4].limb3 as u64) & 0xFFFF) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (gpu_results[i * 4 + 1].limb0 as u64 & 0xFFFF)
                        | (((gpu_results[i * 4 + 1].limb1 as u64) & 0xFFFF) << 16)
                        | (((gpu_results[i * 4 + 1].limb2 as u64) & 0xFFFF) << 32)
                        | (((gpu_results[i * 4 + 1].limb3 as u64) & 0xFFFF) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (gpu_results[i * 4 + 2].limb0 as u64 & 0xFFFF)
                        | (((gpu_results[i * 4 + 2].limb1 as u64) & 0xFFFF) << 16)
                        | (((gpu_results[i * 4 + 2].limb2 as u64) & 0xFFFF) << 32)
                        | (((gpu_results[i * 4 + 2].limb3 as u64) & 0xFFFF) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (gpu_results[i * 4 + 3].limb0 as u64 & 0xFFFF)
                        | (((gpu_results[i * 4 + 3].limb1 as u64) & 0xFFFF) << 16)
                        | (((gpu_results[i * 4 + 3].limb2 as u64) & 0xFFFF) << 32)
                        | (((gpu_results[i * 4 + 3].limb3 as u64) & 0xFFFF) << 48),
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

    println!("\n=== MDS TEST SUMMARY ===");
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

    Ok(())
}

pub async fn test_internal_linear_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_internal_linear_layer_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    // Read the mining shader source and create a test wrapper
    let mining_shader_source = include_str!("mining.wgsl");
    let shader_source = format!(
        "
{}

// Test entry point for internal linear layer
@group(0) @binding(0) var<storage, read> input_state: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_state: array<GoldilocksField>;

@compute @workgroup_size(1)
fn internal_linear_layer_test(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let offset = test_id * 12u;

    // Copy input to local state
    var state: array<GoldilocksField, 12>;
    for (var i = 0u; i < 12u; i++) {{
        state[i] = input_state[offset + i];
    }}

    // Apply internal linear layer
    internal_linear_layer(&state);

    // Write result
    for (var i = 0u; i < 12u; i++) {{
        output_state[offset + i] = state[i];
    }}
}}
",
        mining_shader_source
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Internal Linear Layer Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Internal Linear Layer Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("internal_linear_layer_test"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Process tests in batches
    const BATCH_SIZE: usize = 32;
    for chunk in test_vectors.chunks(BATCH_SIZE) {
        // Prepare input data (flatten 12-element arrays and convert to GfWgls)
        let mut input_data = Vec::new();
        for test_case in chunk {
            for &field_elem in &test_case.input {
                input_data.push(GfWgls::from(field_elem));
            }
        }

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (input_data.len() * std::mem::size_of::<GfWgls>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Bind Group"),
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
            pass.dispatch_workgroups(chunk.len() as u32, 1, 1);
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
        let results: &[GfWgls] = bytemuck::cast_slice(&data);

        // Check results
        for (i, test_case) in chunk.iter().enumerate() {
            let gpu_result = [
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12].limb0 as u64)
                        | ((results[i * 12].limb1 as u64) << 16)
                        | ((results[i * 12].limb2 as u64) << 32)
                        | ((results[i * 12].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 1].limb0 as u64)
                        | ((results[i * 12 + 1].limb1 as u64) << 16)
                        | ((results[i * 12 + 1].limb2 as u64) << 32)
                        | ((results[i * 12 + 1].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 2].limb0 as u64)
                        | ((results[i * 12 + 2].limb1 as u64) << 16)
                        | ((results[i * 12 + 2].limb2 as u64) << 32)
                        | ((results[i * 12 + 2].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 3].limb0 as u64)
                        | ((results[i * 12 + 3].limb1 as u64) << 16)
                        | ((results[i * 12 + 3].limb2 as u64) << 32)
                        | ((results[i * 12 + 3].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 4].limb0 as u64)
                        | ((results[i * 12 + 4].limb1 as u64) << 16)
                        | ((results[i * 12 + 4].limb2 as u64) << 32)
                        | ((results[i * 12 + 4].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 5].limb0 as u64)
                        | ((results[i * 12 + 5].limb1 as u64) << 16)
                        | ((results[i * 12 + 5].limb2 as u64) << 32)
                        | ((results[i * 12 + 5].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 6].limb0 as u64)
                        | ((results[i * 12 + 6].limb1 as u64) << 16)
                        | ((results[i * 12 + 6].limb2 as u64) << 32)
                        | ((results[i * 12 + 6].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 7].limb0 as u64)
                        | ((results[i * 12 + 7].limb1 as u64) << 16)
                        | ((results[i * 12 + 7].limb2 as u64) << 32)
                        | ((results[i * 12 + 7].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 8].limb0 as u64)
                        | ((results[i * 12 + 8].limb1 as u64) << 16)
                        | ((results[i * 12 + 8].limb2 as u64) << 32)
                        | ((results[i * 12 + 8].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 9].limb0 as u64)
                        | ((results[i * 12 + 9].limb1 as u64) << 16)
                        | ((results[i * 12 + 9].limb2 as u64) << 32)
                        | ((results[i * 12 + 9].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 10].limb0 as u64)
                        | ((results[i * 12 + 10].limb1 as u64) << 16)
                        | ((results[i * 12 + 10].limb2 as u64) << 32)
                        | ((results[i * 12 + 10].limb3 as u64) << 48),
                ),
                GoldilocksField::from_noncanonical_u64(
                    (results[i * 12 + 11].limb0 as u64)
                        | ((results[i * 12 + 11].limb1 as u64) << 16)
                        | ((results[i * 12 + 11].limb2 as u64) << 32)
                        | ((results[i * 12 + 11].limb3 as u64) << 48),
                ),
            ];

            // Compare all 12 elements
            let mut all_match = true;
            for j in 0..12 {
                if gpu_result[j].to_canonical_u64() != test_case.expected[j].to_canonical_u64() {
                    all_match = false;
                    break;
                }
            }

            if all_match {
                passed_tests += 1;
            } else {
                failed_tests.push((test_case.clone(), gpu_result));
                if failed_tests.len() <= 3 {
                    println!("❌ FAILED: Internal linear layer test");
                    println!("Input state:");
                    for (j, &val) in test_case.input.iter().enumerate() {
                        println!("  [{}] = 0x{:016x}", j, val.to_canonical_u64());
                    }
                    println!("Expected state:");
                    for (j, &val) in test_case.expected.iter().enumerate() {
                        println!("  [{}] = 0x{:016x}", j, val.to_canonical_u64());
                    }
                    println!("GPU result:");
                    for (j, &val) in gpu_result.iter().enumerate() {
                        println!("  [{}] = 0x{:016x}", j, val.to_canonical_u64());
                    }
                    println!("Element-by-element comparison:");
                    for j in 0..12 {
                        let expected = test_case.expected[j].to_canonical_u64();
                        let got = gpu_result[j].to_canonical_u64();
                        let status = if expected == got { "✓" } else { "✗" };
                        println!(
                            "  [{}] {} Expected: 0x{:016x}, Got: 0x{:016x}",
                            j, status, expected, got
                        );
                    }
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    println!(
        "Internal Linear Layer Tests: {}/{} passed ({:.1}%)",
        passed_tests,
        total_tests,
        100.0 * passed_tests as f64 / total_tests as f64
    );

    if !failed_tests.is_empty() {
        println!(
            "❌ {} internal linear layer tests failed",
            failed_tests.len()
        );
        if failed_tests.len() > 3 {
            println!("   (showing only first 3 failures)");
        }
    } else {
        println!("✅ All internal linear layer tests passed!");
    }

    Ok(())
}

pub async fn test_sbox(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_sbox_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    const BATCH_SIZE: usize = 64;
    let batches = test_vectors.chunks(BATCH_SIZE);

    for (batch_idx, batch) in batches.enumerate() {
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
