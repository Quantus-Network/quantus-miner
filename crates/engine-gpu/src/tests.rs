use plonky2::hash::poseidon2::P2Permuter;
use qp_plonky2_field::goldilocks_field::GoldilocksField;
use qp_plonky2_field::types::{Field, Field64, PrimeField64};

use qp_poseidon_core::serialization::{p2_backend::GF as P2Goldilocks, GoldiCompat};
use rand::{Rng, SeedableRng};
use wgpu::util::DeviceExt;

// A simple struct to hold a test case for gf_mul
#[derive(Debug, Clone)]
struct GfMulTestCase {
    a: GoldilocksField,
    b: GoldilocksField,
    expected: GoldilocksField,
}

#[derive(Debug, Clone)]
struct GfFromConstTestCase {
    a: u32, // low 32 bits
    b: u32, // high 32 bits
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

// A simple struct to hold a test case for external linear layer
#[derive(Debug, Clone)]
struct ExternalLinearLayerTestCase {
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

fn generate_gf_from_const_test_vectors() -> Vec<GfFromConstTestCase> {
    let mut vectors = Vec::new();

    // Test zero
    vectors.push(GfFromConstTestCase {
        a: 0,
        b: 0,
        expected: GoldilocksField::from_canonical_u64(0),
    });

    // Test one
    vectors.push(GfFromConstTestCase {
        a: 1,
        b: 0,
        expected: GoldilocksField::from_canonical_u64(1),
    });

    // Test small values
    for val in [2, 7, 42, 255, 1024, 65535] {
        vectors.push(GfFromConstTestCase {
            a: val,
            b: 0,
            expected: GoldilocksField::from_canonical_u64(val as u64),
        });
    }

    // Test values that use high 32 bits
    for high in [1, 2, 1000, 65535] {
        let val = (high as u64) << 32;
        vectors.push(GfFromConstTestCase {
            a: 0,
            b: high,
            expected: GoldilocksField::from_canonical_u64(val),
        });
    }

    // Test combined low and high bits
    for (low, high) in [(123, 456), (65535, 32767), (4294967295, 65535)] {
        let val = (high as u64) << 32 | (low as u64);
        vectors.push(GfFromConstTestCase {
            a: low,
            b: high,
            expected: GoldilocksField::from_canonical_u64(val),
        });
    }

    // Test actual constants from INITIAL_EXTERNAL_CONSTANTS
    let actual_constants = [
        (2539329031u32, 3221415792u32),
        (1633409695u32, 3565876427u32),
        (2143952946u32, 1246797945u32),
    ];

    for (low, high) in actual_constants {
        let val = (high as u64) << 32 | (low as u64);
        vectors.push(GfFromConstTestCase {
            a: low,
            b: high,
            expected: GoldilocksField::from_canonical_u64(val),
        });
    }

    vectors
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

fn generate_external_linear_layer_test_vectors() -> Vec<ExternalLinearLayerTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(67890);

    // Zero state
    let zero_input = [GoldilocksField::ZERO; 12];
    let zero_expected = apply_full_external_linear_layer_to_state(&zero_input);
    vectors.push(ExternalLinearLayerTestCase {
        input: zero_input,
        expected: zero_expected,
    });

    // Sequential values 1, 2, 3, ..., 12
    let sequential_input: [GoldilocksField; 12] =
        core::array::from_fn(|i| GoldilocksField::from_canonical_u64((i + 1) as u64));
    let sequential_expected = apply_full_external_linear_layer_to_state(&sequential_input);
    vectors.push(ExternalLinearLayerTestCase {
        input: sequential_input,
        expected: sequential_expected,
    });

    // Unit vectors (one element is 1, rest are 0)
    for i in 0..12 {
        let mut input = [GoldilocksField::ZERO; 12];
        input[i] = GoldilocksField::ONE;
        let expected = apply_full_external_linear_layer_to_state(&input);
        vectors.push(ExternalLinearLayerTestCase { input, expected });
    }

    // Random small values (all elements < 2^16)
    for _ in 0..15 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u16>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = apply_full_external_linear_layer_to_state(&input);
        vectors.push(ExternalLinearLayerTestCase { input, expected });
    }

    // Random medium values (all elements < 2^32)
    for _ in 0..10 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u32>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = apply_full_external_linear_layer_to_state(&input);
        vectors.push(ExternalLinearLayerTestCase { input, expected });
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
// Apply full external linear layer to a 12-element state (CPU reference implementation)
fn apply_full_external_linear_layer_to_state(
    state: &[GoldilocksField; 12],
) -> [GoldilocksField; 12] {
    let mut result = *state;

    // First apply the 4x4 MDS matrix to each consecutive 4 elements
    for chunk_idx in 0..3 {
        let offset = chunk_idx * 4;
        let chunk = [
            result[offset],
            result[offset + 1],
            result[offset + 2],
            result[offset + 3],
        ];
        let transformed_chunk = apply_external_linear_layer_to_chunk(&chunk);

        result[offset] = transformed_chunk[0];
        result[offset + 1] = transformed_chunk[1];
        result[offset + 2] = transformed_chunk[2];
        result[offset + 3] = transformed_chunk[3];
    }

    // Now apply the circulant matrix part
    // Precompute the four sums of every four elements
    let mut sums = [GoldilocksField::ZERO; 4];
    for k in 0..4 {
        for j in (k..12).step_by(4) {
            sums[k] += result[j];
        }
    }

    // Add the appropriate sum to each element
    for i in 0..12 {
        result[i] += sums[i % 4];
    }

    result
}

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

#[derive(Debug, Clone)]
struct Poseidon2SqueezeTwiceTestCase {
    input_bytes: Vec<u8>,    // Should be exactly 96 bytes for our GPU implementation
    expected_hash: [u8; 32], // First 32 bytes of the 64-byte squeeze_twice output
}

#[derive(Debug, Clone)]
struct BytesToFieldTestCase {
    input_bytes: [u8; 96],
    expected_felts: Vec<GoldilocksField>,
}

#[derive(Debug, Clone)]
struct FieldToBytesTestCase {
    input_felts: [GoldilocksField; 4],
    expected_bytes: [u8; 32],
}

#[derive(Debug, Clone)]
struct InternalRoundsTestCase {
    input: [GoldilocksField; 12],
    expected: [GoldilocksField; 12],
}

#[derive(Debug, Clone)]
struct DoubleHashTestCase {
    input_96_bytes: [u8; 96], // Header (32) + Nonce (64) bytes
    expected_hash: [u8; 64],  // Full 64-byte double hash output
    description: String,
}

fn generate_poseidon2_test_vectors() -> Vec<Poseidon2TestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);

    // Zero state
    let zero_input = [GoldilocksField::ZERO; 12];
    let zero_expected = <GoldilocksField as P2Permuter>::permute(zero_input);
    vectors.push(Poseidon2TestCase {
        input: zero_input,
        expected: zero_expected,
    });

    // One state (first element = 1, rest = 0)
    let mut one_input = [GoldilocksField::ZERO; 12];
    one_input[0] = GoldilocksField::ONE;
    let one_expected = <GoldilocksField as P2Permuter>::permute(one_input);
    vectors.push(Poseidon2TestCase {
        input: one_input,
        expected: one_expected,
    });

    // Small random values (0-255)
    for _ in 0..30 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u8>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = <GoldilocksField as P2Permuter>::permute(input);
        vectors.push(Poseidon2TestCase { input, expected });
    }

    // Medium random values (0-65535)
    for _ in 0..20 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u16>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = <GoldilocksField as P2Permuter>::permute(input);
        vectors.push(Poseidon2TestCase { input, expected });
    }

    // Large random values (full u32 range)
    for _ in 0..15 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u32>() as u64;
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = <GoldilocksField as P2Permuter>::permute(input);
        vectors.push(Poseidon2TestCase { input, expected });
    }

    // Very large random values (full field range)
    for _ in 0..10 {
        let mut input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = rng.gen::<u64>();
            input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let expected = <GoldilocksField as P2Permuter>::permute(input);
        vectors.push(Poseidon2TestCase { input, expected });
    }

    vectors
}

fn generate_field_to_bytes_test_vectors() -> Vec<FieldToBytesTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x13579BDF);

    // Test 1: All zeros
    let zero_input = [GoldilocksField::ZERO; 4];
    let mut zero_expected = [0u8; 32];
    for (i, &felt) in zero_input.iter().enumerate() {
        let u64_val = felt.to_canonical_u64();
        zero_expected[i * 8..(i + 1) * 8].copy_from_slice(&u64_val.to_le_bytes());
    }
    vectors.push(FieldToBytesTestCase {
        input_felts: zero_input,
        expected_bytes: zero_expected,
    });

    // Test 2: All ones
    let ones_input = [GoldilocksField::ONE; 4];
    let mut ones_expected = [0u8; 32];
    for (i, &felt) in ones_input.iter().enumerate() {
        let u64_val = felt.to_canonical_u64();
        ones_expected[i * 8..(i + 1) * 8].copy_from_slice(&u64_val.to_le_bytes());
    }
    vectors.push(FieldToBytesTestCase {
        input_felts: ones_input,
        expected_bytes: ones_expected,
    });

    // Test 3: Sequential values
    let seq_input = [
        GoldilocksField::from_canonical_u64(0x0123456789ABCDEF),
        GoldilocksField::from_canonical_u64(0xFEDCBA9876543210),
        GoldilocksField::from_canonical_u64(0x1111111111111111),
        GoldilocksField::from_canonical_u64(0x2222222222222222),
    ];
    let mut seq_expected = [0u8; 32];
    for (i, &felt) in seq_input.iter().enumerate() {
        let u64_val = felt.to_canonical_u64();
        seq_expected[i * 8..(i + 1) * 8].copy_from_slice(&u64_val.to_le_bytes());
    }
    vectors.push(FieldToBytesTestCase {
        input_felts: seq_input,
        expected_bytes: seq_expected,
    });

    // Test 4: Random values
    for _ in 0..10 {
        let input = [
            GoldilocksField::from_canonical_u64(rng.gen::<u64>() % GoldilocksField::ORDER),
            GoldilocksField::from_canonical_u64(rng.gen::<u64>() % GoldilocksField::ORDER),
            GoldilocksField::from_canonical_u64(rng.gen::<u64>() % GoldilocksField::ORDER),
            GoldilocksField::from_canonical_u64(rng.gen::<u64>() % GoldilocksField::ORDER),
        ];
        let mut expected = [0u8; 32];
        for (i, &felt) in input.iter().enumerate() {
            let u64_val = felt.to_canonical_u64();
            expected[i * 8..(i + 1) * 8].copy_from_slice(&u64_val.to_le_bytes());
        }
        vectors.push(FieldToBytesTestCase {
            input_felts: input,
            expected_bytes: expected,
        });
    }

    vectors
}

fn generate_poseidon2_squeeze_twice_test_vectors() -> Vec<Poseidon2SqueezeTwiceTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42424242);

    // Our GPU function expects exactly 96 bytes and produces 32 bytes (first half of 64-byte output)
    // Test with 96-byte inputs using qp_poseidon_core::hash_squeeze_twice for reference

    // Zero input (96 bytes of zeros)
    let zero_input = vec![0u8; 96];
    let zero_expected_64 = qp_poseidon_core::hash_squeeze_twice(&zero_input);
    let mut zero_expected_32 = [0u8; 32];
    zero_expected_32.copy_from_slice(&zero_expected_64[0..32]); // Take first 32 bytes
    vectors.push(Poseidon2SqueezeTwiceTestCase {
        input_bytes: zero_input,
        expected_hash: zero_expected_32,
    });

    // All 0x01 input
    let ones_input = vec![1u8; 96];
    let ones_expected_64 = qp_poseidon_core::hash_squeeze_twice(&ones_input);
    let mut ones_expected_32 = [0u8; 32];
    ones_expected_32.copy_from_slice(&ones_expected_64[0..32]);
    vectors.push(Poseidon2SqueezeTwiceTestCase {
        input_bytes: ones_input,
        expected_hash: ones_expected_32,
    });

    // All 0xFF input
    let ff_input = vec![0xFFu8; 96];
    let ff_expected_64 = qp_poseidon_core::hash_squeeze_twice(&ff_input);
    let mut ff_expected_32 = [0u8; 32];
    ff_expected_32.copy_from_slice(&ff_expected_64[0..32]);
    vectors.push(Poseidon2SqueezeTwiceTestCase {
        input_bytes: ff_input,
        expected_hash: ff_expected_32,
    });

    // Sequential pattern input
    let seq_input: Vec<u8> = (0u8..96).collect();
    let seq_expected_64 = qp_poseidon_core::hash_squeeze_twice(&seq_input);
    let mut seq_expected_32 = [0u8; 32];
    seq_expected_32.copy_from_slice(&seq_expected_64[0..32]);
    vectors.push(Poseidon2SqueezeTwiceTestCase {
        input_bytes: seq_input,
        expected_hash: seq_expected_32,
    });

    // Random inputs (96 bytes each)
    for _ in 0..20 {
        let mut input = vec![0u8; 96];
        rng.fill(&mut input[..]);
        let expected_64 = qp_poseidon_core::hash_squeeze_twice(&input);
        let mut expected_32 = [0u8; 32];
        expected_32.copy_from_slice(&expected_64[0..32]);
        vectors.push(Poseidon2SqueezeTwiceTestCase {
            input_bytes: input,
            expected_hash: expected_32,
        });
    }

    vectors
}

#[derive(Debug, Clone)]
struct Poseidon2StandardHashTestCase {
    input_bytes: Vec<u8>,    // Variable length input
    expected_hash: [u8; 32], // Standard 32-byte output
}

fn generate_poseidon2_standard_hash_test_vectors() -> Vec<Poseidon2StandardHashTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xFEEDBEEF);

    // Test 1: Empty input
    let empty_input = vec![];
    let empty_expected = qp_poseidon_core::hash_variable_length_bytes(&empty_input);
    vectors.push(Poseidon2StandardHashTestCase {
        input_bytes: empty_input,
        expected_hash: empty_expected,
    });

    // Test 2: Single byte inputs
    for byte_val in [0u8, 1u8, 255u8, 42u8] {
        let input = vec![byte_val];
        let expected = qp_poseidon_core::hash_variable_length_bytes(&input);
        vectors.push(Poseidon2StandardHashTestCase {
            input_bytes: input,
            expected_hash: expected,
        });
    }

    // Test 3: Small inputs (1-32 bytes)
    for len in [1, 4, 8, 16, 32] {
        let mut input = vec![0u8; len];
        rng.fill(&mut input[..]);
        let expected = qp_poseidon_core::hash_variable_length_bytes(&input);
        vectors.push(Poseidon2StandardHashTestCase {
            input_bytes: input,
            expected_hash: expected,
        });
    }

    // Test 4: String inputs
    let strings = ["", "a", "hello", "test"];
    for s in strings {
        let input = s.as_bytes().to_vec();
        let expected = qp_poseidon_core::hash_variable_length_bytes(&input);
        vectors.push(Poseidon2StandardHashTestCase {
            input_bytes: input,
            expected_hash: expected,
        });
    }

    vectors
}

fn generate_bytes_to_field_test_vectors() -> Vec<BytesToFieldTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x98765432);

    // Test 1: All zeros
    let zero_input = [0u8; 96];
    let zero_expected =
        qp_poseidon_core::serialization::injective_bytes_to_felts::<P2Goldilocks>(&zero_input)
            .into_iter()
            .map(|g| GoldilocksField::from_noncanonical_u64(g.to_u64()))
            .collect();
    vectors.push(BytesToFieldTestCase {
        input_bytes: zero_input,
        expected_felts: zero_expected,
    });

    // Test 2: All 0xFF
    let ff_input = [0xFFu8; 96];
    let ff_expected =
        qp_poseidon_core::serialization::injective_bytes_to_felts::<P2Goldilocks>(&ff_input)
            .into_iter()
            .map(|g| GoldilocksField::from_noncanonical_u64(g.to_u64()))
            .collect();
    vectors.push(BytesToFieldTestCase {
        input_bytes: ff_input,
        expected_felts: ff_expected,
    });

    // Test 3: Sequential bytes
    let mut seq_input = [0u8; 96];
    for (i, byte) in seq_input.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }
    let seq_expected =
        qp_poseidon_core::serialization::injective_bytes_to_felts::<P2Goldilocks>(&seq_input)
            .into_iter()
            .map(|g| GoldilocksField::from_noncanonical_u64(g.to_u64()))
            .collect();
    vectors.push(BytesToFieldTestCase {
        input_bytes: seq_input,
        expected_felts: seq_expected,
    });

    // Test 4: Random inputs
    for _ in 0..10 {
        let mut input = [0u8; 96];
        rng.fill(&mut input[..]);
        let expected =
            qp_poseidon_core::serialization::injective_bytes_to_felts::<P2Goldilocks>(&input)
                .into_iter()
                .map(|g| GoldilocksField::from_noncanonical_u64(g.to_u64()))
                .collect();
        vectors.push(BytesToFieldTestCase {
            input_bytes: input,
            expected_felts: expected,
        });
    }

    vectors
}

// Standardized test reporting functions
struct TestResults {
    name: String,
    passed: usize,
    failed: Vec<usize>,
    total: usize,
}

impl TestResults {
    fn new(name: String, total: usize) -> Self {
        Self {
            name,
            passed: 0,
            failed: Vec::new(),
            total,
        }
    }

    fn add_pass(&mut self) {
        self.passed += 1;
    }

    fn add_failure(&mut self, test_index: usize) {
        self.failed.push(test_index);
    }

    fn print_summary(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n=== {} TEST SUMMARY ===", self.name.to_uppercase());
        println!("Total tests: {}", self.total);
        println!(
            "Passed: {} ({:.1}%)",
            self.passed,
            self.passed as f32 / self.total as f32 * 100.0
        );

        if self.failed.is_empty() {
            println!("🎉 All {} tests PASSED!", self.name.to_lowercase());
            Ok(())
        } else {
            println!(
                "Failed: {} ({:.1}%)",
                self.failed.len(),
                self.failed.len() as f32 / self.total as f32 * 100.0
            );
            if self.failed.len() > 10 {
                println!("  (showing first 10 failures above)");
            }
            Err(format!(
                "{} out of {} {} tests failed",
                self.failed.len(),
                self.total,
                self.name.to_lowercase()
            )
            .into())
        }
    }
}

fn print_test_failure<T: std::fmt::Display>(
    test_name: &str,
    test_index: usize,
    expected: &[T],
    actual: &[T],
    max_elements: usize,
) {
    println!("\n{} test case {} FAILED:", test_name, test_index);
    for j in 0..max_elements.min(expected.len()).min(actual.len()) {
        println!("  Expected [{}]: {}", j, expected[j]);
        println!("  GPU      [{}]: {}", j, actual[j]);
    }
}

fn print_test_failure_hex(
    test_name: &str,
    test_index: usize,
    expected: &[GoldilocksField],
    actual: &[GoldilocksField],
    max_elements: usize,
) {
    println!("\n{} test case {} FAILED:", test_name, test_index);
    for j in 0..max_elements.min(expected.len()).min(actual.len()) {
        println!(
            "  Expected [{}]: 0x{:016x}",
            j,
            expected[j].to_canonical_u64()
        );
        println!(
            "  GPU      [{}]: 0x{:016x}",
            j,
            actual[j].to_canonical_u64()
        );
    }
}

fn generate_double_hash_test_vectors() -> Vec<DoubleHashTestCase> {
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xDEADBEEF);

    // Test 1-250: Random header and nonce combinations (25 tests)
    for i in 0..250 {
        let mut input = [0u8; 96];

        rng.fill(&mut input[..]);

        let expected = {
            let first_hash = qp_poseidon_core::hash_squeeze_twice(&input);
            qp_poseidon_core::hash_squeeze_twice(&first_hash)
        };

        vectors.push(DoubleHashTestCase {
            input_96_bytes: input,
            expected_hash: expected,
            description: format!("random_case_{}", i + 1),
        });
    }

    vectors
}

fn generate_true_internal_only_test_vectors() -> Vec<InternalRoundsTestCase> {
    // This function generates test vectors for INTERNAL ROUNDS ONLY
    // Input: state after initial external rounds
    // Expected: state after internal rounds (before terminal external)
    let mut vectors = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(54321);
    use qp_poseidon_constants::{
        POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW, POSEIDON2_INTERNAL_CONSTANTS_RAW,
    };

    // Helper to apply initial external rounds
    let apply_initial_external = |mut state: [GoldilocksField; 12]| -> [GoldilocksField; 12] {
        for round in 0..4 {
            // Add constants
            for i in 0..12 {
                state[i] += GoldilocksField::from_canonical_u64(
                    POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW[round][i],
                );
            }
            // S-box all elements
            for i in 0..12 {
                state[i] = state[i].exp_u64(7);
            }
            // External linear layer
            state = apply_full_external_linear_layer_to_state(&state);
        }
        state
    };

    // Helper to apply internal rounds only
    let apply_internal_only = |mut state: [GoldilocksField; 12]| -> [GoldilocksField; 12] {
        for round in 0..22 {
            state[0] +=
                GoldilocksField::from_canonical_u64(POSEIDON2_INTERNAL_CONSTANTS_RAW[round]);
            state[0] = state[0].exp_u64(7);
            state = apply_internal_linear_layer_to_state(&state);
        }
        state
    };

    // Zero state
    let zero_input = [GoldilocksField::ZERO; 12];
    let after_initial_external = apply_initial_external(zero_input);
    let after_internal = apply_internal_only(after_initial_external);
    vectors.push(InternalRoundsTestCase {
        input: after_initial_external,
        expected: after_internal,
    });

    // Unit vectors (one element = 1, rest = 0)
    for i in 0..12 {
        let mut unit_input = [GoldilocksField::ZERO; 12];
        unit_input[i] = GoldilocksField::ONE;
        let after_initial_external = apply_initial_external(unit_input);
        let after_internal = apply_internal_only(after_initial_external);
        vectors.push(InternalRoundsTestCase {
            input: after_initial_external,
            expected: after_internal,
        });
    }

    // All ones
    let ones_input = [GoldilocksField::ONE; 12];
    let after_initial_external = apply_initial_external(ones_input);
    let after_internal = apply_internal_only(after_initial_external);
    vectors.push(InternalRoundsTestCase {
        input: after_initial_external,
        expected: after_internal,
    });

    // Sequential [1,2,3,...,12]
    let sequential_input =
        core::array::from_fn(|i| GoldilocksField::from_canonical_u64((i + 1) as u64));
    let after_initial_external = apply_initial_external(sequential_input);
    let after_internal = apply_internal_only(after_initial_external);
    vectors.push(InternalRoundsTestCase {
        input: after_initial_external,
        expected: after_internal,
    });

    // Random small values (all elements < 2^8)
    for _ in 0..15 {
        let mut random_input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = (rng.gen::<u8>()) as u64;
            random_input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let after_initial_external = apply_initial_external(random_input);
        let after_internal = apply_internal_only(after_initial_external);
        vectors.push(InternalRoundsTestCase {
            input: after_initial_external,
            expected: after_internal,
        });
    }

    // Random medium values (all elements < 2^16)
    for _ in 0..10 {
        let mut random_input = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = (rng.gen::<u16>()) as u64;
            random_input[j] = GoldilocksField::from_canonical_u64(val);
        }
        let after_initial_external = apply_initial_external(random_input);
        let after_internal = apply_internal_only(after_initial_external);
        vectors.push(InternalRoundsTestCase {
            input: after_initial_external,
            expected: after_internal,
        });
    }

    vectors
}

pub async fn test_poseidon2_permutation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_poseidon2_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let mut failure_details = Vec::new();

    // Use the same pattern as other successful tests - include mining.wgsl and define our own bindings
    let shader_source = format!(
        "
{}

// Simple test entry point for Poseidon2 permutation
@group(0) @binding(0) var<storage, read> input_state: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_state: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let offset = test_id * 12u;

    // Copy input to local state
    var state: array<GoldilocksField, 12>;
    for (var i = 0u; i < 12u; i++) {{
        state[i] = input_state[offset + i];
    }}

    // Apply Poseidon2 permutation
    poseidon2_permute(&state);

    // Write result
    for (var i = 0u; i < 12u; i++) {{
        output_state[offset + i] = state[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Poseidon2 Permutation Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Poseidon2 Permutation Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Process tests in batches
    const BATCH_SIZE: usize = 16;
    for (batch_idx, chunk) in test_vectors.chunks(BATCH_SIZE).enumerate() {
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
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let results: &[GfWgls] = bytemuck::cast_slice(&data);

        // Check results for the chunk
        for (i, test_case) in chunk.iter().enumerate() {
            // Convert results back to GoldilocksField arrays using proper conversion
            let gpu_result: [GoldilocksField; 12] = [
                results[i * 12].into(),
                results[i * 12 + 1].into(),
                results[i * 12 + 2].into(),
                results[i * 12 + 3].into(),
                results[i * 12 + 4].into(),
                results[i * 12 + 5].into(),
                results[i * 12 + 6].into(),
                results[i * 12 + 7].into(),
                results[i * 12 + 8].into(),
                results[i * 12 + 9].into(),
                results[i * 12 + 10].into(),
                results[i * 12 + 11].into(),
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
                failed_tests.push(batch_idx * BATCH_SIZE + i + 1);
                failure_details.push((test_case.clone(), gpu_result));
                if failure_details.len() <= 3 {
                    println!("❌ FAILED: Poseidon2 permutation test");
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

    let mut results = TestResults::new("poseidon2 permutation".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_poseidon2_initial_external_rounds(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test just the initial 4 external rounds
    let test_vectors = generate_gf_from_const_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    let shader_source = format!(
        "
{}

@group(0) @binding(0) var<storage, read> input_state: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_state: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let offset = test_id * 12u;

    // Copy input to local state
    var state: array<GoldilocksField, 12>;
    for (var i = 0u; i < 12u; i++) {{
        state[i] = input_state[offset + i];
    }}

    // Apply ONLY the 4 initial external rounds
    for (var round = 0u; round < 4u; round++) {{
        // Add round constants
        for (var i = 0u; i < 12u; i++) {{
            state[i] = gf_add(state[i], gf_from_const(INITIAL_EXTERNAL_CONSTANTS[round][i]));
        }}
        // S-box on all elements
        for (var i = 0u; i < 12u; i++) {{
            state[i] = sbox(state[i]);
        }}
        // External linear layer
        external_linear_layer(&state);
    }}

    // Write result
    for (var i = 0u; i < 12u; i++) {{
        output_state[offset + i] = state[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Initial External Rounds Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Initial External Rounds Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Generate comprehensive test cases
    let mut test_cases = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

    // Zero state
    test_cases.push([GoldilocksField::ZERO; 12]);

    // Unit vectors (one element = 1, rest = 0)
    for i in 0..12 {
        let mut state = [GoldilocksField::ZERO; 12];
        state[i] = GoldilocksField::ONE;
        test_cases.push(state);
    }

    // All ones state
    test_cases.push([GoldilocksField::ONE; 12]);

    // Sequential state [1,2,3,...,12]
    let sequential_state: [GoldilocksField; 12] =
        core::array::from_fn(|i| GoldilocksField::from_canonical_u64((i + 1) as u64));
    test_cases.push(sequential_state);

    // Random small values (all elements < 2^8)
    for _ in 0..15 {
        let mut state = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = (rng.gen::<u8>()) as u64;
            state[j] = GoldilocksField::from_canonical_u64(val);
        }
        test_cases.push(state);
    }

    // Random medium values (all elements < 2^16)
    for _ in 0..10 {
        let mut state = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = (rng.gen::<u16>()) as u64;
            state[j] = GoldilocksField::from_canonical_u64(val);
        }
        test_cases.push(state);
    }

    for (test_idx, input_state) in test_cases.iter().enumerate() {
        let input_data: Vec<GfWgls> = input_state.iter().map(|&f| GfWgls::from(f)).collect();

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Initial External Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Initial External Output Buffer"),
            size: (std::mem::size_of::<GfWgls>() * 12) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Initial External Bind Group"),
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
            label: Some("Initial External Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Initial External Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Initial External Staging Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let gpu_result: &[GfWgls] = bytemuck::cast_slice(&data);

        // Compute CPU reference
        let mut cpu_state = *input_state;
        use qp_poseidon_constants::POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW;

        for round in 0..4 {
            // Add constants
            for i in 0..12 {
                let raw_constant = POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW[round][i];
                cpu_state[i] += GoldilocksField::from_canonical_u64(raw_constant);
            }
            // S-box
            for i in 0..12 {
                cpu_state[i] = cpu_state[i].exp_u64(7);
            }
            // External linear layer (simplified CPU version)
            // This is a basic implementation - if this test fails we'll know it's here
            let two = GoldilocksField::from_canonical_u64(2);
            let three = GoldilocksField::from_canonical_u64(3);

            // 4x4 MDS matrix on chunks
            for chunk in 0..3 {
                let offset = chunk * 4;
                let mut chunk_state = [GoldilocksField::ZERO; 4];
                for i in 0..4 {
                    chunk_state[i] = cpu_state[offset + i];
                }
                let new_0 =
                    two * chunk_state[0] + three * chunk_state[1] + chunk_state[2] + chunk_state[3];
                let new_1 =
                    chunk_state[0] + two * chunk_state[1] + three * chunk_state[2] + chunk_state[3];
                let new_2 =
                    chunk_state[0] + chunk_state[1] + two * chunk_state[2] + three * chunk_state[3];
                let new_3 =
                    three * chunk_state[0] + chunk_state[1] + chunk_state[2] + two * chunk_state[3];
                cpu_state[offset] = new_0;
                cpu_state[offset + 1] = new_1;
                cpu_state[offset + 2] = new_2;
                cpu_state[offset + 3] = new_3;
            }
            // Circulant matrix
            let mut sums = [GoldilocksField::ZERO; 4];
            for k in 0..4 {
                for j in (k..12).step_by(4) {
                    sums[k] += cpu_state[j];
                }
            }
            for i in 0..12 {
                cpu_state[i] += sums[i % 4];
            }
        }

        // Compare
        let mut test_passed = true;
        for i in 0..12 {
            let gpu_gf: GoldilocksField = gpu_result[i].into();
            if gpu_gf.to_canonical_u64() != cpu_state[i].to_canonical_u64() {
                test_passed = false;
                if failed_tests.is_empty() {
                    println!("❌ Initial external rounds test {} failed:", test_idx + 1);
                    for j in 0..3 {
                        println!(
                            "  [{}] Expected: 0x{:016x}, Got: 0x{:016x}",
                            j,
                            cpu_state[j].to_canonical_u64(),
                            gpu_gf.to_canonical_u64()
                        );
                    }
                }
                break;
            }
        }

        if test_passed {
            passed_tests += 1;
        } else {
            failed_tests.push(test_idx + 1);
        }

        drop(data);
        staging_buffer.unmap();
    }

    let mut results = TestResults::new("initial external rounds".to_string(), test_cases.len());
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_poseidon2_terminal_external_rounds(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test just the terminal 4 external rounds
    let total_tests = 2;
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    let shader_source = format!(
        "
{}

@group(0) @binding(0) var<storage, read> input_state: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_state: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let offset = test_id * 12u;

    // Copy input to local state
    var state: array<GoldilocksField, 12>;
    for (var i = 0u; i < 12u; i++) {{
        state[i] = input_state[offset + i];
    }}

    // Apply ONLY the 4 terminal external rounds
    for (var round = 0u; round < 4u; round++) {{
        // Add round constants
        for (var i = 0u; i < 12u; i++) {{
            state[i] = gf_add(state[i], gf_from_const(TERMINAL_EXTERNAL_CONSTANTS[round][i]));
        }}
        // S-box on all elements
        for (var i = 0u; i < 12u; i++) {{
            state[i] = sbox(state[i]);
        }}
        // External linear layer
        external_linear_layer(&state);
    }}

    // Write result
    for (var i = 0u; i < 12u; i++) {{
        output_state[offset + i] = state[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Terminal External Rounds Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Terminal External Rounds Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Generate comprehensive test cases
    let mut test_cases = Vec::new();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

    // Zero state
    test_cases.push([GoldilocksField::ZERO; 12]);

    // Unit vectors (one element = 1, rest = 0)
    for i in 0..12 {
        let mut state = [GoldilocksField::ZERO; 12];
        state[i] = GoldilocksField::ONE;
        test_cases.push(state);
    }

    // All ones state
    test_cases.push([GoldilocksField::ONE; 12]);

    // Sequential state [1,2,3,...,12]
    let sequential_state: [GoldilocksField; 12] =
        core::array::from_fn(|i| GoldilocksField::from_canonical_u64((i + 1) as u64));
    test_cases.push(sequential_state);

    // Random small values (all elements < 2^8)
    for _ in 0..15 {
        let mut state = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = (rng.gen::<u8>()) as u64;
            state[j] = GoldilocksField::from_canonical_u64(val);
        }
        test_cases.push(state);
    }

    // Random medium values (all elements < 2^16)
    for _ in 0..10 {
        let mut state = [GoldilocksField::ZERO; 12];
        for j in 0..12 {
            let val = (rng.gen::<u16>()) as u64;
            state[j] = GoldilocksField::from_canonical_u64(val);
        }
        test_cases.push(state);
    }

    for (test_idx, input_state) in test_cases.iter().enumerate() {
        let input_data: Vec<GfWgls> = input_state.iter().map(|&f| GfWgls::from(f)).collect();

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terminal External Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terminal External Output Buffer"),
            size: (std::mem::size_of::<GfWgls>() * 12) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terminal External Bind Group"),
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
            label: Some("Terminal External Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Terminal External Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terminal External Staging Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let gpu_result: &[GfWgls] = bytemuck::cast_slice(&data);

        // Compute CPU reference
        let mut cpu_state = *input_state;
        use qp_poseidon_constants::POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW;

        for round in 0..4 {
            // Add constants
            for i in 0..12 {
                let raw_constant = POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW[round][i];
                cpu_state[i] += GoldilocksField::from_canonical_u64(raw_constant);
            }
            // S-box
            for i in 0..12 {
                cpu_state[i] = cpu_state[i].exp_u64(7);
            }
            // External linear layer (same as initial external)
            let two = GoldilocksField::from_canonical_u64(2);
            let three = GoldilocksField::from_canonical_u64(3);

            // 4x4 MDS matrix on chunks
            for chunk in 0..3 {
                let offset = chunk * 4;
                let mut chunk_state = [GoldilocksField::ZERO; 4];
                for i in 0..4 {
                    chunk_state[i] = cpu_state[offset + i];
                }
                let new_0 =
                    two * chunk_state[0] + three * chunk_state[1] + chunk_state[2] + chunk_state[3];
                let new_1 =
                    chunk_state[0] + two * chunk_state[1] + three * chunk_state[2] + chunk_state[3];
                let new_2 =
                    chunk_state[0] + chunk_state[1] + two * chunk_state[2] + three * chunk_state[3];
                let new_3 =
                    three * chunk_state[0] + chunk_state[1] + chunk_state[2] + two * chunk_state[3];
                cpu_state[offset] = new_0;
                cpu_state[offset + 1] = new_1;
                cpu_state[offset + 2] = new_2;
                cpu_state[offset + 3] = new_3;
            }
            // Circulant matrix
            let mut sums = [GoldilocksField::ZERO; 4];
            for k in 0..4 {
                for j in (k..12).step_by(4) {
                    sums[k] += cpu_state[j];
                }
            }
            for i in 0..12 {
                cpu_state[i] += sums[i % 4];
            }
        }

        // Compare
        let mut test_passed = true;
        for i in 0..12 {
            let gpu_gf: GoldilocksField = gpu_result[i].into();
            if gpu_gf.to_canonical_u64() != cpu_state[i].to_canonical_u64() {
                test_passed = false;
                if failed_tests.is_empty() {
                    println!("❌ Terminal external rounds test {} failed:", test_idx + 1);
                    for j in 0..3 {
                        println!(
                            "  [{}] Expected: 0x{:016x}, Got: 0x{:016x}",
                            j,
                            cpu_state[j].to_canonical_u64(),
                            gpu_gf.to_canonical_u64()
                        );
                    }
                }
                break;
            }
        }

        if test_passed {
            passed_tests += 1;
        } else {
            failed_tests.push(test_idx + 1);
        }

        drop(data);
        staging_buffer.unmap();
    }

    let mut results = TestResults::new("terminal external rounds".to_string(), test_cases.len());
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_gf_from_const(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test the gf_from_const function that converts constant arrays to GoldilocksField
    let test_vectors = generate_gf_from_const_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    let shader_source = format!(
        "
{}

@group(0) @binding(0) var<storage, read> input_constants: array<array<u32, 2>>;
@group(0) @binding(1) var<storage, read_write> output_fields: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    output_fields[index] = gf_from_const(input_constants[index]);
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gf_from_const Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gf_from_const Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    const BATCH_SIZE: usize = 32;
    for (batch_idx, chunk) in test_vectors.chunks(BATCH_SIZE).enumerate() {
        let input_data: Vec<[u32; 2]> = chunk.iter().map(|tc| [tc.a, tc.b]).collect();

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gf_from_const Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gf_from_const Output Buffer"),
            size: (std::mem::size_of::<GfWgls>() * chunk.len()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gf_from_const Bind Group"),
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
            label: Some("gf_from_const Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gf_from_const Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(chunk.len() as u32, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gf_from_const Staging Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let results: &[GfWgls] = bytemuck::cast_slice(&data);

        for (i, test_case) in chunk.iter().enumerate() {
            let gpu_result = GoldilocksField::from(results[i]);
            if gpu_result.to_canonical_u64() == test_case.expected.to_canonical_u64() {
                passed_tests += 1;
            } else {
                failed_tests.push(batch_idx * BATCH_SIZE + i + 1);
                if failed_tests.len() <= 5 {
                    println!(
                        "❌ gf_from_const test failed: input=[{}, {}], expected=0x{:016x}, got=0x{:016x}",
                        test_case.a, test_case.b,
                        test_case.expected.to_canonical_u64(),
                        gpu_result.to_canonical_u64()
                    );
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    let mut results = TestResults::new("gf_from_const".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_poseidon2_constants_verification(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let total_tests = 24;

    // Create shader that just outputs the constants
    let shader_source = format!(
        "
{}

// Constants verification entry point
@group(0) @binding(0) var<storage, read_write> output_constants: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    // Output first round constants (we know these work)
    for (var i = 0u; i < 12u; i++) {{
        output_constants[i] = gf_from_const(INITIAL_EXTERNAL_CONSTANTS[0][i]);
    }}

    // Output second round constants (these seem to have issues)
    for (var i = 0u; i < 12u; i++) {{
        output_constants[i + 12u] = gf_from_const(INITIAL_EXTERNAL_CONSTANTS[1][i]);
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Constants Verification Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Constants Verification Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Constants Output Buffer"),
        size: (std::mem::size_of::<GfWgls>() * 24) as u64, // 2 rounds * 12 elements
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Constants Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Constants Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Constants Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Constants Staging Buffer"),
        size: (std::mem::size_of::<GfWgls>() * 24) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (std::mem::size_of::<GfWgls>() * 24) as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    let data = buffer_slice.get_mapped_range();
    let gpu_constants: &[GfWgls] = bytemuck::cast_slice(&data);

    use qp_poseidon_constants::POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW;

    for i in 0..12 {
        let gpu_gf: GoldilocksField = gpu_constants[i].into();
        let expected = POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW[0][i];
        if gpu_gf.to_canonical_u64() == expected {
            passed_tests += 1;
        } else {
            failed_tests.push(i + 1);
        }
    }

    for i in 0..12 {
        let gpu_gf: GoldilocksField = gpu_constants[12 + i].into();
        let expected = POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW[1][i];
        if gpu_gf.to_canonical_u64() == expected {
            passed_tests += 1;
        } else {
            failed_tests.push(12 + i + 1);
        }
    }

    drop(data);
    staging_buffer.unmap();

    // Print final summary
    let mut results = TestResults::new("constants verification".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_poseidon2_internal_rounds_only(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test ONLY the internal rounds (22 rounds) without initial/terminal external rounds
    let test_vectors = generate_true_internal_only_test_vectors();
    let total_tests = test_vectors.len();
    let mut results = TestResults::new("internal rounds only".to_string(), total_tests);

    for (test_idx, test_case) in test_vectors.iter().enumerate() {
        // test_case.input is already the state after initial external rounds
        // test_case.expected is the state after internal rounds only
        let input_data: Vec<GfWgls> = test_case.input.map(GfWgls::from).to_vec();

        let shader_source = format!(
            "
{}

@group(0) @binding(0) var<storage, read> input_data: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_data: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    if (index >= 1u) {{
        return;
    }}

    let base_idx = index * 12u;

    // Copy input to local state
    var state: array<GoldilocksField, 12>;
    for (var i = 0u; i < 12u; i++) {{
        state[i] = input_data[base_idx + i];
    }}

    // Apply ONLY internal rounds (22 rounds)
    for (var round = 0u; round < 22u; round++) {{
        // Add round constant to first element only
        state[0] = gf_add(state[0], gf_from_const(INTERNAL_CONSTANTS[round]));
        // S-box on first element only
        state[0] = sbox(state[0]);
        // Internal linear layer (diagonal matrix)
        internal_linear_layer(&state);
    }}

    // Store result
    for (var i = 0u; i < 12u; i++) {{
        output_data[base_idx + i] = state[i];
    }}
}}
",
            include_str!("mining.wgsl")
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Internal Rounds Only Test Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Internal Rounds Only Test Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Internal Rounds Only Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Internal Rounds Only Output Buffer"),
            size: (std::mem::size_of::<GfWgls>() * 12) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Internal Rounds Only Bind Group"),
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
            label: Some("Internal Rounds Only Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Internal Rounds Only Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Internal Rounds Only Staging Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = buffer_slice.get_mapped_range();
        let gpu_result: &[GfWgls] = bytemuck::cast_slice(&data);

        // The expected result is already computed in the test vector
        let cpu_after_internal = test_case.expected;

        // Compare GPU vs CPU results
        let gpu_state: [GoldilocksField; 12] = core::array::from_fn(|j| {
            let wgls_result = gpu_result[j];
            GoldilocksField::from(wgls_result)
        });

        let mut test_passed = true;
        for j in 0..12 {
            if gpu_state[j].to_canonical_u64() != cpu_after_internal[j].to_canonical_u64() {
                test_passed = false;
                break;
            }
        }

        if test_passed {
            results.add_pass();
        } else {
            results.add_failure(test_idx + 1);
            if results.failed.len() <= 5 {
                let expected_strings: Vec<String> = cpu_after_internal
                    .iter()
                    .take(3)
                    .map(|x| format!("0x{:016x}", x.to_canonical_u64()))
                    .collect();
                let actual_strings: Vec<String> = gpu_state
                    .iter()
                    .take(3)
                    .map(|x| format!("0x{:016x}", x.to_canonical_u64()))
                    .collect();
                print_test_failure(
                    "Internal rounds only",
                    test_idx + 1,
                    &expected_strings,
                    &actual_strings,
                    3,
                );
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    // Print final summary
    results.print_summary()
}

pub async fn test_poseidon2_internal_constants_verification(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let total_tests = 22;

    // Create shader that just outputs the internal constants
    let shader_source = format!(
        "
{}

// Internal constants verification entry point
@group(0) @binding(0) var<storage, read_write> output_constants: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    // Output all 22 internal constants
    for (var i = 0u; i < 22u; i++) {{
        output_constants[i] = gf_from_const(INTERNAL_CONSTANTS[i]);
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Internal Constants Verification Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Internal Constants Verification Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Internal Constants Output Buffer"),
        size: (std::mem::size_of::<GfWgls>() * 22) as u64, // 22 internal constants
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Internal Constants Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Internal Constants Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Internal Constants Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Internal Constants Staging Buffer"),
        size: (std::mem::size_of::<GfWgls>() * 22) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (std::mem::size_of::<GfWgls>() * 22) as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    let data = buffer_slice.get_mapped_range();
    let gpu_constants: &[GfWgls] = bytemuck::cast_slice(&data);

    use qp_poseidon_constants::POSEIDON2_INTERNAL_CONSTANTS_RAW;

    for i in 0..22 {
        let gpu_gf: GoldilocksField = gpu_constants[i].into();
        let expected = POSEIDON2_INTERNAL_CONSTANTS_RAW[i];
        if gpu_gf.to_canonical_u64() == expected {
            passed_tests += 1;
        } else {
            failed_tests.push(i + 1);
        }
    }

    let mut results = TestResults::new("internal constants".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_poseidon2_terminal_external_rounds_issue(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing terminal external rounds issue for sequential input...");

    // Create shader that tests terminal external rounds specifically
    let shader_source = format!(
        "
{}

// Debug entry point for terminal external rounds
@group(0) @binding(0) var<storage, read> state_before_terminal: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> after_terminal: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let offset = test_id * 12u;

    // Copy the state after initial+internal rounds
    var state: array<GoldilocksField, 12>;
    for (var i = 0u; i < 12u; i++) {{
        state[i] = state_before_terminal[offset + i];
    }}

    // Apply terminal external rounds
    for (var round = 0u; round < 4u; round++) {{
        for (var i = 0u; i < 12u; i++) {{
            state[i] = gf_add(state[i], gf_from_const(TERMINAL_EXTERNAL_CONSTANTS[round][i]));
        }}
        for (var i = 0u; i < 12u; i++) {{
            state[i] = sbox(state[i]);
        }}
        external_linear_layer(&state);
    }}

    // Save final result
    for (var i = 0u; i < 12u; i++) {{
        after_terminal[offset + i] = state[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Terminal External Rounds Issue Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Terminal External Rounds Issue Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Use the state after initial+internal for sequential input: 0x5f14ee0d55368aa5, etc.
    let state_after_internal = [
        GoldilocksField::from_canonical_u64(0x5f14ee0d55368aa5),
        GoldilocksField::from_canonical_u64(0xd858a148e6ba3c15),
        GoldilocksField::from_canonical_u64(0xa63e197036ba7747),
        // We need to get the full state - let me create a more complete test
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
    ];

    let mut input_data = Vec::new();
    for &field_elem in &state_after_internal {
        input_data.push(GfWgls::from(field_elem));
    }

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Terminal Issue Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Terminal Issue Output Buffer"),
        size: (std::mem::size_of::<GfWgls>() * 12) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Terminal Issue Bind Group"),
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
        label: Some("Terminal Issue Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Terminal Issue Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Terminal Issue Staging Buffer"),
        size: (std::mem::size_of::<GfWgls>() * 12) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (std::mem::size_of::<GfWgls>() * 12) as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    let data = buffer_slice.get_mapped_range();
    let gpu_results: &[GfWgls] = bytemuck::cast_slice(&data);

    println!("🔍 GPU result after terminal external rounds:");
    for i in 0..3 {
        let gf: GoldilocksField = gpu_results[i].into();
        println!("  [{}] = 0x{:016x}", i, gf.to_canonical_u64());
    }

    println!("Expected final result for sequential [1,2,3...12]: 0x7e9574e2a3d6c48b");
    println!("This test helps isolate if terminal external constants are wrong");

    drop(data);
    staging_buffer.unmap();

    Ok(())
}

pub async fn test_poseidon2_terminal_external_constants_verification(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let total_tests = 24;

    // Create shader that just outputs the terminal external constants
    let shader_source = format!(
        "
{}

// Terminal constants verification entry point
@group(0) @binding(0) var<storage, read_write> output_constants: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    // Output first round terminal constants
    for (var i = 0u; i < 12u; i++) {{
        output_constants[i] = gf_from_const(TERMINAL_EXTERNAL_CONSTANTS[0][i]);
    }}

    // Output second round terminal constants
    for (var i = 0u; i < 12u; i++) {{
        output_constants[i + 12u] = gf_from_const(TERMINAL_EXTERNAL_CONSTANTS[1][i]);
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Terminal Constants Verification Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Terminal Constants Verification Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Terminal Constants Output Buffer"),
        size: (std::mem::size_of::<GfWgls>() * 24) as u64, // 2 rounds * 12 elements
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Terminal Constants Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Terminal Constants Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Terminal Constants Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Terminal Constants Staging Buffer"),
        size: (std::mem::size_of::<GfWgls>() * 24) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (std::mem::size_of::<GfWgls>() * 24) as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    let data = buffer_slice.get_mapped_range();
    let gpu_constants: &[GfWgls] = bytemuck::cast_slice(&data);

    use qp_poseidon_constants::POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW;

    for i in 0..12 {
        let gpu_gf: GoldilocksField = gpu_constants[i].into();
        let expected = POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW[0][i];
        if gpu_gf.to_canonical_u64() == expected {
            passed_tests += 1;
        } else {
            failed_tests.push(i + 1);
        }
    }

    for i in 0..12 {
        let gpu_gf: GoldilocksField = gpu_constants[12 + i].into();
        let expected = POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW[1][i];
        if gpu_gf.to_canonical_u64() == expected {
            passed_tests += 1;
        } else {
            failed_tests.push(12 + i + 1);
        }
    }

    // Print final summary
    let mut results = TestResults::new("terminal external constants".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
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
        entry_point: Some("main"),
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
    let mut results = TestResults::new("gf_mul".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_mds_matrix(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_mds_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let mut failure_details = Vec::new();

    // Read the mining shader source and create a test wrapper
    let mining_shader_source = include_str!("mining.wgsl");
    let shader_source = format!(
        "
{}

// MDS matrix test entry point
@group(0) @binding(0) var<storage, read> input_data: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_data: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
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
        entry_point: Some("main"),
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
                let test_index = i + 1;
                failed_tests.push(test_index);
                failure_details.push((test_case.clone(), gpu_result));
                if failure_details.len() <= 5 {
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

    let mut results = TestResults::new("mds matrix".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_internal_linear_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_internal_linear_layer_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let mut failure_details = Vec::new();

    // Read the mining shader source and create a test wrapper
    let mining_shader_source = include_str!("mining.wgsl");
    let shader_source = format!(
        "
{}

// Test entry point for internal linear layer
@group(0) @binding(0) var<storage, read> input_state: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_state: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
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
        entry_point: Some("main"),
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
                let test_index = i + 1;
                failed_tests.push(test_index);
                failure_details.push((test_case.clone(), gpu_result));
                if failure_details.len() <= 3 {
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

    let mut results = TestResults::new("internal linear layer".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_external_linear_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_external_linear_layer_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    const BATCH_SIZE: usize = 20;

    // Use the same pattern as other successful tests - include mining.wgsl and define our own bindings
    let shader_source = format!(
        "
{}

// Simple test entry point for external linear layer
@group(0) @binding(0) var<storage, read> input_state: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_state: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let offset = test_id * 12u;

    // Copy input to local state
    var state: array<GoldilocksField, 12>;
    for (var i = 0u; i < 12u; i++) {{
        state[i] = input_state[offset + i];
    }}

    // Apply external linear layer
    external_linear_layer(&state);

    // Write result
    for (var i = 0u; i < 12u; i++) {{
        output_state[offset + i] = state[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("External Linear Layer Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("External Linear Layer Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Process tests in batches
    for (batch_start, batch) in test_vectors.chunks(BATCH_SIZE).enumerate() {
        let batch_size = batch.len();

        // Prepare input data
        let mut input_data = Vec::with_capacity(batch_size * 12);
        for vector in batch {
            for &element in &vector.input {
                let gf_wgls: GfWgls = element.into();
                input_data.push(gf_wgls);
            }
        }

        // Create buffers
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("External Linear Layer Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("External Linear Layer Output Buffer"),
            size: (batch_size * 12 * std::mem::size_of::<GfWgls>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("External Linear Layer Readback Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            label: Some("External Linear Layer Test Bind Group"),
        });

        // Execute
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("External Linear Layer Test Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("External Linear Layer Test Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_buffer.size());
        queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let gpu_results: &[GfWgls] = bytemuck::cast_slice(&data);

        // Verify results
        for (i, vector) in batch.iter().enumerate() {
            let offset = i * 12;
            let mut all_match = true;

            for j in 0..12 {
                let gpu_result: GoldilocksField = gpu_results[offset + j].into();
                let expected = vector.expected[j];
                if gpu_result != expected {
                    all_match = false;
                    break;
                }
            }

            if all_match {
                passed_tests += 1;
            } else {
                let test_index = batch_start * BATCH_SIZE + i;
                failed_tests.push(test_index);
            }
        }

        drop(data);
        readback_buffer.unmap();
    }

    // Print results summary
    let mut results = TestResults::new("external linear layer".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_sbox(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_sbox_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let mut failure_details = Vec::new();

    const BATCH_SIZE: usize = 64;
    let batches = test_vectors.chunks(BATCH_SIZE);

    for (_batch_idx, batch) in batches.enumerate() {
        // Convert test vectors to WGSL format
        let input_data: Vec<GfWgls> = batch
            .iter()
            .map(|test_case| GfWgls::from(test_case.input))
            .collect();

        // Create compute shader for S-box testing using the same pattern as other tests
        let shader_source = format!(
            "
{}

@group(0) @binding(0) var<storage, read> input_data: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_data: array<GoldilocksField>;

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
",
            include_str!("mining.wgsl"),
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
                let test_index = i + 1;
                failed_tests.push(test_index);
                failure_details.push((test_case.clone(), gpu_result));
                if failure_details.len() <= 10 {
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

    let mut results = TestResults::new("s-box".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_poseidon2_squeeze_twice(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_poseidon2_squeeze_twice_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let mut failure_details = Vec::new();

    let shader_source = format!(
        "
{}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_hashes: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let input_offset = test_id * 24u; // 24 u32s = 96 bytes per test

    // Copy input data to array (exactly 96 bytes = 24 u32s)
    var input_array: array<u32, 24>;
    for (var i = 0u; i < 24u; i++) {{
        input_array[i] = input_data[input_offset + i];
    }}

    // Hash the input (96 bytes -> 64 bytes)
    let hash_result = poseidon2_hash_squeeze_twice(input_array);

    // Write first 32 bytes (8 u32s) of result as the hash
    let output_offset = test_id * 8u;
    for (var i = 0u; i < 8u; i++) {{
        output_hashes[output_offset + i] = hash_result[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Poseidon2 Squeeze Twice Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Poseidon2 Squeeze Twice Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Process tests in batches
    const BATCH_SIZE: usize = 16;
    for (batch_idx, chunk) in test_vectors.chunks(BATCH_SIZE).enumerate() {
        // Prepare input data - all inputs are exactly 96 bytes
        let mut input_data = Vec::new();

        for test_case in chunk {
            // Ensure input is exactly 96 bytes
            assert_eq!(
                test_case.input_bytes.len(),
                96,
                "All test inputs must be exactly 96 bytes"
            );

            // Convert bytes to u32s (96 bytes = 24 u32s)
            for chunk_4 in test_case.input_bytes.chunks(4) {
                let u32_val = u32::from_le_bytes([chunk_4[0], chunk_4[1], chunk_4[2], chunk_4[3]]);
                input_data.push(u32_val);
            }
        }

        let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Data Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (chunk.len() * 8 * 4) as u64, // 8 u32s per hash * 4 bytes per u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_data_buffer.as_entire_binding(),
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
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let results: &[u32] = bytemuck::cast_slice(&data);

        // Check results for the chunk
        for (i, test_case) in chunk.iter().enumerate() {
            // Get GPU result (first 32 bytes = 8 u32s)
            let gpu_hash_u32s = &results[i * 8..(i + 1) * 8];
            let mut gpu_hash = [0u8; 32];
            for (j, &u32_val) in gpu_hash_u32s.iter().enumerate() {
                let bytes = u32_val.to_le_bytes();
                gpu_hash[j * 4..(j + 1) * 4].copy_from_slice(&bytes);
            }

            if gpu_hash == test_case.expected_hash {
                passed_tests += 1;
            } else {
                failed_tests.push(batch_idx * BATCH_SIZE + i + 1);
                failure_details.push((test_case.clone(), gpu_hash));

                if failure_details.len() <= 3 {
                    println!("❌ FAILED: Poseidon2 squeeze-twice test (96 bytes input)");
                    println!(
                        "Input bytes (first 32): {:02x?}",
                        &test_case.input_bytes[..32]
                    );
                    println!("Expected hash: {:02x?}", &test_case.expected_hash[..]);
                    println!("GPU hash:     {:02x?}", &gpu_hash[..]);
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    let mut results = TestResults::new("poseidon2 squeeze-twice".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_bytes_to_field_elements(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_bytes_to_field_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let mut failure_details = Vec::new();

    let shader_source = format!(
        "
{}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_felts: array<GoldilocksField>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let input_offset = test_id * 24u; // 24 u32s = 96 bytes per test

    // Copy input data to array (exactly 96 bytes = 24 u32s)
    var input_array: array<u32, 24>;
    for (var i = 0u; i < 24u; i++) {{
        input_array[i] = input_data[input_offset + i];
    }}

    // Convert to field elements
    let felts = bytes_to_field_elements(input_array);

    // Write result (25 field elements per test)
    let output_offset = test_id * 25u;
    for (var i = 0u; i < 25u; i++) {{
        output_felts[output_offset + i] = felts[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Bytes to Field Elements Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Bytes to Field Elements Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Process tests in batches
    const BATCH_SIZE: usize = 8;
    for (batch_idx, chunk) in test_vectors.chunks(BATCH_SIZE).enumerate() {
        // Prepare input data
        let mut input_data = Vec::new();
        for test_case in chunk {
            // Convert bytes to u32s (96 bytes = 24 u32s)
            for chunk_4 in test_case.input_bytes.chunks(4) {
                let u32_val = u32::from_le_bytes([chunk_4[0], chunk_4[1], chunk_4[2], chunk_4[3]]);
                input_data.push(u32_val);
            }
        }

        let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Data Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (chunk.len() * 25 * std::mem::size_of::<GfWgls>()) as u64, // 25 felts per test
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_data_buffer.as_entire_binding(),
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
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let results: &[GfWgls] = bytemuck::cast_slice(&data);

        // Check results for the chunk
        for (i, test_case) in chunk.iter().enumerate() {
            let mut all_match = true;
            let mut gpu_felts = Vec::new();

            // Convert GPU results back to GoldilocksField
            for j in 0..25 {
                let gpu_felt: GoldilocksField = results[i * 25 + j].into();
                gpu_felts.push(gpu_felt);

                if j < test_case.expected_felts.len() {
                    if gpu_felt.to_canonical_u64() != test_case.expected_felts[j].to_canonical_u64()
                    {
                        all_match = false;
                    }
                }
            }

            if all_match && gpu_felts.len() == test_case.expected_felts.len() {
                passed_tests += 1;
            } else {
                failed_tests.push(batch_idx * BATCH_SIZE + i + 1);
                failure_details.push((test_case.clone(), gpu_felts.clone()));

                if failure_details.len() <= 3 {
                    println!("❌ FAILED: Bytes to field elements test");
                    println!(
                        "Input bytes (first 16): {:02x?}",
                        &test_case.input_bytes[..16]
                    );
                    println!("Expected {} elements:", test_case.expected_felts.len());
                    for (j, &expected) in test_case.expected_felts.iter().enumerate().take(5) {
                        println!("  [{}] = 0x{:016x}", j, expected.to_canonical_u64());
                    }
                    println!("GPU produced {} elements:", gpu_felts.len());
                    for (j, &gpu) in gpu_felts.iter().enumerate().take(5) {
                        println!("  [{}] = 0x{:016x}", j, gpu.to_canonical_u64());
                    }
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    let mut results = TestResults::new("bytes to field elements".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_field_elements_to_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_field_to_bytes_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let mut failure_details = Vec::new();

    let shader_source = format!(
        "
{}

@group(0) @binding(0) var<storage, read> input_felts: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read_write> output_bytes: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let input_offset = test_id * 4u; // 4 field elements per test

    // Copy input field elements
    var felts: array<GoldilocksField, 4>;
    for (var i = 0u; i < 4u; i++) {{
        felts[i] = input_felts[input_offset + i];
    }}

    // Convert to bytes
    let bytes_result = field_elements_to_bytes(felts);

    // Write result (8 u32s = 32 bytes per test)
    let output_offset = test_id * 8u;
    for (var i = 0u; i < 8u; i++) {{
        output_bytes[output_offset + i] = bytes_result[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Field Elements to Bytes Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Field Elements to Bytes Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Process tests in batches
    const BATCH_SIZE: usize = 8;
    for (batch_idx, chunk) in test_vectors.chunks(BATCH_SIZE).enumerate() {
        // Prepare input data
        let mut input_data = Vec::new();
        for test_case in chunk {
            for &felt in &test_case.input_felts {
                input_data.push(GfWgls::from(felt));
            }
        }

        let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Data Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (chunk.len() * 8 * 4) as u64, // 8 u32s per test * 4 bytes per u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_data_buffer.as_entire_binding(),
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
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let results: &[u32] = bytemuck::cast_slice(&data);

        // Check results for the chunk
        for (i, test_case) in chunk.iter().enumerate() {
            // Get GPU result (8 u32s = 32 bytes)
            let gpu_bytes_u32s = &results[i * 8..(i + 1) * 8];
            let mut gpu_bytes = [0u8; 32];
            for (j, &u32_val) in gpu_bytes_u32s.iter().enumerate() {
                let bytes = u32_val.to_le_bytes();
                gpu_bytes[j * 4..(j + 1) * 4].copy_from_slice(&bytes);
            }

            if gpu_bytes == test_case.expected_bytes {
                passed_tests += 1;
            } else {
                failed_tests.push(batch_idx * BATCH_SIZE + i + 1);
                failure_details.push((test_case.clone(), gpu_bytes));

                if failure_details.len() <= 3 {
                    println!("❌ FAILED: Field elements to bytes test");
                    println!("Input felts:");
                    for (j, &felt) in test_case.input_felts.iter().enumerate() {
                        println!("  [{}] = 0x{:016x}", j, felt.to_canonical_u64());
                    }
                    println!("Expected bytes: {:02x?}", &test_case.expected_bytes[..]);
                    println!("GPU bytes:     {:02x?}", &gpu_bytes[..]);
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    let mut results = TestResults::new("field elements to bytes".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}

pub async fn test_double_hash(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_vectors = generate_double_hash_test_vectors();
    let total_tests = test_vectors.len();
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();
    let mut failure_details = Vec::new();

    let shader_source = format!(
        "
{}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_hashes: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let test_id = global_id.x;
    let input_offset = test_id * 24u; // 24 u32s = 96 bytes per test

    // Copy input data to array (exactly 96 bytes = 24 u32s)
    var input_array: array<u32, 24>;
    for (var i = 0u; i < 24u; i++) {{
        input_array[i] = input_data[input_offset + i];
    }}

    // Hash the input using double hash (96 bytes -> 64 bytes)
    let hash_result = double_hash(input_array);

    // Write full 64 bytes (16 u32s) of result as the hash
    let output_offset = test_id * 16u;
    for (var i = 0u; i < 16u; i++) {{
        output_hashes[output_offset + i] = hash_result[i];
    }}
}}
",
        include_str!("mining.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Double Hash Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Double Hash Test Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Process tests in batches
    const BATCH_SIZE: usize = 8;
    for (batch_idx, chunk) in test_vectors.chunks(BATCH_SIZE).enumerate() {
        // Prepare input data - all inputs are exactly 96 bytes
        let mut input_data = Vec::new();

        for test_case in chunk {
            // Convert bytes to u32s (96 bytes = 24 u32s)
            for chunk_4 in test_case.input_96_bytes.chunks(4) {
                let u32_val = u32::from_le_bytes([chunk_4[0], chunk_4[1], chunk_4[2], chunk_4[3]]);
                input_data.push(u32_val);
            }
        }

        let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Data Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (chunk.len() * 16 * 4) as u64, // 16 u32s per hash * 4 bytes per u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_data_buffer.as_entire_binding(),
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
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let results: &[u32] = bytemuck::cast_slice(&data);

        // Check results for the chunk
        for (i, test_case) in chunk.iter().enumerate() {
            // Get GPU result (full 64 bytes = 16 u32s)
            let gpu_hash_u32s = &results[i * 16..(i + 1) * 16];
            let mut gpu_hash = [0u8; 64];
            for (j, &u32_val) in gpu_hash_u32s.iter().enumerate() {
                let bytes = u32_val.to_le_bytes();
                gpu_hash[j * 4..(j + 1) * 4].copy_from_slice(&bytes);
            }

            if gpu_hash == test_case.expected_hash {
                passed_tests += 1;
            } else {
                failed_tests.push(batch_idx * BATCH_SIZE + i + 1);
                failure_details.push((test_case.clone(), gpu_hash));

                if failure_details.len() <= 3 {
                    println!("❌ FAILED: Double hash test ({})", test_case.description);
                    println!(
                        "Input (first 16 bytes): {:02x?}",
                        &test_case.input_96_bytes[..16]
                    );
                    println!("Expected hash: {:02x?}", &test_case.expected_hash[..]);
                    println!("GPU hash:     {:02x?}", &gpu_hash[..]);

                    // Show first difference
                    for k in 0..64 {
                        if gpu_hash[k] != test_case.expected_hash[k] {
                            println!(
                                "First difference at byte {}: Expected={:02x}, GPU={:02x}",
                                k, test_case.expected_hash[k], gpu_hash[k]
                            );
                            break;
                        }
                    }
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    let mut results = TestResults::new("double hash".to_string(), total_tests);
    results.passed = passed_tests;
    results.failed = failed_tests;
    results.print_summary()
}
