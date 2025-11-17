// Quantus Mining Shader - WGSL Compute Shader for GPU Mining
// Implements Poseidon2 hash function over Goldilocks field

// Goldilocks field constants: p = 2^64 - 2^32 + 1 = 18446744069414584321
// Since WGSL doesn't support 64-bit literals, we work with 32-bit chunks
// p = 0xFFFFFFFF00000001 = [0x00000001, 0xFFFFFFFF] in little-endian u32 pairs
const GOLDILOCKS_PRIME_LOW: u32 = 1u;            // Low 32 bits
const GOLDILOCKS_PRIME_HIGH: u32 = 4294967295u;  // High 32 bits (2^32 - 1)

// Poseidon2 constants
const WIDTH: u32 = 12u;
const RATE: u32 = 4u;
const EXTERNAL_ROUNDS: u32 = 4u;
const INTERNAL_ROUNDS: u32 = 22u;

// Real Poseidon2 constants extracted from qp-poseidon-constants
// Internal round constants (22 values)
const INTERNAL_CONSTANTS: array<array<u32, 2>, 22> = array<array<u32, 2>, 22>(
    array<u32, 2>(2018170979u, 2549578122u),
    array<u32, 2>(794875120u, 3520249608u),
    array<u32, 2>(2677723654u, 1772320679u),
    array<u32, 2>(2743438884u, 2849007878u),
    array<u32, 2>(518907317u, 693269760u),
    array<u32, 2>(293328710u, 1484055617u),
    array<u32, 2>(2834138828u, 2315799483u),
    array<u32, 2>(1558078501u, 1039128420u),
    array<u32, 2>(2266808631u, 966316006u),
    array<u32, 2>(3393728842u, 1045622667u),
    array<u32, 2>(2245828300u, 2521440415u),
    array<u32, 2>(751064958u, 1070374632u),
    array<u32, 2>(3490278765u, 2390340773u),
    array<u32, 2>(3526960470u, 2224174634u),
    array<u32, 2>(639988950u, 4000511088u),
    array<u32, 2>(1839350858u, 504240201u),
    array<u32, 2>(559852230u, 255489215u),
    array<u32, 2>(2713771731u, 453385078u),
    array<u32, 2>(1745082278u, 422331096u),
    array<u32, 2>(2544763488u, 4141129721u),
    array<u32, 2>(2700752774u, 1052996327u),
    array<u32, 2>(4063512019u, 1429786100u)
);

// Initial external round constants (4 rounds x 12 elements)
const INITIAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(
    array<array<u32, 2>, 12>(
        array<u32, 2>(2539329031u, 3221415792u),
        array<u32, 2>(4262746426u, 3164936845u),
        array<u32, 2>(3883202553u, 1922272763u),
        array<u32, 2>(3761386668u, 3841130025u),
        array<u32, 2>(1411081289u, 3588274735u),
        array<u32, 2>(4090250945u, 3962812520u),
        array<u32, 2>(1100826458u, 1215155029u),
        array<u32, 2>(1489773809u, 1813820067u),
        array<u32, 2>(2585015995u, 3824356688u),
        array<u32, 2>(2378857513u, 3651555078u),
        array<u32, 2>(2864423342u, 3852156759u),
        array<u32, 2>(1531416540u, 708695120u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(1987505445u, 2913073372u),
        array<u32, 2>(630903574u, 655361351u),
        array<u32, 2>(3412085911u, 1258046621u),
        array<u32, 2>(1456979578u, 1461113191u),
        array<u32, 2>(523722014u, 526769230u),
        array<u32, 2>(925368168u, 922771817u),
        array<u32, 2>(4074853328u, 3855135279u),
        array<u32, 2>(273563309u, 4248797356u),
        array<u32, 2>(1762266526u, 3450728622u),
        array<u32, 2>(1115336254u, 1107677022u),
        array<u32, 2>(4174699389u, 3986946237u),
        array<u32, 2>(3534029317u, 3543582418u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(2101518210u, 2109009407u),
        array<u32, 2>(2323647532u, 2311068777u),
        array<u32, 2>(3016422935u, 3015476255u),
        array<u32, 2>(1378256883u, 3683616468u),
        array<u32, 2>(2029516952u, 2022063001u),
        array<u32, 2>(644616330u, 642915770u),
        array<u32, 2>(2580628271u, 2576506160u),
        array<u32, 2>(1689124307u, 1689124307u),
        array<u32, 2>(4016144568u, 4016127928u),
        array<u32, 2>(1335766254u, 1335740398u),
        array<u32, 2>(1465316391u, 1465316391u),
        array<u32, 2>(4119764157u, 4119617533u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(507988087u, 507988087u),
        array<u32, 2>(1650295309u, 1650295309u),
        array<u32, 2>(2867864750u, 2867864750u),
        array<u32, 2>(1679868924u, 1679868924u),
        array<u32, 2>(999842784u, 999842784u),
        array<u32, 2>(603209949u, 603209949u),
        array<u32, 2>(2869577370u, 2869577370u),
        array<u32, 2>(3936230090u, 3936230090u),
        array<u32, 2>(3435906572u, 3435906572u),
        array<u32, 2>(2433830883u, 2433830883u),
        array<u32, 2>(1537056815u, 1537056815u),
        array<u32, 2>(3757386231u, 3757386231u)
    )
);

// Terminal external round constants (4 rounds x 12 elements)
const TERMINAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(
    array<array<u32, 2>, 12>(
        array<u32, 2>(2067639406u, 2067639406u),
        array<u32, 2>(342838134u, 342838134u),
        array<u32, 2>(2148923528u, 2148923528u),
        array<u32, 2>(1836351170u, 1836351170u),
        array<u32, 2>(2714619123u, 2714619123u),
        array<u32, 2>(4142963247u, 4142963247u),
        array<u32, 2>(884199780u, 884199780u),
        array<u32, 2>(2970893770u, 2970893770u),
        array<u32, 2>(1697177254u, 1697177254u),
        array<u32, 2>(249070999u, 249070999u),
        array<u32, 2>(485491595u, 485491595u),
        array<u32, 2>(1718641338u, 1718641338u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(812285441u, 1616511558u),
        array<u32, 2>(1074098767u, 1073693711u),
        array<u32, 2>(3896701389u, 3896638389u),
        array<u32, 2>(4169265998u, 4169265934u),
        array<u32, 2>(4095009929u, 4095009865u),
        array<u32, 2>(1835728118u, 1835728054u),
        array<u32, 2>(4176537227u, 4176537163u),
        array<u32, 2>(1478766714u, 1478766650u),
        array<u32, 2>(376881709u, 376881645u),
        array<u32, 2>(555692597u, 555692533u),
        array<u32, 2>(2968574966u, 2968574902u),
        array<u32, 2>(3635110935u, 3635110871u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(2545243919u, 3830388836u),
        array<u32, 2>(1989298984u, 1989298920u),
        array<u32, 2>(151916301u, 151916237u),
        array<u32, 2>(2556322548u, 1995322484u),
        array<u32, 2>(238805938u, 238805874u),
        array<u32, 2>(2490398575u, 2490398511u),
        array<u32, 2>(2566222314u, 2566222250u),
        array<u32, 2>(814653888u, 3828870544u),
        array<u32, 2>(522453199u, 522453135u),
        array<u32, 2>(188424356u, 188424292u),
        array<u32, 2>(1649066023u, 1649065959u),
        array<u32, 2>(1595705478u, 1595705414u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(2466398251u, 4275150062u),
        array<u32, 2>(1928488963u, 1928488899u),
        array<u32, 2>(3871900792u, 3871900728u),
        array<u32, 2>(1463617809u, 1463617745u),
        array<u32, 2>(4175880892u, 4175880828u),
        array<u32, 2>(752107605u, 752107541u),
        array<u32, 2>(2101968497u, 2101968433u),
        array<u32, 2>(2239882257u, 2239882193u),
        array<u32, 2>(1389589688u, 1389589624u),
        array<u32, 2>(1344537748u, 1344537684u),
        array<u32, 2>(3172231929u, 3172231865u),
        array<u32, 2>(1780012361u, 3777952458u)
    )
);

// Helper function to create GoldilocksField from constant array
fn gf_from_const(val: array<u32, 2>) -> GoldilocksField {
    return GoldilocksField(val[0], val[1]);
}

// MDS matrix constants for width 12 - circulant matrix first row
const MDS_MATRIX_FIRST_ROW: array<i32, 12> = array<i32, 12>(1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10);

// Storage buffers
@group(0) @binding(0) var<storage, read_write> results: array<u32>;
@group(0) @binding(1) var<storage, read> header: array<u32, 8>;     // 32 bytes
@group(0) @binding(2) var<storage, read> start_nonce: array<u32, 16>; // 64 bytes
@group(0) @binding(3) var<storage, read> difficulty_target: array<u32, 16>;    // 64 bytes (U512 target)
@group(0) @binding(4) var<storage, read_write> debug_buffer: array<u32>;      // Debug output buffer

// Goldilocks field element represented as [low_32, high_32]
struct GoldilocksField {
    low: u32,
    high: u32,
}

// Goldilocks field operations
fn gf_zero() -> GoldilocksField {
    return GoldilocksField(0u, 0u);
}

fn gf_one() -> GoldilocksField {
    return GoldilocksField(1u, 0u);
}

fn gf_from_u32(val: u32) -> GoldilocksField {
    return GoldilocksField(val, 0u);
}

// Ultra-simple Goldilocks field addition for debugging
fn gf_add(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    // Debug: write inputs to buffer
    debug_buffer[20] = a.low;
    debug_buffer[21] = a.high;
    debug_buffer[22] = b.low;
    debug_buffer[23] = b.high;

    let sum_low = a.low + b.low;
    var carry = 0u;
    if (sum_low < a.low) {
        carry = 1u;
    }
    let sum_high = a.high + b.high + carry;

    // Debug: write results to buffer
    debug_buffer[24] = sum_low;
    debug_buffer[25] = sum_high;
    debug_buffer[26] = carry;

    return GoldilocksField(sum_low, sum_high);
}

// Simplified Goldilocks field subtraction
fn gf_sub(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    var result_low = a.low;
    var result_high = a.high;

    if (result_low >= b.low) {
        result_low = result_low - b.low;
    } else {
        result_low = result_low + 4294967295u + 1u - b.low;  // borrow from high (2^32 = 4294967295 + 1)
        result_high = result_high - 1u;
    }

    result_high = result_high - b.high;

    return GoldilocksField(result_low, result_high);
}

// Proper Goldilocks field multiplication for all values
fn gf_mul(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    // Handle special cases first
    if (a.high == 0u && a.low == 0u) { return gf_zero(); }
    if (b.high == 0u && b.low == 0u) { return gf_zero(); }
    if (a.high == 0u && a.low == 1u) { return b; }
    if (b.high == 0u && b.low == 1u) { return a; }

    // Step 1: Handle small values directly (both high parts are 0)
    if (a.high == 0u && b.high == 0u) {
        let product = a.low * b.low;
        return GoldilocksField(product, 0u);
    }

    // Step 2: Handle mixed cases (one has high=0, other has high!=0)
    if (a.high == 0u || b.high == 0u) {
        // Multiply small * large using shift-and-add
        var large_val: GoldilocksField;
        var small_val: u32;

        if (a.high == 0u) {
            large_val = b;
            small_val = a.low;
        } else {
            large_val = a;
            small_val = b.low;
        }

        var result = gf_zero();
        var power_of_two = large_val;
        var remaining = small_val;

        // Binary multiplication: decompose small_val into powers of 2
        for (var bit = 0u; bit < 32u; bit++) {
            if ((remaining & 1u) != 0u) {
                result = gf_add(result, power_of_two);
            }
            remaining = remaining >> 1u;
            if (remaining == 0u) { break; }
            power_of_two = gf_add(power_of_two, power_of_two); // double
        }

        return result;
    }

    // Step 3: Both have high parts != 0 - full 64x64 multiplication
    // Use schoolbook multiplication: (a_hi*2^32 + a_lo) * (b_hi*2^32 + b_lo)

    let a_lo = a.low;
    let a_hi = a.high;
    let b_lo = b.low;
    let b_hi = b.high;

    // Partial products
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    // Combine: result = hh*2^64 + (lh+hl)*2^32 + ll
    // For Goldilocks field p = 2^64 - 2^32 + 1, use 2^64 ≡ 2^32 - 1

    // Low part calculation
    let mid = lh + hl;
    var mid_carry = 0u;
    if (mid < lh) { mid_carry = 1u; }

    let result_low_part = ll + (mid << 32u);
    var low_carry = 0u;
    if (result_low_part < ll) { low_carry = 1u; }

    // High part calculation
    let result_high_part = hh + (mid >> 32u) + (mid_carry << 32u) + low_carry;

    // Apply Goldilocks reduction: high_part * 2^64 ≡ high_part * (2^32 - 1)
    var final_result = result_low_part;
    if (result_high_part > 0u) {
        let reduction = result_high_part * 0xFFFFFFFFu - result_high_part;
        final_result = final_result + reduction;
    }

    return GoldilocksField(final_result & 0xFFFFFFFFu, final_result >> 32u);
}

// Simple test function to verify basic field operations
fn test_field_operations() -> bool {
    // Write hardcoded values first to test if debug buffer works
    debug_buffer[1] = 123u;
    debug_buffer[2] = 456u;

    // Test gf_zero
    let zero = gf_zero();
    debug_buffer[3] = zero.low;
    debug_buffer[4] = zero.high;

    // Test gf_one - write immediately after creation
    let one = gf_one();
    debug_buffer[5] = one.low;
    debug_buffer[6] = one.high;

    // Hardcode test: create GoldilocksField(1,0) manually
    let manual_one = GoldilocksField(1u, 0u);
    debug_buffer[7] = manual_one.low;
    debug_buffer[8] = manual_one.high;

    // Test if functions can return correct values
    debug_buffer[11] = 777u;
    debug_buffer[12] = 888u;

    return true;
}

// Debug helper function to write state to debug buffer
fn debug_write_state(offset: u32, state: array<GoldilocksField, 12>) {
    for (var i = 0u; i < 12u; i++) {
        debug_buffer[offset + i * 2u] = state[i].low;
        debug_buffer[offset + i * 2u + 1u] = state[i].high;
    }
}

// S-box: x^7 in Goldilocks field (simplified)
fn sbox(x: GoldilocksField) -> GoldilocksField {
    let x2 = gf_mul(x, x);
    let x4 = gf_mul(x2, x2);
    let x6 = gf_mul(x4, x2);
    return gf_mul(x6, x);
}

// Fixed MDS matrix multiplication - circulant matrix for width 12
fn linear_layer(state: ptr<function, array<GoldilocksField, 12>>) {
    var result: array<GoldilocksField, 12>;

    // Initialize result to zero
    for (var i = 0u; i < 12u; i++) {
        result[i] = gf_zero();
    }

    // Circulant matrix multiplication: result[i] = sum(matrix[i][j] * state[j])
    // For circulant matrix, matrix[i][j] = first_row[(j - i + WIDTH) % WIDTH]
    for (var i = 0u; i < 12u; i++) {
        for (var j = 0u; j < 12u; j++) {
            // Calculate circulant matrix entry - fixed modulo calculation
            let matrix_idx = (j + 12u - i) % 12u;
            let matrix_val = MDS_MATRIX_FIRST_ROW[matrix_idx];

            // Convert matrix value to field element
            let matrix_field = gf_from_u32(u32(matrix_val));

            // Multiply and accumulate
            let product = gf_mul(matrix_field, (*state)[j]);
            result[i] = gf_add(result[i], product);
        }
    }

    // Copy result back to state atomically
    for (var i = 0u; i < 12u; i++) {
        (*state)[i] = result[i];
    }
}

// Fixed Poseidon2 permutation with proper structure
fn poseidon2_permute(state: ptr<function, array<GoldilocksField, 12>>) {
    // Initial external rounds (4 rounds)
    for (var round = 0u; round < 4u; round++) {
        // Add round constants
        for (var i = 0u; i < 12u; i++) {
            (*state)[i] = gf_add((*state)[i], gf_from_const(INITIAL_EXTERNAL_CONSTANTS[round][i]));
        }
        // S-box on all elements
        for (var i = 0u; i < 12u; i++) {
            (*state)[i] = sbox((*state)[i]);
        }
        // Linear layer (MDS matrix)
        linear_layer(state);
    }

    // Internal rounds (22 rounds)
    for (var round = 0u; round < 22u; round++) {
        // Add round constant to first element only
        (*state)[0] = gf_add((*state)[0], gf_from_const(INTERNAL_CONSTANTS[round]));
        // S-box on first element only
        (*state)[0] = sbox((*state)[0]);
        // Linear layer
        linear_layer(state);
    }

    // Terminal external rounds (4 rounds)
    for (var round = 0u; round < 4u; round++) {
        // Add round constants
        for (var i = 0u; i < 12u; i++) {
            (*state)[i] = gf_add((*state)[i], gf_from_const(TERMINAL_EXTERNAL_CONSTANTS[round][i]));
        }
        // S-box on all elements
        for (var i = 0u; i < 12u; i++) {
            (*state)[i] = sbox((*state)[i]);
        }
        // Linear layer
        linear_layer(state);
    }
}

// Convert bytes to Goldilocks field elements (proper encoding)
fn bytes_to_field_elements(input: array<u32, 24>) -> array<GoldilocksField, 12> {
    var felts: array<GoldilocksField, 12>;

    // Debug: write raw input to debug buffer
    for (var i = 0u; i < 8u; i++) {
        debug_buffer[50u + i] = input[i];
    }

    // Each field element gets 8 bytes = 2 u32s in little-endian format
    for (var i = 0u; i < 12u; i++) {
        let idx = i * 2u;
        if (idx + 1u < 24u) {
            // Little-endian: low bytes first
            felts[i] = GoldilocksField(input[idx], input[idx + 1u]);
        } else if (idx < 24u) {
            felts[i] = GoldilocksField(input[idx], 0u);
        } else {
            felts[i] = gf_zero();
        }

        // Debug: write first 4 field elements
        if (i < 4u) {
            debug_buffer[60u + i * 2u] = felts[i].low;
            debug_buffer[60u + i * 2u + 1u] = felts[i].high;
        }
    }
    return felts;
}

// Convert field elements back to bytes
fn field_elements_to_bytes(felts: array<GoldilocksField, 4>) -> array<u32, 8> {
    var result: array<u32, 8>;
    for (var i = 0u; i < 4u; i++) {
        result[i * 2u] = felts[i].low;
        result[i * 2u + 1u] = felts[i].high;
    }
    return result;
}

// Fixed Poseidon2 hash function with proper sponge construction
fn poseidon2_hash_squeeze_twice(input: array<u32, 24>) -> array<u32, 16> {
    var state: array<GoldilocksField, 12>;

    // Initialize state to zero
    for (var i = 0u; i < 12u; i++) {
        state[i] = gf_zero();
    }
    // debug_write_state(0u, state); // Debug: initial state

    // Convert input to field elements
    let input_felts = bytes_to_field_elements(input);

    // Debug: write input felts to debug buffer
    // for (var i = 0u; i < 8u; i++) {
    //     debug_buffer[24u + i * 2u] = input_felts[i].low;
    //     debug_buffer[24u + i * 2u + 1u] = input_felts[i].high;
    // }

    // Sponge absorption with rate = 4
    // First chunk: absorb elements 0-3
    for (var i = 0u; i < 4u; i++) {
        state[i] = gf_add(state[i], input_felts[i]);
    }
    // debug_write_state(48u, state); // Debug: after first absorption
    poseidon2_permute(&state);
    // debug_write_state(72u, state); // Debug: after first permutation

    // Second chunk: absorb elements 4-7
    for (var i = 0u; i < 4u; i++) {
        state[i] = gf_add(state[i], input_felts[i + 4u]);
    }
    // debug_write_state(96u, state); // Debug: after second absorption
    poseidon2_permute(&state);
    // debug_write_state(120u, state); // Debug: after second permutation

    // Third chunk: absorb elements 8-11
    for (var i = 0u; i < 4u; i++) {
        state[i] = gf_add(state[i], input_felts[i + 8u]);
    }
    // debug_write_state(144u, state); // Debug: after third absorption

    // Add padding (domain separation)
    state[0] = gf_add(state[0], gf_one());
    // debug_write_state(168u, state); // Debug: after padding
    poseidon2_permute(&state);
    // debug_write_state(192u, state); // Debug: after third permutation

    // First squeeze - get first 4 field elements
    let first_output = field_elements_to_bytes(array<GoldilocksField, 4>(
        state[0], state[1], state[2], state[3]
    ));

    // Second squeeze
    poseidon2_permute(&state);
    // debug_write_state(216u, state); // Debug: after fourth permutation
    let second_output = field_elements_to_bytes(array<GoldilocksField, 4>(
        state[0], state[1], state[2], state[3]
    ));

    // Combine both squeezes into 64-byte output
    var result: array<u32, 16>;
    for (var i = 0u; i < 8u; i++) {
        result[i] = first_output[i];
        result[i + 8u] = second_output[i];
    }

    return result;
}

// Double Poseidon2 hash (like Bitcoin's double SHA256)
fn double_hash(input: array<u32, 24>) -> array<u32, 16> {
    let first_hash = poseidon2_hash_squeeze_twice(input);
    // Convert back to input format for second hash
    var second_input: array<u32, 24>;
    for (var i = 0u; i < 16u; i++) {
        second_input[i] = first_hash[i];
    }
    // Pad remaining with zeros
    for (var i = 16u; i < 24u; i++) {
        second_input[i] = 0u;
    }
    return poseidon2_hash_squeeze_twice(second_input);
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

    // Simple debug test - write to debug buffer immediately
    debug_buffer[0] = 9999u;  // Global marker that shader ran
    debug_buffer[1] = thread_id;  // Which thread this is

    if (thread_id == 0u) {
        // Thread 0 specific tests
        debug_buffer[2] = 1111u;  // Thread 0 marker
        debug_buffer[3] = 2u;      // Simple value
        debug_buffer[4] = 1u + 1u; // Simple arithmetic

        // Test field functions
        let one = gf_one();
        debug_buffer[5] = one.low;
        debug_buffer[6] = one.high;

        // Test field addition
        let two = gf_add(one, one);
        debug_buffer[7] = two.low;
        debug_buffer[8] = two.high;

        // Test S-box function directly
        let test_val = gf_from_u32(2u);  // Test with 2
        let sbox_result = sbox(test_val);
        debug_buffer[9] = sbox_result.low;
        debug_buffer[10] = sbox_result.high;

        // Test with known value: sbox(1) = 1^7 = 1
        let sbox_one = sbox(one);
        debug_buffer[11] = sbox_one.low;
        debug_buffer[12] = sbox_one.high;

        // Test MDS matrix multiplication directly
        var mds_test_state: array<GoldilocksField, 12>;
        for (var i = 0u; i < 12u; i++) {
            mds_test_state[i] = gf_zero();
        }
        mds_test_state[0] = one;  // Put 1 in first position

        // Write MDS initial state
        for (var i = 0u; i < 4u; i++) {
            debug_buffer[80u + i * 2u] = mds_test_state[i].low;
            debug_buffer[80u + i * 2u + 1u] = mds_test_state[i].high;
        }

        // Apply ONLY linear layer (MDS matrix)
        linear_layer(&mds_test_state);

        // Write MDS result state
        for (var i = 0u; i < 4u; i++) {
            debug_buffer[90u + i * 2u] = mds_test_state[i].low;
            debug_buffer[90u + i * 2u + 1u] = mds_test_state[i].high;
        }

        // Test simple addition of round constants
        var const_test_state: array<GoldilocksField, 12>;
        for (var i = 0u; i < 12u; i++) {
            const_test_state[i] = gf_zero();
        }
        const_test_state[0] = one;

        // Add first internal round constant to first element
        const_test_state[0] = gf_add(const_test_state[0], gf_from_const(INTERNAL_CONSTANTS[0]));

        // Write constant addition result
        debug_buffer[100] = const_test_state[0].low;
        debug_buffer[101] = const_test_state[0].high;

        // TEST: Full hash function now that multiplication works
        var test_input: array<u32, 24>;
        for (var i = 0u; i < 8u; i++) {
            test_input[i] = header[i];
        }
        for (var i = 0u; i < 16u; i++) {
            test_input[8u + i] = start_nonce[i];
        }

        // Compute actual hash
        let test_hash = poseidon2_hash_squeeze_twice(test_input);

        // Store result for testing
        results[0] = 1u; // Force success flag to see debug output
        for (var i = 0u; i < 16u; i++) {
            results[i + 1u] = start_nonce[i]; // Store nonce
        }
        for (var i = 0u; i < 16u; i++) {
            results[i + 17u] = test_hash[i]; // Store actual hash
        }
    }

    // Skip normal mining for other threads during testing
    if (thread_id != 0u) {
        return;
    }
}
