// WGSL Goldilocks Field and Operations for Testing
// Using four u16 limbs for better overflow handling

// Goldilocks field element represented as [limb0, limb1, limb2, limb3]
// where the value is limb0 + limb1*2^16 + limb2*2^32 + limb3*2^48
struct GoldilocksField {
    limb0: u32,  // Actually u16, but WGSL doesn't have native u16
    limb1: u32,
    limb2: u32,
    limb3: u32,
}

// Struct to represent a pair of GoldilocksField values
struct GoldilocksFieldPair {
    first: GoldilocksField,
    second: GoldilocksField,
}

// Field modulus P = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
// In u16 limbs: P = [1, 0, 0, 0xFFFF]
const P_LIMB0: u32 = 1u;
const P_LIMB1: u32 = 0u;
const P_LIMB2: u32 = 0u;
const P_LIMB3: u32 = 0xFFFFu;

// EPSILON = 2^32 - 1 = 0xFFFFFFFF
// In u16 limbs: EPSILON = [0xFFFF, 0xFFFF, 0, 0]
const EPSILON_LIMB0: u32 = 0xFFFFu;
const EPSILON_LIMB1: u32 = 0xFFFFu;
const EPSILON_LIMB2: u32 = 0u;
const EPSILON_LIMB3: u32 = 0u;

// Helper to create field elements
fn gf_from_limbs(l0: u32, l1: u32, l2: u32, l3: u32) -> GoldilocksField {
    return GoldilocksField(l0 & 0xFFFFu, l1 & 0xFFFFu, l2 & 0xFFFFu, l3 & 0xFFFFu);
}

fn gf_zero() -> GoldilocksField {
    return gf_from_limbs(0u, 0u, 0u, 0u);
}

fn gf_one() -> GoldilocksField {
    return gf_from_limbs(1u, 0u, 0u, 0u);
}

// Convert a 64-bit value (as two u32s) to GoldilocksField
fn gf_from_u64_parts(low: u32, high: u32) -> GoldilocksField {
    return GoldilocksField(
        low & 0xFFFFu,         // bits 0-15
        (low >> 16u) & 0xFFFFu, // bits 16-31
        high & 0xFFFFu,        // bits 32-47
        (high >> 16u) & 0xFFFFu // bits 48-63
    );
}

// Convert GoldilocksField back to 64-bit (as two u32s) - for testing/debugging
fn gf_to_u64_parts(gf: GoldilocksField) -> vec2<u32> {
    let low = gf.limb0 | (gf.limb1 << 16u);
    let high = gf.limb2 | (gf.limb3 << 16u);
    return vec2<u32>(low, high);
}

// Addition with carry for u16 values (stored in u32)
// Returns vec2<u32>(sum, carry) where sum is masked to 16 bits
fn u16_add_with_carry(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum = a + b + carry_in;
    return vec2<u32>(sum & 0xFFFFu, sum >> 16u);
}

// Subtraction with borrow for u16 values (stored in u32)
// Returns vec2<u32>(diff, borrow) where diff is masked to 16 bits
fn u16_sub_with_borrow(a: u32, b: u32, borrow_in: u32) -> vec2<u32> {
    let temp = a + 0x10000u - b - borrow_in;  // Add 2^16 to avoid underflow
    return vec2<u32>(temp & 0xFFFFu, select(0u, 1u, temp < 0x10000u));
}

// Compare two GoldilocksField values
// Returns: 0 if a == b, 1 if a > b, -1 if a < b (but we'll use 2 for <)
fn gf_compare(a: GoldilocksField, b: GoldilocksField) -> u32 {
    // Compare from most significant limb to least
    if (a.limb3 != b.limb3) {
        return select(2u, 1u, a.limb3 > b.limb3);
    }
    if (a.limb2 != b.limb2) {
        return select(2u, 1u, a.limb2 > b.limb2);
    }
    if (a.limb1 != b.limb1) {
        return select(2u, 1u, a.limb1 > b.limb1);
    }
    if (a.limb0 != b.limb0) {
        return select(2u, 1u, a.limb0 > b.limb0);
    }
    return 0u; // Equal
}

// Goldilocks field addition
fn gf_add(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    // Add limb by limb with carry propagation
    let add0 = u16_add_with_carry(a.limb0, b.limb0, 0u);
    let add1 = u16_add_with_carry(a.limb1, b.limb1, add0.y);
    let add2 = u16_add_with_carry(a.limb2, b.limb2, add1.y);
    let add3 = u16_add_with_carry(a.limb3, b.limb3, add2.y);

    var result = GoldilocksField(add0.x, add1.x, add2.x, add3.x);

    // If there's overflow (carry out of most significant limb), add EPSILON manually
    if (add3.y != 0u) {
        // result += EPSILON = [0xFFFF, 0xFFFF, 0, 0]
        let eps_add0 = u16_add_with_carry(result.limb0, EPSILON_LIMB0, 0u);
        let eps_add1 = u16_add_with_carry(result.limb1, EPSILON_LIMB1, eps_add0.y);
        let eps_add2 = u16_add_with_carry(result.limb2, EPSILON_LIMB2, eps_add1.y);
        let eps_add3 = u16_add_with_carry(result.limb3, EPSILON_LIMB3, eps_add2.y);
        result = GoldilocksField(eps_add0.x, eps_add1.x, eps_add2.x, eps_add3.x);
    }

    // Reduce if result >= P manually
    let p = gf_from_limbs(P_LIMB0, P_LIMB1, P_LIMB2, P_LIMB3);
    if (gf_compare(result, p) != 2u) { // if result >= P
        // result -= P = [1, 0, 0, 0xFFFF]
        let p_sub0 = u16_sub_with_borrow(result.limb0, P_LIMB0, 0u);
        let p_sub1 = u16_sub_with_borrow(result.limb1, P_LIMB1, p_sub0.y);
        let p_sub2 = u16_sub_with_borrow(result.limb2, P_LIMB2, p_sub1.y);
        let p_sub3 = u16_sub_with_borrow(result.limb3, P_LIMB3, p_sub2.y);
        result = GoldilocksField(p_sub0.x, p_sub1.x, p_sub2.x, p_sub3.x);
    }

    return result;
}

// Goldilocks field subtraction
fn gf_sub(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    // Subtract limb by limb with borrow propagation
    let sub0 = u16_sub_with_borrow(a.limb0, b.limb0, 0u);
    let sub1 = u16_sub_with_borrow(a.limb1, b.limb1, sub0.y);
    let sub2 = u16_sub_with_borrow(a.limb2, b.limb2, sub1.y);
    let sub3 = u16_sub_with_borrow(a.limb3, b.limb3, sub2.y);

    var result = GoldilocksField(sub0.x, sub1.x, sub2.x, sub3.x);

    // If there's underflow (borrow out of most significant limb), subtract EPSILON manually
    if (sub3.y != 0u) {
        // result -= EPSILON = [0xFFFF, 0xFFFF, 0, 0]
        let eps_sub0 = u16_sub_with_borrow(result.limb0, EPSILON_LIMB0, 0u);
        let eps_sub1 = u16_sub_with_borrow(result.limb1, EPSILON_LIMB1, eps_sub0.y);
        let eps_sub2 = u16_sub_with_borrow(result.limb2, EPSILON_LIMB2, eps_sub1.y);
        let eps_sub3 = u16_sub_with_borrow(result.limb3, EPSILON_LIMB3, eps_sub2.y);
        result = GoldilocksField(eps_sub0.x, eps_sub1.x, eps_sub2.x, eps_sub3.x);
    }

    return result;
}

// Multiply two u16 values to get a u32 result
fn u16_mul(a: u32, b: u32) -> u32 {
    return (a & 0xFFFFu) * (b & 0xFFFFu);
}

// Simplified multiplication that handles carries more carefully
fn gf_mul_unreduced(a: GoldilocksField, b: GoldilocksField) -> array<u32, 8> {
    var result: array<u32, 8>;

    // Initialize all to zero
    for (var i = 0u; i < 8u; i++) {
        result[i] = 0u;
    }

    // Use double-precision arithmetic for each partial product
    // and add with proper carry propagation
    var temp: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) {
        temp[i] = 0u;
    }

    // a.limb0 * b (multiply a.limb0 by each limb of b)
    var carry = 0u;
    var prod = a.limb0 * b.limb0;
    temp[0] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb0 * b.limb1 + carry;
    temp[1] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb0 * b.limb2 + carry;
    temp[2] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb0 * b.limb3 + carry;
    temp[3] = prod & 0xFFFFu;
    temp[4] = prod >> 16u;

    // Add temp to result
    carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let sum = result[i] + temp[i] + carry;
        result[i] = sum & 0xFFFFu;
        carry = sum >> 16u;
    }

    // Clear temp and do a.limb1 * b << 16
    for (var i = 0u; i < 8u; i++) {
        temp[i] = 0u;
    }

    carry = 0u;
    prod = a.limb1 * b.limb0;
    temp[1] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb1 * b.limb1 + carry;
    temp[2] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb1 * b.limb2 + carry;
    temp[3] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb1 * b.limb3 + carry;
    temp[4] = prod & 0xFFFFu;
    temp[5] = prod >> 16u;

    // Add temp to result
    carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let sum = result[i] + temp[i] + carry;
        result[i] = sum & 0xFFFFu;
        carry = sum >> 16u;
    }

    // Clear temp and do a.limb2 * b << 32
    for (var i = 0u; i < 8u; i++) {
        temp[i] = 0u;
    }

    carry = 0u;
    prod = a.limb2 * b.limb0;
    temp[2] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb2 * b.limb1 + carry;
    temp[3] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb2 * b.limb2 + carry;
    temp[4] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb2 * b.limb3 + carry;
    temp[5] = prod & 0xFFFFu;
    temp[6] = prod >> 16u;

    // Add temp to result
    carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let sum = result[i] + temp[i] + carry;
        result[i] = sum & 0xFFFFu;
        carry = sum >> 16u;
    }

    // Clear temp and do a.limb3 * b << 48
    for (var i = 0u; i < 8u; i++) {
        temp[i] = 0u;
    }

    carry = 0u;
    prod = a.limb3 * b.limb0;
    temp[3] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb3 * b.limb1 + carry;
    temp[4] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb3 * b.limb2 + carry;
    temp[5] = prod & 0xFFFFu;
    carry = prod >> 16u;

    prod = a.limb3 * b.limb3 + carry;
    temp[6] = prod & 0xFFFFu;
    temp[7] = prod >> 16u;

    // Add temp to result
    carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let sum = result[i] + temp[i] + carry;
        result[i] = sum & 0xFFFFu;
        carry = sum >> 16u;
    }

    return result;
}

// Reduce an 8-limb number modulo the Goldilocks prime
// Based on the plonky2 reduce128 algorithm
fn gf_reduce_8limb(limbs: array<u32, 8>) -> GoldilocksField {
    // Convert 8 u16 limbs to representation similar to plonky2's u64 low/high
    // limbs[0..3] represent the low 64 bits
    // limbs[4..7] represent the high 64 bits

    let x_lo = gf_from_limbs(limbs[0], limbs[1], limbs[2], limbs[3]);
    let x_hi = gf_from_limbs(limbs[4], limbs[5], limbs[6], limbs[7]);

    // If high part is zero, just return low part
    if (limbs[4] == 0u && limbs[5] == 0u && limbs[6] == 0u && limbs[7] == 0u) {
        return x_lo;
    }

    // x_hi_hi = x_hi >> 32 (upper 32 bits of high part)
    let x_hi_hi = gf_from_limbs(limbs[6], limbs[7], 0u, 0u);

    // x_hi_lo = x_hi & EPSILON (lower 32 bits of high part)
    let x_hi_lo = gf_from_limbs(limbs[4], limbs[5], 0u, 0u);

    // t0 = x_lo - x_hi_hi (implement manually to avoid recursion)
    var t0 = x_lo;
    var borrow_occurred = false;

    // Check if x_lo < x_hi_hi
    if (gf_compare(x_lo, x_hi_hi) == 2u) {  // x_lo < x_hi_hi
        borrow_occurred = true;
        // Add P to x_lo before subtracting (equivalent to the borrow handling)
        // t0 += P = [1, 0, 0, 0xFFFF]
        let p_add0 = u16_add_with_carry(t0.limb0, P_LIMB0, 0u);
        let p_add1 = u16_add_with_carry(t0.limb1, P_LIMB1, p_add0.y);
        let p_add2 = u16_add_with_carry(t0.limb2, P_LIMB2, p_add1.y);
        let p_add3 = u16_add_with_carry(t0.limb3, P_LIMB3, p_add2.y);
        t0 = GoldilocksField(p_add0.x, p_add1.x, p_add2.x, p_add3.x);
    }
    // t0 -= x_hi_hi
    let t0_sub0 = u16_sub_with_borrow(t0.limb0, x_hi_hi.limb0, 0u);
    let t0_sub1 = u16_sub_with_borrow(t0.limb1, x_hi_hi.limb1, t0_sub0.y);
    let t0_sub2 = u16_sub_with_borrow(t0.limb2, x_hi_hi.limb2, t0_sub1.y);
    let t0_sub3 = u16_sub_with_borrow(t0.limb3, x_hi_hi.limb3, t0_sub2.y);
    t0 = GoldilocksField(t0_sub0.x, t0_sub1.x, t0_sub2.x, t0_sub3.x);

    // if borrow { t0 -= EPSILON; }
    if (borrow_occurred) {
        // t0 -= EPSILON = [0xFFFF, 0xFFFF, 0, 0]
        let eps_sub0 = u16_sub_with_borrow(t0.limb0, EPSILON_LIMB0, 0u);
        let eps_sub1 = u16_sub_with_borrow(t0.limb1, EPSILON_LIMB1, eps_sub0.y);
        let eps_sub2 = u16_sub_with_borrow(t0.limb2, EPSILON_LIMB2, eps_sub1.y);
        let eps_sub3 = u16_sub_with_borrow(t0.limb3, EPSILON_LIMB3, eps_sub2.y);
        t0 = GoldilocksField(eps_sub0.x, eps_sub1.x, eps_sub2.x, eps_sub3.x);
    }

    // t1 = x_hi_lo * EPSILON
    // EPSILON = [0xFFFF, 0xFFFF, 0, 0] = 0xFFFFFFFF
    // We can compute this carefully using the definition:
    // x_hi_lo * EPSILON = x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo

    // Since x_hi_lo has only lower 32 bits (limb0, limb1), we can compute this directly
    // Step 1: shift x_hi_lo left by 32 bits (2 limbs)
    let shifted = GoldilocksField(0u, 0u, x_hi_lo.limb0, x_hi_lo.limb1);

    // Step 2: subtract x_hi_lo from shifted result
    // This is guaranteed to fit in 64 bits since x_hi_lo < 2^32
    let t1_sub0 = u16_sub_with_borrow(shifted.limb0, x_hi_lo.limb0, 0u);
    let t1_sub1 = u16_sub_with_borrow(shifted.limb1, x_hi_lo.limb1, t1_sub0.y);
    let t1_sub2 = u16_sub_with_borrow(shifted.limb2, x_hi_lo.limb2, t1_sub1.y);
    let t1_sub3 = u16_sub_with_borrow(shifted.limb3, x_hi_lo.limb3, t1_sub2.y);
    let t1 = GoldilocksField(t1_sub0.x, t1_sub1.x, t1_sub2.x, t1_sub3.x);

    // t2 = t0 + t1 (implementing add_no_canonicalize_trashing_input from plonky2)
    // This function adds EPSILON * carry when there's an overflow
    let res_add0 = u16_add_with_carry(t0.limb0, t1.limb0, 0u);
    let res_add1 = u16_add_with_carry(t0.limb1, t1.limb1, res_add0.y);
    let res_add2 = u16_add_with_carry(t0.limb2, t1.limb2, res_add1.y);
    let res_add3 = u16_add_with_carry(t0.limb3, t1.limb3, res_add2.y);
    var result = GoldilocksField(res_add0.x, res_add1.x, res_add2.x, res_add3.x);

    // If there's a final carry, add EPSILON * carry (which is just EPSILON since carry is 1)
    if (res_add3.y != 0u) {
        let eps_add0 = u16_add_with_carry(result.limb0, EPSILON_LIMB0, 0u);
        let eps_add1 = u16_add_with_carry(result.limb1, EPSILON_LIMB1, eps_add0.y);
        let eps_add2 = u16_add_with_carry(result.limb2, EPSILON_LIMB2, eps_add1.y);
        let eps_add3 = u16_add_with_carry(result.limb3, EPSILON_LIMB3, eps_add2.y);
        result = GoldilocksField(eps_add0.x, eps_add1.x, eps_add2.x, eps_add3.x);
    }

    return result;
}

// Main Goldilocks field multiplication
fn gf_mul(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    // Handle special cases
    if (a.limb0 == 0u && a.limb1 == 0u && a.limb2 == 0u && a.limb3 == 0u) { return gf_zero(); }
    if (b.limb0 == 0u && b.limb1 == 0u && b.limb2 == 0u && b.limb3 == 0u) { return gf_zero(); }
    if (a.limb0 == 1u && a.limb1 == 0u && a.limb2 == 0u && a.limb3 == 0u) { return b; }
    if (b.limb0 == 1u && b.limb1 == 0u && b.limb2 == 0u && b.limb3 == 0u) { return a; }

    // General case: multiply and reduce
    let unreduced = gf_mul_unreduced(a, b);
    return gf_reduce_8limb(unreduced);
}

@group(0) @binding(0) var<storage, read> input_a: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read> input_b: array<GoldilocksField>;
@group(0) @binding(2) var<storage, read_write> output: array<GoldilocksField>;

@compute @workgroup_size(1)
fn gf_mul_test(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    output[i] = gf_mul(input_a[i], input_b[i]);
}
