// WGSL Goldilocks Field and Operations for Testing

// Goldilocks field element represented as [low_32, high_32]
struct GoldilocksField {
    low: u32,
    high: u32,
}

// Struct to represent a pair of GoldilocksField values (replaces vec2<GoldilocksField>)
struct GoldilocksFieldPair {
    first: GoldilocksField,
    second: GoldilocksField,
}

// Field modulus P = 2^64 - 2^32 + 1
// P_LOW = 1
// P_HIGH = 0xFFFFFFFF
const P_LOW: u32 = 1u;
const P_HIGH: u32 = 0xFFFFFFFFu;

// EPSILON = 2^32 - 1
const EPSILON: u32 = 0xFFFFFFFFu;

fn gf_zero() -> GoldilocksField {
    return GoldilocksField(0u, 0u);
}

fn gf_one() -> GoldilocksField {
    return GoldilocksField(1u, 0u);
}

// Helper for 64-bit addition with carry
// Returns vec2<u32>(sum, carry_out)
fn u32_add_with_carry(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum_temp = a + b;
    let carry_0 = select(0u, 1u, sum_temp < a); // Carry from a + b
    let sum = sum_temp + carry_in;
    let carry_1 = select(0u, 1u, sum < sum_temp); // Carry from (a+b) + carry_in
    return vec2<u32>(sum, carry_0 + carry_1);
}

// Helper for 64-bit subtraction with borrow
// Returns vec2<u32>(diff, borrow_out)
fn u32_sub_with_borrow(a: u32, b: u32, borrow_in: u32) -> vec2<u32> {
    let diff_temp = a - b;
    let borrow_0 = select(0u, 1u, diff_temp > a); // Borrow from a - b
    let diff = diff_temp - borrow_in;
    let borrow_1 = select(0u, 1u, diff > diff_temp); // Borrow from (a-b) - borrow_in
    return vec2<u32>(diff, borrow_0 + borrow_1);
}

// Multiplies two u32s to produce a 64-bit result (low, high)
fn u32_mul_extended(a: u32, b: u32) -> GoldilocksField {
    let a_low = a & 0xFFFFu;
    let a_high = a >> 16u;
    let b_low = b & 0xFFFFu;
    let b_high = b >> 16u;

    let p0 = a_low * b_low;
    let p1 = a_low * b_high;
    let p2 = a_high * b_low;
    let p3 = a_high * b_high;

    // Add carries
    let c1 = p0 >> 16u;
    let sum1 = p1 + p2 + c1;
    let c2 = sum1 >> 16u;

    let low = (sum1 << 16u) | (p0 & 0xFFFFu);
    let high = p3 + c2;

    return GoldilocksField(low, high);
}

// Multiplies two GoldilocksField values (representing 64-bit numbers).
// Returns a 128-bit result as (GoldilocksField_low_64_bits, GoldilocksField_high_64_bits).
fn gf_mul_u64_by_u64(a: GoldilocksField, b: GoldilocksField) -> GoldilocksFieldPair {
    let a0 = a.low;
    let a1 = a.high;
    let b0 = b.low;
    let b1 = b.high;

    let p0 = u32_mul_extended(a0, b0); // 64-bit: (p0.low, p0.high)
    let p1 = u32_mul_extended(a1, b0); // 64-bit: (p1.low, p1.high)
    let p2 = u32_mul_extended(a0, b1); // 64-bit: (p2.low, p2.high)
    let p3 = u32_mul_extended(a1, b1); // 64-bit: (p3.low, p3.high)

    // Combine the products:
    // Full product is p3 * 2^64 + (p1 + p2) * 2^32 + p0

    // Low 32 bits of the 128-bit result
    let res0 = p0.low;

    // Next 32 bits (from p0.high, p1.low, p2.low)
    let sum1_temp = u32_add_with_carry(p0.high, p1.low, 0u);
    let sum1_final = u32_add_with_carry(sum1_temp.x, p2.low, sum1_temp.y);
    let res1 = sum1_final.x;
    let carry_to_res2_from_sum1 = sum1_final.y;

    // Next 32 bits (from p1.high, p2.high, p3.low, carry_to_res2_from_sum1)
    let sum2_temp = u32_add_with_carry(p1.high, p2.high, 0u);
    let sum2_temp2 = u32_add_with_carry(sum2_temp.x, p3.low, sum2_temp.y);
    let sum2_final = u32_add_with_carry(sum2_temp2.x, carry_to_res2_from_sum1, sum2_temp2.y);
    let res2 = sum2_final.x;
    let carry_to_res3_from_sum2 = sum2_final.y;

    // High 32 bits (from p3.high, carry_to_res3_from_sum2)
    let res3 = p3.high + carry_to_res3_from_sum2; // This might overflow, but it's the highest part.

    let low_64 = GoldilocksField(res0, res1);
    let high_64 = GoldilocksField(res2, res3);

    return GoldilocksFieldPair(low_64, high_64);
}

// Reduces a 128-bit number (represented as two GoldilocksField values) modulo P.
// Based on plonky2's reduce128, adapted for WGSL u32.
fn gf_reduce128(val_low: GoldilocksField, val_high: GoldilocksField) -> GoldilocksField {
    // x_lo is (val_low.high:val_low.low)
    // x_hi is (val_high.high:val_high.low)

    // x_hi_hi = val_high.high
    // x_hi_lo = val_high.low

    // t0 = x_lo - x_hi_hi
    var t0_low = val_low.low;
    var t0_high = val_low.high;
    var borrow_from_t0 = 0u;

    // Subtract x_hi_hi (val_high.high) from x_lo (val_low.high:val_low.low)
    let sub_res_low = u32_sub_with_borrow(t0_low, val_high.high, 0u);
    t0_low = sub_res_low.x;
    borrow_from_t0 = sub_res_low.y; // Borrow from t0_high if t0_low < val_high.high

    let sub_res_high = u32_sub_with_borrow(t0_high, 0u, borrow_from_t0); // t0_high - borrow
    t0_high = sub_res_high.x;
    borrow_from_t0 = sub_res_high.y; // Borrow from beyond 64 bits

    // if borrow { t0 -= EPSILON; }
    if (borrow_from_t0 != 0u) {
        // This means t0 is negative. Add P to it.
        // P = (P_HIGH:P_LOW)
        let add_p_low = u32_add_with_carry(t0_low, P_LOW, 0u);
        t0_low = add_p_low.x;
        let add_p_high = u32_add_with_carry(t0_high, P_HIGH, add_p_low.y);
        t0_high = add_p_high.x;
    }

    // t1 = x_hi_lo * EPSILON
    // x_hi_lo is val_high.low
    // t1 = val_high.low * EPSILON
    // This is `val_high.low * (2^32 - 1) = (val_high.low << 32) - val_high.low`
    var t1_low: u32;
    var t1_high: u32;
    if (val_high.low == 0u) {
        t1_low = 0u;
        t1_high = 0u;
    } else {
        t1_low = 0u - val_high.low; // This wraps around, equivalent to (2^32 - val_high.low)
        t1_high = val_high.low - 1u;
    }
    let t1 = GoldilocksField(t1_low, t1_high);

    // t2 = t0 + t1
    var result_low = t0_low;
    var result_high = t0_high;

    let add_t1_low = u32_add_with_carry(result_low, t1.low, 0u);
    result_low = add_t1_low.x;
    let add_t1_high = u32_add_with_carry(result_high, t1.high, add_t1_low.y);
    result_high = add_t1_high.x;
    let final_carry = add_t1_high.y;

    var final_result = GoldilocksField(result_low, result_high);

    // If there's a final_carry, it means we effectively added 2^64, so add EPSILON.
    if (final_carry != 0u) {
        let add_epsilon_res = u32_add_with_carry(final_result.low, EPSILON, 0u);
        final_result.low = add_epsilon_res.x;
        final_result.high = final_result.high + add_epsilon_res.y;
    }

    // Final reduction: if result >= P, subtract P.
    if (final_result.high > P_HIGH || (final_result.high == P_HIGH && final_result.low >= P_LOW)) {
        let sub_p_res = u32_sub_with_borrow(final_result.low, P_LOW, 0u);
        final_result.low = sub_p_res.x;
        final_result.high = final_result.high - P_HIGH - sub_p_res.y;
    }
    return final_result;
}

// Proper Goldilocks field addition for all values
fn gf_add(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    var sum_low_res = u32_add_with_carry(a.low, b.low, 0u);
    var sum_high_res = u32_add_with_carry(a.high, b.high, sum_low_res.y);

    var result = GoldilocksField(sum_low_res.x, sum_high_res.x);
    var carry_high = sum_high_res.y;

    // If there's a carry_high, it means the sum is >= 2^64.
    // In Goldilocks field, 2^64 = 2^32 - 1 (mod P).
    // So if carry_high is 1, we effectively add EPSILON (2^32 - 1) to the result.
    if (carry_high != 0u) {
        let add_epsilon_res = u32_add_with_carry(result.low, EPSILON, 0u);
        result.low = add_epsilon_res.x;
        result.high = result.high + add_epsilon_res.y; // This could cause another carry
    }

    // Final reduction: if result >= P, subtract P.
    if (result.high > P_HIGH || (result.high == P_HIGH && result.low >= P_LOW)) {
        let sub_p_res = u32_sub_with_borrow(result.low, P_LOW, 0u);
        result.low = sub_p_res.x;
        result.high = result.high - P_HIGH - sub_p_res.y;
    }
    return result;
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
        let product_64 = u32_mul_extended(a.low, b.low);
        return gf_reduce128(product_64, gf_zero());
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

    // Step 3: General case (both a.high != 0u and b.high != 0u)
    let product_128 = gf_mul_u64_by_u64(a, b);
    return gf_reduce128(product_128.first, product_128.second);
}

@group(0) @binding(0) var<storage, read> input_a: array<GoldilocksField>;
@group(0) @binding(1) var<storage, read> input_b: array<GoldilocksField>;
@group(0) @binding(2) var<storage, read_write> output: array<GoldilocksField>;

@compute @workgroup_size(1)
fn gf_mul_test(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    output[i] = gf_mul(input_a[i], input_b[i]);
}
