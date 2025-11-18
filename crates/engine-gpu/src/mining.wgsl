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
    return gf_from_u64_parts(val[0], val[1]);
}

// MDS matrix constants for width 12 - circulant matrix first row
const MDS_MATRIX_FIRST_ROW: array<i32, 12> = array<i32, 12>(1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10);

// Storage buffers
@group(0) @binding(0) var<storage, read_write> results: array<u32>;
@group(0) @binding(1) var<storage, read> header: array<u32, 8>;     // 32 bytes
@group(0) @binding(2) var<storage, read> start_nonce: array<u32, 16>; // 64 bytes
@group(0) @binding(3) var<storage, read> difficulty_target: array<u32, 16>;    // 64 bytes (U512 target)
@group(0) @binding(4) var<storage, read_write> debug_buffer: array<u32>;      // Debug output buffer

// Goldilocks field element represented as [limb0, limb1, limb2, limb3]
// where the value is limb0 + limb1*2^16 + limb2*2^32 + limb3*2^48
struct GoldilocksField {
    limb0: u32,  // Actually u16, but WGSL doesn't have native u16
    limb1: u32,
    limb2: u32,
    limb3: u32,
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

fn gf_from_u32(val: u32) -> GoldilocksField {
    return GoldilocksField(val & 0xFFFFu, (val >> 16u) & 0xFFFFu, 0u, 0u);
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
// Returns: 0 if a == b, 1 if a > b, 2 if a < b
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
    // Debug: write inputs to buffer (convert to old format for debugging)
    debug_buffer[20] = a.limb0 | (a.limb1 << 16u);
    debug_buffer[21] = a.limb2 | (a.limb3 << 16u);
    debug_buffer[22] = b.limb0 | (b.limb1 << 16u);
    debug_buffer[23] = b.limb2 | (b.limb3 << 16u);

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

    // Debug: write results to buffer (convert to old format)
    debug_buffer[24] = result.limb0 | (result.limb1 << 16u);
    debug_buffer[25] = result.limb2 | (result.limb3 << 16u);
    debug_buffer[26] = add3.y;

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

// Proper Goldilocks field multiplication for all values
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

// Simple test function to verify basic field operations
fn test_field_operations() -> bool {
    // Write hardcoded values first to test if debug buffer works
    debug_buffer[1] = 123u;
    debug_buffer[2] = 456u;

    // Test gf_zero
    let zero = gf_zero();
    debug_buffer[3] = zero.limb0 | (zero.limb1 << 16u);
    debug_buffer[4] = zero.limb2 | (zero.limb3 << 16u);

    // Test gf_one - write immediately after creation
    let one = gf_one();
    debug_buffer[5] = one.limb0 | (one.limb1 << 16u);
    debug_buffer[6] = one.limb2 | (one.limb3 << 16u);

    // Hardcode test: create GoldilocksField(1,0,0,0) manually
    let manual_one = GoldilocksField(1u, 0u, 0u, 0u);
    debug_buffer[7] = manual_one.limb0 | (manual_one.limb1 << 16u);
    debug_buffer[8] = manual_one.limb2 | (manual_one.limb3 << 16u);

    // Test if functions can return correct values
    debug_buffer[11] = 777u;
    debug_buffer[12] = 888u;

    return true;
}

// Debug helper function to write state to debug buffer
fn debug_write_state(offset: u32, state: array<GoldilocksField, 12>) {
    for (var i = 0u; i < 12u; i++) {
        debug_buffer[offset + i * 2u] = state[i].limb0 | (state[i].limb1 << 16u);
        debug_buffer[offset + i * 2u + 1u] = state[i].limb2 | (state[i].limb3 << 16u);
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

// Convert bytes to Goldilocks field elements matching reference implementation
// Reference uses 4 bytes per field element with injective padding, creating 25 field elements from 96 bytes
fn bytes_to_field_elements(input: array<u32, 24>) -> array<GoldilocksField, 25> {
    var felts: array<GoldilocksField, 25>;

    // Convert u32 array to bytes (96 bytes total)
    var bytes: array<u32, 96>;  // Using u32s to store bytes for easier processing
    for (var i = 0u; i < 24u; i++) {
        let val = input[i];
        bytes[i * 4u + 0u] = val & 0xFFu;         // byte 0
        bytes[i * 4u + 1u] = (val >> 8u) & 0xFFu;  // byte 1
        bytes[i * 4u + 2u] = (val >> 16u) & 0xFFu; // byte 2
        bytes[i * 4u + 3u] = (val >> 24u) & 0xFFu; // byte 3
    }

    // Apply injective padding: add 1 byte, then pad with zeros to 4-byte alignment
    var padded_len = 96u + 1u; // 96 bytes + 1 marker byte = 97
    let padding_needed = (4u - (padded_len % 4u)) % 4u;
    padded_len = padded_len + padding_needed; // Should be 100 bytes (25 u32s worth)

    // Create padded byte array
    var padded_bytes: array<u32, 100>;
    for (var i = 0u; i < 96u; i++) {
        padded_bytes[i] = bytes[i];
    }
    padded_bytes[96] = 1u; // End marker
    for (var i = 97u; i < 100u; i++) {
        padded_bytes[i] = 0u; // Padding zeros
    }

    // Convert every 4 bytes to one field element (25 field elements total)
    for (var i = 0u; i < 25u; i++) {
        let byte_idx = i * 4u;
        // Create u32 from 4 bytes in little-endian order
        let val = padded_bytes[byte_idx] |
                 (padded_bytes[byte_idx + 1u] << 8u) |
                 (padded_bytes[byte_idx + 2u] << 16u) |
                 (padded_bytes[byte_idx + 3u] << 24u);
        felts[i] = gf_from_u32(val);
    }

    return felts;
}

// Convert field elements back to bytes
fn field_elements_to_bytes(felts: array<GoldilocksField, 4>) -> array<u32, 8> {
    var result: array<u32, 8>;
    for (var i = 0u; i < 4u; i++) {
        result[i * 2u] = felts[i].limb0 | (felts[i].limb1 << 16u);
        result[i * 2u + 1u] = felts[i].limb2 | (felts[i].limb3 << 16u);
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

    // Convert input to field elements (25 total)
    let input_felts = bytes_to_field_elements(input);

    // Debug: write input felts to debug buffer
    for (var i = 0u; i < 8u; i++) {
        debug_buffer[24u + i * 2u] = input_felts[i].limb0 | (input_felts[i].limb1 << 16u);
        debug_buffer[24u + i * 2u + 1u] = input_felts[i].limb2 | (input_felts[i].limb3 << 16u);
    }

    // Sponge construction matching CPU reference exactly:
    // CPU processes field elements using push_to_buf() which absorbs in chunks of RATE=4

    // Process first 24 elements (6 complete chunks of 4)
    for (var chunk = 0u; chunk < 6u; chunk++) {
        // Absorb 4 elements for this chunk
        for (var i = 0u; i < 4u; i++) {
            let felt_idx = chunk * 4u + i;
            state[i] = gf_add(state[i], input_felts[felt_idx]);
        }
        // Permute after each complete chunk
        poseidon2_permute(&state);

        // Debug first few chunks
        if (chunk == 0u) { debug_write_state(48u, state); }
        else if (chunk == 1u) { debug_write_state(72u, state); }
    }

    // Now we have element 24 (the padding marker = 1) remaining
    // This goes into buffer position 0, leaving buffer_len = 1
    state[0] = gf_add(state[0], input_felts[24u]); // Add the padding marker

    // CPU finalize_twice() adds another '1' (message-end marker)
    // This goes into buffer position 1, leaving buffer_len = 2
    state[1] = gf_add(state[1], gf_one()); // Add message-end '1'

    // CPU then zero-pads remaining buffer positions (positions 2,3 get zeros)
    // These are already zero in our state, so no action needed

    // Final permutation
    debug_write_state(96u, state); // Debug: after final absorption with padding
    poseidon2_permute(&state);
    debug_write_state(120u, state); // Debug: after final permutation

    // First squeeze - get first 4 field elements
    let first_output = field_elements_to_bytes(array<GoldilocksField, 4>(
        state[0], state[1], state[2], state[3]
    ));

    // Second squeeze
    poseidon2_permute(&state);
    debug_write_state(216u, state); // Debug: after fourth permutation
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
        debug_buffer[5] = one.limb0 | (one.limb1 << 16u);
        debug_buffer[6] = one.limb2 | (one.limb3 << 16u);

        // Test field addition
        let two = gf_add(one, one);
        debug_buffer[7] = two.limb0 | (two.limb1 << 16u);
        debug_buffer[8] = two.limb2 | (two.limb3 << 16u);

        // Test S-box function directly
        let test_val = gf_from_u32(2u);  // Test with 2
        let sbox_result = sbox(test_val);
        debug_buffer[9] = sbox_result.limb0 | (sbox_result.limb1 << 16u);
        debug_buffer[10] = sbox_result.limb2 | (sbox_result.limb3 << 16u);

        // Test with known value: sbox(1) = 1^7 = 1
        let sbox_one = sbox(one);
        debug_buffer[11] = sbox_one.limb0 | (sbox_one.limb1 << 16u);
        debug_buffer[12] = sbox_one.limb2 | (sbox_one.limb3 << 16u);

        // Test MDS matrix multiplication directly
        var mds_test_state: array<GoldilocksField, 12>;
        for (var i = 0u; i < 12u; i++) {
            mds_test_state[i] = gf_zero();
        }
        mds_test_state[0] = one;  // Put 1 in first position

        // Write MDS initial state
        for (var i = 0u; i < 4u; i++) {
            debug_buffer[80u + i * 2u] = mds_test_state[i].limb0 | (mds_test_state[i].limb1 << 16u);
            debug_buffer[80u + i * 2u + 1u] = mds_test_state[i].limb2 | (mds_test_state[i].limb3 << 16u);
        }

        // Apply ONLY linear layer (MDS matrix)
        linear_layer(&mds_test_state);

        // Write MDS result state
        for (var i = 0u; i < 4u; i++) {
            debug_buffer[90u + i * 2u] = mds_test_state[i].limb0 | (mds_test_state[i].limb1 << 16u);
            debug_buffer[90u + i * 2u + 1u] = mds_test_state[i].limb2 | (mds_test_state[i].limb3 << 16u);
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
        debug_buffer[100] = const_test_state[0].limb0 | (const_test_state[0].limb1 << 16u);
        debug_buffer[101] = const_test_state[0].limb2 | (const_test_state[0].limb3 << 16u);

        // TEST: Manual byte-to-field conversion verification
        // Test 1: Simple 4-byte input [1, 2, 3, 4]
        var simple_input: array<u32, 24>;
        for (var i = 0u; i < 24u; i++) {
            simple_input[i] = 0u;
        }
        simple_input[0] = 0x04030201u; // [1, 2, 3, 4] as little-endian u32

        let simple_felts = bytes_to_field_elements(simple_input);

        // Manual calculation: 4 bytes + 1 padding = 5 bytes, padded to 8 bytes = 2 field elements
        // Should be: [0x04030201, 0x00000001]
        debug_buffer[13] = simple_felts[0].limb0 | (simple_felts[0].limb1 << 16u); // Should be 0x04030201
        debug_buffer[14] = simple_felts[1].limb0 | (simple_felts[1].limb1 << 16u); // Should be 0x00000001

        // Test 2: All zeros (current test)
        var test_input: array<u32, 24>;
        for (var i = 0u; i < 8u; i++) {
            test_input[i] = header[i];
        }
        for (var i = 0u; i < 16u; i++) {
            test_input[8u + i] = start_nonce[i];
        }

        let test_felts = bytes_to_field_elements(test_input);

        // For 96 zero bytes: 96 + 1 = 97, padded to 100 = 25 field elements
        // First 24 should be zero, last should be 1
        debug_buffer[15] = test_felts[23].limb0 | (test_felts[23].limb1 << 16u); // Should be 0
        debug_buffer[16] = test_felts[24].limb0 | (test_felts[24].limb1 << 16u); // Should be 1

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
