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

// Individual constants for testing
const INTERNAL_CONST_0_LOW: u32 = 2018170979u;
const INTERNAL_CONST_0_HIGH: u32 = 2549578122u;
const INTERNAL_CONST_1_LOW: u32 = 794875120u;
const INTERNAL_CONST_1_HIGH: u32 = 3520249608u;

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
        array<u32, 2>(1426707734u, 655469195u),
        array<u32, 2>(3385403543u, 1256631504u),
        array<u32, 2>(1381422714u, 1458257259u),
        array<u32, 2>(2453402910u, 528129365u),
        array<u32, 2>(964329320u, 905986685u),
        array<u32, 2>(1534247888u, 3842469367u),
        array<u32, 2>(744525997u, 4241857185u),
        array<u32, 2>(1756723870u, 3448331916u),
        array<u32, 2>(3610291774u, 1105166073u),
        array<u32, 2>(2596181885u, 3997051784u),
        array<u32, 2>(3199845381u, 3533420525u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(4127777666u, 2103183598u),
        array<u32, 2>(2867888172u, 2309916828u),
        array<u32, 2>(1831532055u, 3009056407u),
        array<u32, 2>(2947502451u, 3675530062u),
        array<u32, 2>(3565886616u, 2029012066u),
        array<u32, 2>(3833391242u, 642945968u),
        array<u32, 2>(1773785903u, 2577032347u),
        array<u32, 2>(1770914259u, 1689297286u),
        array<u32, 2>(3752758200u, 3993707216u),
        array<u32, 2>(3389302766u, 1339375184u),
        array<u32, 2>(2180141127u, 1466089441u),
        array<u32, 2>(3199591357u, 4111832034u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(1625498743u, 509487959u),
        array<u32, 2>(4188712685u, 1646551713u),
        array<u32, 2>(3451003566u, 2854767422u),
        array<u32, 2>(1412166652u, 1674110767u),
        array<u32, 2>(3410212320u, 1000704202u),
        array<u32, 2>(3381743837u, 602777331u),
        array<u32, 2>(3131873882u, 2866003620u),
        array<u32, 2>(2610174026u, 3923414377u),
        array<u32, 2>(3644719692u, 3450945356u),
        array<u32, 2>(1458984419u, 2418851081u),
        array<u32, 2>(3344519983u, 1531855103u),
        array<u32, 2>(2721413879u, 3732495392u)
    )
);

// Terminal external round constants (4 rounds x 12 elements)
const TERMINAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(
    array<array<u32, 2>, 12>(
        array<u32, 2>(3773451374u, 2071119161u),
        array<u32, 2>(3805190518u, 340095962u),
        array<u32, 2>(2402679944u, 2149591222u),
        array<u32, 2>(743434178u, 1832305922u),
        array<u32, 2>(2847530739u, 2718290175u),
        array<u32, 2>(514243119u, 4142392203u),
        array<u32, 2>(3844443492u, 888639642u),
        array<u32, 2>(2008645578u, 2957397405u),
        array<u32, 2>(3732799654u, 1692252629u),
        array<u32, 2>(664231319u, 248567644u),
        array<u32, 2>(287781771u, 482031345u),
        array<u32, 2>(3486561978u, 1718871301u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(814165505u, 1616875560u),
        array<u32, 2>(2123759183u, 1070984082u),
        array<u32, 2>(2722916813u, 3893372341u),
        array<u32, 2>(3726899022u, 4157656693u),
        array<u32, 2>(2824360073u, 4086907574u),
        array<u32, 2>(4155973110u, 1837140488u),
        array<u32, 2>(2297731723u, 4169165669u),
        array<u32, 2>(707924090u, 1474243980u),
        array<u32, 2>(1298483757u, 384287239u),
        array<u32, 2>(4243798069u, 557703745u),
        array<u32, 2>(1510569718u, 2968696976u),
        array<u32, 2>(3174388759u, 3638808363u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(2044277519u, 3835193622u),
        array<u32, 2>(2743212840u, 1983595986u),
        array<u32, 2>(3943309069u, 151568229u),
        array<u32, 2>(424355060u, 1989274413u),
        array<u32, 2>(867046322u, 239293714u),
        array<u32, 2>(4230997871u, 2479068123u),
        array<u32, 2>(1565052394u, 2566260552u),
        array<u32, 2>(815274432u, 3822673712u),
        array<u32, 2>(1051683535u, 519405993u),
        array<u32, 2>(2687564964u, 186958263u),
        array<u32, 2>(1450226471u, 1648586942u),
        array<u32, 2>(1511122054u, 1595811937u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(2426274859u, 4261676319u),
        array<u32, 2>(1207777283u, 1918363057u),
        array<u32, 2>(3090099832u, 3870167883u),
        array<u32, 2>(4101522705u, 1460796764u),
        array<u32, 2>(201900220u, 4164567654u),
        array<u32, 2>(2587682901u, 752404845u),
        array<u32, 2>(2967564913u, 2100296475u),
        array<u32, 2>(3404347409u, 2242778408u),
        array<u32, 2>(3350048952u, 1386431957u),
        array<u32, 2>(4093308564u, 1347177553u),
        array<u32, 2>(2633812729u, 3169012324u),
        array<u32, 2>(1727753673u, 3768793234u)
    )
);

// Helper function to create GoldilocksField from constant array
fn gf_from_const(val: array<u32, 2>) -> GoldilocksField {
    return gf_from_u64_parts(val[0], val[1]);
}

// MDS matrix constants for width 12 - circulant matrix first row
// MDS matrix diagonal for width 12 Goldilocks (from p3_goldilocks constants)
// Each element is stored as [low32, high32] pairs
const MDS_MATRIX_DIAG_12: array<array<u32, 2>, 12> = array<array<u32, 2>, 12>(
    array<u32, 2>(0x23ba9300u, 0xc3b6c08eu), // 0xc3b6c08e23ba9300u
    array<u32, 2>(0x4a324fb6u, 0xd84b5de9u), // 0xd84b5de94a324fb6u
    array<u32, 2>(0x5b35b84fu, 0x0d0c371cu), // 0x0d0c371c5b35b84fu
    array<u32, 2>(0xe7188037u, 0x7964f570u), // 0x7964f570e7188037u
    array<u32, 2>(0xd996604bu, 0x5daf18bbu), // 0x5daf18bbd996604bu
    array<u32, 2>(0xb9595257u, 0x6743bc47u), // 0x6743bc47b9595257u
    array<u32, 2>(0x2c59bb70u, 0x5528b936u), // 0x5528b9362c59bb70u
    array<u32, 2>(0x7127b68bu, 0xac45e25bu), // 0xac45e25b7127b68bu
    array<u32, 2>(0xfbb606b5u, 0xa2077d7du), // 0xa2077d7dfbb606b5u
    array<u32, 2>(0xaee378aeu, 0xf3faac6fu), // 0xf3faac6faee378aeu
    array<u32, 2>(0x1545e883u, 0x0c6388b5u), // 0x0c6388b51545e883u
    array<u32, 2>(0x44917b60u, 0xd27dbb69u)  // 0xd27dbb6944917b60u
);

// Storage buffers
@group(0) @binding(0) var<storage, read_write> results: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> header: array<u32, 8>;     // 32 bytes
@group(0) @binding(2) var<storage, read> start_nonce: array<u32, 16>; // 64 bytes
@group(0) @binding(3) var<storage, read> difficulty_target: array<u32, 16>;    // 64 bytes (U512 target)
@group(0) @binding(4) var<storage, read> dispatch_config: array<u32, 4>; // [total_threads (logical), nonces_per_thread, total_nonces (logical nonces in this dispatch), threads_per_workgroup]

// Goldilocks field element represented as [limb0, limb1]
// where the value is limb0 + limb1*2^32
struct GoldilocksField {
    limb0: u32,
    limb1: u32,
}

// Field modulus P = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
// In u32 limbs: P = [1, 0xFFFFFFFF]
const P_LIMB0: u32 = 1u;
const P_LIMB1: u32 = 0xFFFFFFFFu;

// EPSILON = 2^32 - 1 = 0x00000000FFFFFFFF
// In u32 limbs: EPSILON = [0xFFFFFFFF, 0]
const EPSILON_LIMB0: u32 = 0xFFFFFFFFu;
const EPSILON_LIMB1: u32 = 0u;

// Helper to create field elements
fn gf_from_limbs(l0: u32, l1: u32) -> GoldilocksField {
    return GoldilocksField(l0, l1);
}

fn gf_zero() -> GoldilocksField {
    return gf_from_limbs(0u, 0u);
}

fn gf_one() -> GoldilocksField {
    return gf_from_limbs(1u, 0u);
}

fn gf_from_u32(val: u32) -> GoldilocksField {
    return GoldilocksField(val, 0u);
}

// Convert a 64-bit value (as two u32s) to GoldilocksField
fn gf_from_u64_parts(low: u32, high: u32) -> GoldilocksField {
    var result = GoldilocksField(low, high);

    // Reduce if >= P
    if (gf_compare(result, gf_from_limbs(P_LIMB0, P_LIMB1)) != 2u) {
        result = gf_sub(result, gf_from_limbs(P_LIMB0, P_LIMB1));
    }

    return result;
}

// Addition with carry for u32 values
// Returns vec2<u32>(sum, carry)
fn u32_add_with_carry(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum1 = a + b;
    let c1 = select(0u, 1u, sum1 < a);
    let sum2 = sum1 + carry_in;
    let c2 = select(0u, 1u, sum2 < sum1);
    return vec2<u32>(sum2, c1 + c2);
}

// Subtraction with borrow for u32 values
// Returns vec2<u32>(diff, borrow)
fn u32_sub_with_borrow(a: u32, b: u32, borrow_in: u32) -> vec2<u32> {
    let diff1 = a - b;
    let b1 = select(0u, 1u, diff1 > a); // if a - b > a, then underflow happened (since b > 0)
    // Wait, if a < b, a - b wraps around to a large number.
    // e.g. 2 - 3 = 0xFFFFFFFF. 0xFFFFFFFF > 2. Correct.

    let diff2 = diff1 - borrow_in;
    let b2 = select(0u, 1u, diff2 > diff1);

    return vec2<u32>(diff2, b1 + b2);
}

// Multiply two u32 values to get a u64 result (as vec2<u32>)
fn u32_mul_to_u64(a: u32, b: u32) -> vec2<u32> {
    // Use unpack2x16u to split 32-bit values into 16-bit halves efficiently
    let a_parts = unpack2x16u(a);
    let a_lo = a_parts.x;
    let a_hi = a_parts.y;
    
    let b_parts = unpack2x16u(b);
    let b_lo = b_parts.x;
    let b_hi = b_parts.y;

    let p0 = a_lo * b_lo;
    let p1 = a_hi * b_lo;
    let p2 = a_lo * b_hi;
    let p3 = a_hi * b_hi;

    let sum_mid_part = p1 + p2;
    let carry_mid = select(0u, 1u, sum_mid_part < p1);

    let term_mid_lo = sum_mid_part << 16u;
    let term_mid_hi = (sum_mid_part >> 16u) | (carry_mid << 16u);

    let res_lo = p0 + term_mid_lo;
    let carry_lo = select(0u, 1u, res_lo < p0);

    let res_hi = p3 + term_mid_hi + carry_lo;

    return vec2<u32>(res_lo, res_hi);
}

// Compare two GoldilocksField values
// Returns: 0 if a == b, 1 if a > b, 2 if a < b
fn gf_compare(a: GoldilocksField, b: GoldilocksField) -> u32 {
    // Compare from most significant limb to least
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
    let add0 = u32_add_with_carry(a.limb0, b.limb0, 0u);
    let add1 = u32_add_with_carry(a.limb1, b.limb1, add0.y);

    var result = GoldilocksField(add0.x, add1.x);

    // Handle overflow: if carry out, we've computed a + b = result + 2^64
    // In Goldilocks: 2^64 ≡ 2^32 - 1 (mod P)
    if (add1.y != 0u) {
        // Add EPSILON = 0xFFFFFFFF00000000 (Wait, EPSILON is 2^32 - 1)
        // EPSILON = 0x00000000FFFFFFFF
        // EPSILON_LIMB0 = 0xFFFFFFFF, EPSILON_LIMB1 = 0
        let eps_add0 = u32_add_with_carry(result.limb0, EPSILON_LIMB0, 0u);
        let eps_add1 = u32_add_with_carry(result.limb1, EPSILON_LIMB1, eps_add0.y);
        result = GoldilocksField(eps_add0.x, eps_add1.x);

        // If adding EPSILON caused another overflow, add EPSILON again
        if (eps_add1.y != 0u) {
            let eps2_add0 = u32_add_with_carry(result.limb0, EPSILON_LIMB0, 0u);
            let eps2_add1 = u32_add_with_carry(result.limb1, EPSILON_LIMB1, eps2_add0.y);
            result = GoldilocksField(eps2_add0.x, eps2_add1.x);
        }
    }

    // Final reduction if result >= P
    let p = gf_from_limbs(P_LIMB0, P_LIMB1);
    if (gf_compare(result, p) != 2u) { // if result >= P
        result = gf_sub_no_underflow(result, p);
    }

    return result;
}

// Helper for subtraction without underflow (assumes a >= b)
fn gf_sub_no_underflow(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    let sub0 = u32_sub_with_borrow(a.limb0, b.limb0, 0u);
    let sub1 = u32_sub_with_borrow(a.limb1, b.limb1, sub0.y);
    return GoldilocksField(sub0.x, sub1.x);
}

// Goldilocks field subtraction
fn gf_sub(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    // If a >= b, do direct subtraction
    if (gf_compare(a, b) != 2u) {
        return gf_sub_no_underflow(a, b);
    }

    // Otherwise: a < b, so compute a - b + P
    let p = gf_from_limbs(P_LIMB0, P_LIMB1);
    let a_plus_p = gf_add_no_reduction(a, p);
    return gf_sub_no_underflow(a_plus_p, b);
}

// Addition without final modular reduction (used internally)
fn gf_add_no_reduction(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    let add0 = u32_add_with_carry(a.limb0, b.limb0, 0u);
    let add1 = u32_add_with_carry(a.limb1, b.limb1, add0.y);
    return GoldilocksField(add0.x, add1.x);
}

// Simplified multiplication that handles carries more carefully
fn gf_mul_unreduced(a: GoldilocksField, b: GoldilocksField) -> array<u32, 4> {
    // p0 = a0 * b0 (64 bits)
    let p0 = u32_mul_to_u64(a.limb0, b.limb0);

    // p1 = a0 * b1 (64 bits)
    let p1 = u32_mul_to_u64(a.limb0, b.limb1);

    // p2 = a1 * b0 (64 bits)
    let p2 = u32_mul_to_u64(a.limb1, b.limb0);

    // p3 = a1 * b1 (64 bits)
    let p3 = u32_mul_to_u64(a.limb1, b.limb1);

    // result[0] = p0.x
    // result[1] = p0.y + p1.x + p2.x + carry
    // result[2] = p1.y + p2.y + p3.x + carry
    // result[3] = p3.y + carry

    var r0 = p0.x;

    let sum1 = u32_add_with_carry(p0.y, p1.x, 0u);
    let sum2 = u32_add_with_carry(sum1.x, p2.x, 0u);
    var r1 = sum2.x;
    var c1 = sum1.y + sum2.y; // carry to next limb

    let sum3 = u32_add_with_carry(p1.y, p2.y, c1);
    let sum4 = u32_add_with_carry(sum3.x, p3.x, 0u);
    var r2 = sum4.x;
    var c2 = sum3.y + sum4.y;

    let sum5 = u32_add_with_carry(p3.y, 0u, c2);
    var r3 = sum5.x;

    return array<u32, 4>(r0, r1, r2, r3);
}

// Reduce a 4-limb number modulo the Goldilocks prime
fn gf_reduce_4limb(limbs: array<u32, 4>) -> GoldilocksField {
    // x_lo = limbs[0], limbs[1]
    let x_lo = gf_from_limbs(limbs[0], limbs[1]);

    // x_hi = limbs[2], limbs[3]
    // x_hi_hi = limbs[3] (upper 32 bits of x_hi)
    let x_hi_hi = gf_from_limbs(limbs[3], 0u);

    // x_hi_lo = limbs[2] (lower 32 bits of x_hi)
    let x_hi_lo = gf_from_limbs(limbs[2], 0u);

    // Step 1: t0 = x_lo - x_hi_hi (with underflow detection)
    var t0: GoldilocksField;
    var underflow = false;

    let sub0 = u32_sub_with_borrow(x_lo.limb0, x_hi_hi.limb0, 0u);
    let sub1 = u32_sub_with_borrow(x_lo.limb1, x_hi_hi.limb1, sub0.y);
    t0 = GoldilocksField(sub0.x, sub1.x);

    if (sub1.y != 0u) {
        underflow = true;
    }

    // Step 2: if underflow { t0 -= NEG_ORDER; }
    if (underflow) {
        let eps_sub0 = u32_sub_with_borrow(t0.limb0, EPSILON_LIMB0, 0u);
        let eps_sub1 = u32_sub_with_borrow(t0.limb1, EPSILON_LIMB1, eps_sub0.y);
        t0 = GoldilocksField(eps_sub0.x, eps_sub1.x);
    }

    // Step 3: t1 = x_hi_lo * NEG_ORDER
    // NEG_ORDER = 2^32 - 1
    // x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo
    // x_hi_lo is [limbs[2], 0]
    // x_hi_lo << 32 is [0, limbs[2]]

    let shifted = GoldilocksField(0u, x_hi_lo.limb0);
    let t1_sub0 = u32_sub_with_borrow(shifted.limb0, x_hi_lo.limb0, 0u);
    let t1_sub1 = u32_sub_with_borrow(shifted.limb1, x_hi_lo.limb1, t1_sub0.y);
    let t1 = GoldilocksField(t1_sub0.x, t1_sub1.x);

    // Step 4: result = t0 + t1 (with overflow handling like CPU)
    let add0 = u32_add_with_carry(t0.limb0, t1.limb0, 0u);
    let add1 = u32_add_with_carry(t0.limb1, t1.limb1, add0.y);
    var result = GoldilocksField(add0.x, add1.x);

    // If overflow, add NEG_ORDER
    if (add1.y != 0u) {
        let eps_add0 = u32_add_with_carry(result.limb0, EPSILON_LIMB0, 0u);
        let eps_add1 = u32_add_with_carry(result.limb1, EPSILON_LIMB1, eps_add0.y);
        result = GoldilocksField(eps_add0.x, eps_add1.x);
    }

    return result;
}

// Main Goldilocks field multiplication
fn gf_mul(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    // Removed branching checks for 0 and 1 to improve GPU uniformity.
    // The general math handles 0 and 1 correctly.
    
    // General case: multiply and reduce
    let unreduced = gf_mul_unreduced(a, b);
    return gf_reduce_4limb(unreduced);
}

// S-box: x^7 in Goldilocks field (efficient approach)
fn sbox(x: GoldilocksField) -> GoldilocksField {
    let x2 = gf_mul(x, x);
    let x4 = gf_mul(x2, x2);
    let x6 = gf_mul(x4, x2);
    return gf_mul(x6, x);
}

// External linear layer for width 12 using correct 4x4 MDS matrix
// Standard MDSMat4: [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
fn external_linear_layer(state: ptr<function, array<GoldilocksField, 12>>) {
    // First apply the 4x4 MDS matrix to each consecutive 4 elements
    for (var chunk = 0u; chunk < 3u; chunk++) {
        let offset = chunk * 4u;
        var x: array<GoldilocksField, 4>;

        // Copy chunk
        for (var i = 0u; i < 4u; i++) {
            x[i] = (*state)[offset + i];
        }

        // Optimized 4x4 MDS application using only additions and doubles,
        // matching apply_external_linear_layer_to_chunk in the CPU implementation.
        let t01 = gf_add(x[0], x[1]);
        let t23 = gf_add(x[2], x[3]);
        let t0123 = gf_add(t01, t23);
        let t01123 = gf_add(t0123, x[1]);
        let t01233 = gf_add(t0123, x[3]);

        let two_x0 = gf_add(x[0], x[0]);
        let two_x2 = gf_add(x[2], x[2]);

        // The order of updates matches the reference algorithm:
        // x[3] = t01233 + 2*x[0]
        // x[1] = t01123 + 2*x[2]
        // x[0] = t01123 + t01
        // x[2] = t01233 + t23
        let new_3 = gf_add(t01233, two_x0);
        let new_1 = gf_add(t01123, two_x2);
        let new_0 = gf_add(t01123, t01);
        let new_2 = gf_add(t01233, t23);

        // Copy back
        (*state)[offset + 0u] = new_0;
        (*state)[offset + 1u] = new_1;
        (*state)[offset + 2u] = new_2;
        (*state)[offset + 3u] = new_3;
    }

    // Now apply the circulant matrix part
    // Precompute the four sums of every four elements
    var sums: array<GoldilocksField, 4>;
    for (var k = 0u; k < 4u; k++) {
        sums[k] = gf_zero();
        for (var j = k; j < 12u; j += 4u) {
            sums[k] = gf_add(sums[k], (*state)[j]);
        }
    }

    // Add the appropriate sum to each element
    for (var i = 0u; i < 12u; i++) {
        (*state)[i] = gf_add((*state)[i], sums[i % 4u]);
    }
}

// Internal linear layer using diagonal matrix for width 12
fn internal_linear_layer(state: ptr<function, array<GoldilocksField, 12>>) {
    var result: array<GoldilocksField, 12>;

    // Sum all elements
    var sum = gf_zero();
    for (var i = 0u; i < 12u; i++) {
        sum = gf_add(sum, (*state)[i]);
    }

    // Apply diagonal matrix: result[i] = state[i] * diag[i] + sum
    for (var i = 0u; i < 12u; i++) {
        let diag_val = gf_from_u64_parts(
            MDS_MATRIX_DIAG_12[i][0],  // low 32 bits
            MDS_MATRIX_DIAG_12[i][1]   // high 32 bits
        );
        let scaled = gf_mul((*state)[i], diag_val);
        result[i] = gf_add(scaled, sum);
    }

    // Copy result back to state
    for (var i = 0u; i < 12u; i++) {
        (*state)[i] = result[i];
    }
}

// Fixed Poseidon2 permutation with proper structure
fn poseidon2_permute(state: ptr<function, array<GoldilocksField, 12>>) {

    // Initial MDS permutation (required by p3-poseidon2 spec)
    external_linear_layer(state);

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

        // External linear layer (4x4 MDS matrix)
        external_linear_layer(state);

    }

    // Internal rounds (22 rounds)
    for (var round = 0u; round < 22u; round++) {
        // Add round constant to first element only
        (*state)[0] = gf_add((*state)[0], gf_from_const(INTERNAL_CONSTANTS[round]));
        // S-box on first element only
        (*state)[0] = sbox((*state)[0]);
        // Internal linear layer (diagonal matrix)
        internal_linear_layer(state);
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
        // External linear layer (4x4 MDS matrix)
        external_linear_layer(state);
    }
}

// Convert bytes to Goldilocks field elements matching reference implementation
// Reference uses 4 bytes per field element with injective padding, creating 25 field elements from 96 bytes
fn bytes_to_field_elements(input: array<u32, 24>) -> array<GoldilocksField, 25> {
    var felts: array<GoldilocksField, 25>;

    // Optimized conversion: The input is already aligned as u32s.
    // Bytes 0..95 correspond exactly to input[0]..input[23].
    for (var i = 0u; i < 24u; i++) {
        felts[i] = gf_from_u32(input[i]);
    }

    // Apply injective padding:
    // Byte 96 is the start of the 25th element (index 24).
    // The padding byte is 0x01, followed by 0x00s.
    // So the 25th u32 is 0x00000001 (Little Endian).
    felts[24] = gf_from_u32(1u);

    return felts;
}

// Convert field elements back to bytes
fn field_elements_to_bytes(felts: array<GoldilocksField, 4>) -> array<u32, 8> {
    var result: array<u32, 8>;
    for (var i = 0u; i < 4u; i++) {
        result[i * 2u] = felts[i].limb0;
        result[i * 2u + 1u] = felts[i].limb1;
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

    // Convert input to field elements (25 total)
    let input_felts = bytes_to_field_elements(input);

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
    }

    // Process remaining 1 element (partial chunk)
    // Now we have element 24 (the padding marker = 1) remaining
    // This simulates CPU's finalize_twice adding ONE to buffer position 0
    state[0] = gf_add(state[0], input_felts[24u]); // Add the padding marker (should be 1)

    // Add sponge padding marker (ONE) to the next position
    state[1] = gf_add(state[1], gf_one());

    // Final permutation (CPU calls permute after completing the block)
    poseidon2_permute(&state);

    // First squeeze - get first 4 field elements
    let first_output = field_elements_to_bytes(array<GoldilocksField, 4>(
        state[0], state[1], state[2], state[3]
    ));

    // Second squeeze
    poseidon2_permute(&state);

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

// Poseidon2 hash function for 64-byte input (used in double hash)
fn poseidon2_hash_squeeze_twice_64(input: array<u32, 16>) -> array<u32, 16> {
    var state: array<GoldilocksField, 12>;

    // Initialize state to zero
    for (var i = 0u; i < 12u; i++) {
        state[i] = gf_zero();
    }

    // Convert input to field elements
    // 64 bytes = 16 u32s = 16 felts
    // Padding adds 1 byte + 3 zeros = 4 bytes = 1 felt (value 1)
    // Total 17 felts

    // Process first 16 elements (4 complete chunks of 4)
    for (var chunk = 0u; chunk < 4u; chunk++) {
        // Absorb 4 elements for this chunk
        for (var i = 0u; i < 4u; i++) {
            let felt_idx = chunk * 4u + i;
            state[i] = gf_add(state[i], gf_from_u32(input[felt_idx]));
        }
        // Permute after each complete chunk
        poseidon2_permute(&state);
    }

    // Handle the last felt (padding marker = 1) and sponge padding
    // The last felt is 1 (from bytes 1, 0, 0, 0)
    state[0] = gf_add(state[0], gf_one());

    // Add sponge padding marker (ONE) to the next position
    state[1] = gf_add(state[1], gf_one());

    // Final permutation
    poseidon2_permute(&state);

    // First squeeze
    let first_output = field_elements_to_bytes(array<GoldilocksField, 4>(
        state[0], state[1], state[2], state[3]
    ));

    // Second squeeze
    poseidon2_permute(&state);

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
    return poseidon2_hash_squeeze_twice_64(first_hash);
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

// Main mining kernel
@compute @workgroup_size(256)
fn mining_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // If solution already found, exit early
    if (atomicLoad(&results[0]) != 0u) {
        return;
    }

    let thread_id = global_id.x;
    // Read dispatch configuration from buffer
    let total_threads = dispatch_config[0];        // Total logical threads in this dispatch
    let nonces_per_thread = dispatch_config[1];    // Nonces processed by each thread
    let total_nonces = dispatch_config[2];         // Total logical nonces this dispatch should cover

    // Guard against threads beyond configured total_threads
    if (thread_id >= total_threads) {
        return;
    }

    // Base logical index for this thread
    let base_index = thread_id * nonces_per_thread;

    // Work coarsening: each thread processes a contiguous block of nonces
    for (var j = 0u; j < nonces_per_thread; j = j + 1u) {
        let logical_index = base_index + j;
        if (logical_index >= total_nonces) {
            break;
        }

        // Check if solution already found (early exit for entire dispatch)
        if (atomicLoad(&results[0]) != 0u) {
            return;
        }

        // current_nonce = start_nonce + logical_index
        var current_nonce: array<u32, 16>;
        var carry: u32 = 0u;

        // Add logical_index into the low limb and propagate carry through the U512 nonce
        let val0 = start_nonce[0];
        let sum0 = val0 + logical_index;
        current_nonce[0] = sum0;
        carry = select(0u, 1u, sum0 < val0);

        // Propagate carry through remaining limbs
        for (var i = 1u; i < 16u; i++) {
            let val = start_nonce[i];
            let sum = val + carry;
            current_nonce[i] = sum;
            carry = select(0u, 1u, sum < val);
        }

        // Construct input (96 bytes = 24 u32s)
        // Header (32 bytes = 8 u32s) followed by Nonce (64 bytes = 16 u32s)
        var input: array<u32, 24>;
        for (var i = 0u; i < 8u; i++) {
            input[i] = header[i];
        }
        // Nonce needs to be Big Endian in the byte stream for hashing.
        // current_nonce is Little Endian words.
        for (var i = 0u; i < 16u; i++) {
            let val = current_nonce[15u - i];
            // Reverse bytes
            let rev = ((val & 0xFFu) << 24u) |
                      ((val & 0xFF00u) << 8u) |
                      ((val & 0xFF0000u) >> 8u) |
                      ((val & 0xFF000000u) >> 24u);
            input[8u + i] = rev;
        }

        // Hash (Big Endian)
        let hash_be = double_hash(input);

        // Convert to Little Endian for difficulty check and storage
        var hash_le: array<u32, 16>;
        for (var i = 0u; i < 16u; i++) {
            let val = hash_be[15u - i];
            // Reverse bytes in u32
            hash_le[i] = ((val & 0xFFu) << 24u) |
                         ((val & 0xFF00u) << 8u) |
                         ((val & 0xFF0000u) >> 8u) |
                         ((val & 0xFF000000u) >> 24u);
        }

        // Check target
        if (is_below_target(hash_le, difficulty_target)) {
            // Try to claim the solution
            if (atomicExchange(&results[0], 1u) == 0u) {
                // We won! Write nonce and hash
                // results layout: [0]=found, [1..16]=nonce, [17..32]=hash
                for (var i = 0u; i < 16u; i++) {
                    atomicStore(&results[1u + i], current_nonce[i]);
                    atomicStore(&results[17u + i], hash_le[i]);
                }
            }
            return; // Exit loop after finding solution
        }
    }
}
