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
@group(0) @binding(0) var<storage, read_write> results: array<u32>;
@group(0) @binding(1) var<storage, read> header: array<u32, 8>;     // 32 bytes
@group(0) @binding(2) var<storage, read> start_nonce: array<u32, 16>; // 64 bytes
@group(0) @binding(3) var<storage, read> difficulty_target: array<u32, 16>;    // 64 bytes (U512 target)

// Goldilocks field element represented as [limb0, limb1, limb2, limb3]
// where the value is limb0 + limb1*2^16 + limb2*2^32 + limb3*2^48
struct GoldilocksField {
    limb0: u32,  // Actually u16, but WGSL doesn't have native u16
    limb1: u32,
    limb2: u32,
    limb3: u32,
}

// Field modulus P = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
// In u16 limbs: P = [1, 0, 0xFFFF, 0xFFFF]
const P_LIMB0: u32 = 1u;
const P_LIMB1: u32 = 0u;
const P_LIMB2: u32 = 0xFFFFu;
const P_LIMB3: u32 = 0xFFFFu;

// EPSILON = 2^32 - 1 = 0x00000000FFFFFFFF
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
    var result = GoldilocksField(
        low & 0xFFFFu,         // bits 0-15
        (low >> 16u) & 0xFFFFu, // bits 16-31
        high & 0xFFFFu,        // bits 32-47
        (high >> 16u) & 0xFFFFu // bits 48-63
    );

    // Reduce if >= P
    if (gf_compare(result, gf_from_limbs(P_LIMB0, P_LIMB1, P_LIMB2, P_LIMB3)) != 2u) {
        result = gf_sub(result, gf_from_limbs(P_LIMB0, P_LIMB1, P_LIMB2, P_LIMB3));
    }

    return result;
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
    // Add limb by limb with carry propagation
    let add0 = u16_add_with_carry(a.limb0, b.limb0, 0u);
    let add1 = u16_add_with_carry(a.limb1, b.limb1, add0.y);
    let add2 = u16_add_with_carry(a.limb2, b.limb2, add1.y);
    let add3 = u16_add_with_carry(a.limb3, b.limb3, add2.y);

    var result = GoldilocksField(add0.x, add1.x, add2.x, add3.x);

    // Handle overflow: if carry out, we've computed a + b = result + 2^64
    // In Goldilocks: 2^64 ≡ 2^32 - 1 (mod P)
    if (add3.y != 0u) {
        // Add 2^32 - 1 = 0xFFFFFFFF = [0xFFFF, 0xFFFF, 0, 0]
        let eps_add0 = u16_add_with_carry(result.limb0, EPSILON_LIMB0, 0u);
        let eps_add1 = u16_add_with_carry(result.limb1, EPSILON_LIMB1, eps_add0.y);
        let eps_add2 = u16_add_with_carry(result.limb2, EPSILON_LIMB2, eps_add1.y);
        let eps_add3 = u16_add_with_carry(result.limb3, EPSILON_LIMB3, eps_add2.y);
        result = GoldilocksField(eps_add0.x, eps_add1.x, eps_add2.x, eps_add3.x);

        // If adding EPSILON caused another overflow, add EPSILON again
        if (eps_add3.y != 0u) {
            let eps2_add0 = u16_add_with_carry(result.limb0, EPSILON_LIMB0, 0u);
            let eps2_add1 = u16_add_with_carry(result.limb1, EPSILON_LIMB1, eps2_add0.y);
            let eps2_add2 = u16_add_with_carry(result.limb2, EPSILON_LIMB2, eps2_add1.y);
            let eps2_add3 = u16_add_with_carry(result.limb3, EPSILON_LIMB3, eps2_add2.y);
            result = GoldilocksField(eps2_add0.x, eps2_add1.x, eps2_add2.x, eps2_add3.x);
        }
    }

    // Final reduction if result >= P
    let p = gf_from_limbs(P_LIMB0, P_LIMB1, P_LIMB2, P_LIMB3);
    if (gf_compare(result, p) != 2u) { // if result >= P
        result = gf_sub_no_underflow(result, p);
    }

    return result;
}

// Helper for subtraction without underflow (assumes a >= b)
fn gf_sub_no_underflow(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    let sub0 = u16_sub_with_borrow(a.limb0, b.limb0, 0u);
    let sub1 = u16_sub_with_borrow(a.limb1, b.limb1, sub0.y);
    let sub2 = u16_sub_with_borrow(a.limb2, b.limb2, sub1.y);
    let sub3 = u16_sub_with_borrow(a.limb3, b.limb3, sub2.y);
    return GoldilocksField(sub0.x, sub1.x, sub2.x, sub3.x);
}

// Goldilocks field subtraction
fn gf_sub(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    // If a >= b, do direct subtraction
    if (gf_compare(a, b) != 2u) {
        return gf_sub_no_underflow(a, b);
    }

    // Otherwise: a < b, so compute a - b + P
    let p = gf_from_limbs(P_LIMB0, P_LIMB1, P_LIMB2, P_LIMB3);
    let a_plus_p = gf_add_no_reduction(a, p);
    return gf_sub_no_underflow(a_plus_p, b);
}

// Addition without final modular reduction (used internally)
fn gf_add_no_reduction(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    let add0 = u16_add_with_carry(a.limb0, b.limb0, 0u);
    let add1 = u16_add_with_carry(a.limb1, b.limb1, add0.y);
    let add2 = u16_add_with_carry(a.limb2, b.limb2, add1.y);
    let add3 = u16_add_with_carry(a.limb3, b.limb3, add2.y);
    return GoldilocksField(add0.x, add1.x, add2.x, add3.x);
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
// Simplified version matching CPU algorithm exactly
fn gf_reduce_8limb(limbs: array<u32, 8>) -> GoldilocksField {
    // Convert to 64-bit representation like CPU
    let x_lo = gf_from_limbs(limbs[0], limbs[1], limbs[2], limbs[3]);
    let x_hi = gf_from_limbs(limbs[4], limbs[5], limbs[6], limbs[7]);

    // If high part is zero, just return low part
    if (limbs[4] == 0u && limbs[5] == 0u && limbs[6] == 0u && limbs[7] == 0u) {
        return x_lo;
    }

    // x_hi_hi = x_hi >> 32 (upper 32 bits of high part)
    let x_hi_hi = gf_from_limbs(limbs[6], limbs[7], 0u, 0u);

    // x_hi_lo = x_hi & NEG_ORDER (lower 32 bits of high part)
    let x_hi_lo = gf_from_limbs(limbs[4], limbs[5], 0u, 0u);

    // Step 1: t0 = x_lo - x_hi_hi (with underflow detection)
    var t0: GoldilocksField;
    var underflow = false;

    // Perform subtraction with borrow propagation
    let sub0 = u16_sub_with_borrow(x_lo.limb0, x_hi_hi.limb0, 0u);
    let sub1 = u16_sub_with_borrow(x_lo.limb1, x_hi_hi.limb1, sub0.y);
    let sub2 = u16_sub_with_borrow(x_lo.limb2, x_hi_hi.limb2, sub1.y);
    let sub3 = u16_sub_with_borrow(x_lo.limb3, x_hi_hi.limb3, sub2.y);
    t0 = GoldilocksField(sub0.x, sub1.x, sub2.x, sub3.x);

    // Check for final borrow (underflow)
    if (sub3.y != 0u) {
        underflow = true;
    }

    // Step 2: if underflow { t0 -= NEG_ORDER; }
    if (underflow) {
        let eps_sub0 = u16_sub_with_borrow(t0.limb0, EPSILON_LIMB0, 0u);
        let eps_sub1 = u16_sub_with_borrow(t0.limb1, EPSILON_LIMB1, eps_sub0.y);
        let eps_sub2 = u16_sub_with_borrow(t0.limb2, EPSILON_LIMB2, eps_sub1.y);
        let eps_sub3 = u16_sub_with_borrow(t0.limb3, EPSILON_LIMB3, eps_sub2.y);
        t0 = GoldilocksField(eps_sub0.x, eps_sub1.x, eps_sub2.x, eps_sub3.x);
    }

    // Step 3: t1 = x_hi_lo * NEG_ORDER
    // Since x_hi_lo fits in 32 bits and NEG_ORDER = 2^32 - 1, we can optimize:
    // x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo
    let shifted = GoldilocksField(0u, 0u, x_hi_lo.limb0, x_hi_lo.limb1);
    let t1_sub0 = u16_sub_with_borrow(shifted.limb0, x_hi_lo.limb0, 0u);
    let t1_sub1 = u16_sub_with_borrow(shifted.limb1, x_hi_lo.limb1, t1_sub0.y);
    let t1_sub2 = u16_sub_with_borrow(shifted.limb2, x_hi_lo.limb2, t1_sub1.y);
    let t1_sub3 = u16_sub_with_borrow(shifted.limb3, x_hi_lo.limb3, t1_sub2.y);
    let t1 = GoldilocksField(t1_sub0.x, t1_sub1.x, t1_sub2.x, t1_sub3.x);

    // Step 4: result = t0 + t1 (with overflow handling like CPU)
    let add0 = u16_add_with_carry(t0.limb0, t1.limb0, 0u);
    let add1 = u16_add_with_carry(t0.limb1, t1.limb1, add0.y);
    let add2 = u16_add_with_carry(t0.limb2, t1.limb2, add1.y);
    let add3 = u16_add_with_carry(t0.limb3, t1.limb3, add2.y);
    var result = GoldilocksField(add0.x, add1.x, add2.x, add3.x);

    // If overflow, add NEG_ORDER (like CPU add_no_canonicalize_trashing_input)
    if (add3.y != 0u) {
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
        var chunk_state: array<GoldilocksField, 4>;

        // Copy chunk
        for (var i = 0u; i < 4u; i++) {
            chunk_state[i] = (*state)[offset + i];
        }

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

    // Now we have element 24 (the padding marker = 1) remaining
    // This simulates CPU's finalize_twice adding ONE to buffer position 0
    state[0] = gf_add(state[0], input_felts[24u]); // Add the padding marker (should be 1)

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

// No main function - individual tests will create their own entry points
