// Native-u64 Poseidon2 mining kernel. Apple Metal path only.
// Requires wgpu Features::SHADER_INT64. Bit-exact with mining.wgsl / pow_core.

@group(0) @binding(0) var<storage, read_write> results: array<atomic<u32>>;
// Sponge state after absorbing header + high nonce half (12 felts as LE u32 pairs),
// precomputed on the host per batch. See pow_core::mining_midstate.
// Pack u32 words into vec4s to preserve the byte layout with uniform alignment.
@group(0) @binding(1) var<uniform> midstate: array<vec4<u32>, 6>;
@group(0) @binding(2) var<uniform> start_nonce: array<vec4<u32>, 4>;
@group(0) @binding(3) var<uniform> difficulty_target: array<vec4<u32>, 4>;
@group(0) @binding(4) var<uniform> dispatch_config: vec4<u32>;

const P64: u64 = 0xFFFFFFFF00000001lu;
// EPS64 = 2^32 - 1 = 2^64 mod P
const EPS64: u64 = 0xFFFFFFFFlu;

// Padded with a trailing zero so the internal layer can add the next round's
// constant unconditionally.
const RC_INTERNAL: array<u64, 23> = array<u64, 23>(
    0x97f7798a784ad863lu, 0xd1d2bf082f60d4f0lu, 0x69a377a79f9ad206lu, 0xa9d06906a3858e24lu, 0x295275001eede5b5lu, 0x5874e441117bd746lu, 0x8a084bbba8ed86cclu, 0x3defd7645cde6425lu, 0x3998cfe6871cc137lu, 0x3e52ef8bca48314alu, 0x964a209f85dc9ecclu, 0x3fcc9ee82cc4577elu, 0x8e79b4a5d0096d6dlu, 0x8492362ad2392556lu, 0xee72f470262574d6lu, 0x1e0e18496da2444alu, 0x0f3a74bf215eaac6lu, 0x1b061b76a1c0ded3lu, 0x192c42d86803d7a6lu, 0xf6d49ff997ae0260lu, 0x3ec372e7a0fa3786lu, 0x5538cdf4f23445d3lu, 0lu
);
const RC_INITIAL: array<array<u64, 12>, 4> = array<array<u64, 12>, 4>(
    array<u64, 12>(0xc002e770975b1607lu, 0xbca51a8dfe14593alu, 0x72938dfbe774f7f9lu, 0xe4f2fe29e03234aclu, 0xd5e0ba2f541b6449lu, 0xec33b868f3cc46c1lu, 0x486dcb55419d475alu, 0x6c1cb2a358cc24f1lu, 0xe3f30d509a1436bblu, 0xd9a64f068dca7c29lu, 0xe59b3f57aabba1aelu, 0x2a3dd4505b478fdclu),
    array<u64, 12>(0xada1f8dc7676ed25lu, 0x2711aa8b5509d516lu, 0x4ae6acd0c9c92897lu, 0x56eb3d6b5256d67alu, 0x1f7a9d55923bf51elu, 0x3600427d397a7f68lu, 0xe5076df75b72c3d0lu, 0xfcd59aa12c6090adlu, 0xcd895e8c68b57a9elu, 0x41df7ef9d730ae3elu, 0xee3e2b889abe977dlu, 0xd29bb7edbeb9c405lu),
    array<u64, 12>(0x7d5c08eef608e382lu, 0x89ae889caaf0802clu, 0xb35a8e976d2af617lu, 0xdb14234eafaf5173lu, 0x78f04462d48b1c98lu, 0x265293b0e47ce88alu, 0x999a649b69b9d32flu, 0x64b0a186698e01d3lu, 0xee0b22d0dfae8bb8lu, 0x4fd53e50ca04a7eelu, 0x5762bfe181f25047lu, 0xf51593e2beb5e3bdlu),
    array<u64, 12>(0x1e5e2b5760e32477lu, 0x622462a1f9aaaeedlu, 0xaa284b3ecdb222aelu, 0x63c8e72f542bf3fclu, 0x3ba588cacb43b5e0lu, 0x23eda6f3c99150ddlu, 0xaad3bea4baac9a5alu, 0xe9da8d699b94184alu, 0xcdb13f4cd93e024clu, 0x902cbd0956f655e3lu, 0x5b4e40ffc759532flu, 0xde795c20a2357af7lu)
);
const RC_TERMINAL: array<array<u64, 12>, 4> = array<array<u64, 12>, 4>(
    array<u64, 12>(0x7b72c539e0ea4c6elu, 0x144573dae2ce9976lu, 0x802028b68f35fc88lu, 0x6d36c5022c4fe7c2lu, 0xa205d0ffa9b9def3lu, 0xf6e7e38b1ea6ba2flu, 0x34f7909ae5258d64lu, 0xb0464d9d77b97fcalu, 0x64ddb9d5de7e00a6lu, 0x0ed0d75c27975d97lu, 0x1cbb36f11127338blu, 0x6673e505cfd0b6balu),
    array<u64, 12>(0x605f902830872e01lu, 0x3fd5eb927e95fe4flu, 0xe81025b5a24c69cdlu, 0xf7d0ce75de23f74elu, 0xf39942b6a8585089lu, 0x6d808a08f7b71df6lu, 0xf8806b6588f49a8blu, 0x57df2d8c2a32107alu, 0x16e7c2074d654a2dlu, 0x213de241fcf33835lu, 0xb0f2b8905a0976f6lu, 0xd8e3cf2bbd355417lu),
    array<u64, 12>(0xe498691679d9330flu, 0x763b45d2a3821b28lu, 0x0908bf65eb0a1f0dlu, 0x7691eb2d194b24f4lu, 0x0e43551233ae13b2lu, 0x93c393dbfc2fe76flu, 0x98f607485d48cdealu, 0xe3d95f30309819c0lu, 0x1ef581a93eaf6acflu, 0x0b24c1b7a030fca4lu, 0x624370be5670b327lu, 0x5f1e28615a11e486lu),
    array<u64, 12>(0xfe04051f909e042blu, 0x7257e5b147fd3803lu, 0xe6ae134bb82f2e78lu, 0x5711fd5cf4784511lu, 0xf83a42660c08c0bclu, 0x2cd8c96d9a3ce855lu, 0x7d2ffb1bb0e17271lu, 0x85ae1528caea3811lu, 0x52a345d5c7adb0b8lu, 0x504c4c51f3faee94lu, 0xbce34a649cfccaf9lu, 0xe0a3389266fb6dc9lu)
);
const MDS_DIAG: array<u64, 12> = array<u64, 12>(
    0xc3b6c08e23ba9300lu, 0xd84b5de94a324fb6lu, 0x0d0c371c5b35b84flu, 0x7964f570e7188037lu, 0x5daf18bbd996604blu, 0x6743bc47b9595257lu, 0x5528b9362c59bb70lu, 0xac45e25b7127b68blu, 0xa2077d7dfbb606b5lu, 0xf3faac6faee378aelu, 0x0c6388b51545e883lu, 0xd27dbb6944917b60lu
);

// a + b mod P in lazy form. Wrapping carries fold back via 2^64 ≡ EPS64 (mod P).
fn gf64_add(a: u64, b: u64) -> u64 {
    let s0 = a + b;
    let c1 = s0 < a;
    let s1 = s0 + select(0lu, EPS64, c1);
    let c2 = c1 && (s1 < s0);
    return s1 + select(0lu, EPS64, c2);
}

// Sum accumulator: value = lo + carries * 2^64, folded once via 2^64 ≡ EPS64 (mod P).
struct Acc {
    lo: u64,
    carries: u32,
}

fn acc_add(a: Acc, b: u64) -> Acc {
    let s = a.lo + b;
    return Acc(s, a.carries + select(0u, 1u, s < a.lo));
}

fn acc_add2(a: Acc, b: Acc) -> Acc {
    let s = a.lo + b.lo;
    return Acc(s, a.carries + b.carries + select(0u, 1u, s < a.lo));
}

fn acc_fold(a: Acc) -> u64 {
    let c = u64(a.carries);
    let t = a.lo + ((c << 32u) - c);
    return t + select(0lu, EPS64, t < a.lo);
}

struct U128 {
    lo: u64,
    hi: u64,
}

// Reduce a 128-bit value (lo + hi*2^64) mod P using
// 2^64 ≡ EPS64 and 2^96 ≡ -1 (mod P).
fn gf64_reduce(v: U128) -> u64 {
    // In radix B=2^32, B^2 = B-1 and B^3 = -1 modulo P. Fold
    // the four limbs directly, avoiding 64-bit carry/borrow comparisons.
    let low = i64(u32(v.lo)) - i64(u32(v.hi)) - i64(u32(v.hi >> 32u));
    let high = i64(u32(v.lo >> 32u)) + i64(u32(v.hi)) + (low >> 32u);
    let result = (u64(u32(high)) << 32u) | u64(u32(low));
    // -1 <= high <= 2B-2: the correction is -1, 0, or 1.
    // A positive correction has result <= B^2-B-1; a negative one
    // has its high limb equal to B-1. Neither requires another fold.
    let correction = bitcast<u64>(high >> 32u);
    return result + ((correction << 32u) - correction);
}

fn mul_wide(a: u64, b: u64) -> U128 {
    let a_lo = a & EPS64;
    let a_hi = a >> 32u;
    let b_lo = b & EPS64;
    let b_hi = b >> 32u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    // Accumulate the middle limb in 32 bits; keep both carry bits explicitly.
    let mid0 = u32(ll >> 32u) + u32(lh);
    let c0 = select(0u, 1u, mid0 < u32(lh));
    let mid = mid0 + u32(hl);
    let c = c0 + select(0u, 1u, mid < mid0);
    return U128((u64(mid) << 32u) | u64(u32(ll)), hh + (lh >> 32u) + (hl >> 32u) + u64(c));
}

// (a*b + addend) mod P for b <= 2^64 - 2^32 (all MDS_DIAG entries): the addend's
// value and carries are folded into the 128-bit product before reduction.
fn gf64_mul_add(a: u64, b: u64, addend: Acc) -> u64 {
    let v = mul_wide(a, b);
    let lo = v.lo + addend.lo;
    return gf64_reduce(U128(lo, v.hi + u64(addend.carries) + select(0lu, 1lu, lo < v.lo)));
}

fn gf64_mul(a: u64, b: u64) -> u64 {
    return gf64_reduce(mul_wide(a, b));
}

fn gf64_sqr(a: u64) -> u64 {
    let a_lo = a & EPS64;
    let a_hi = a >> 32u;
    let ll = a_lo * a_lo;
    let lh = a_lo * a_hi;
    let hh = a_hi * a_hi;
    // Accumulate the middle limb in 32 bits; keep both carry bits explicitly.
    let mid0 = u32(ll >> 32u) + u32(lh);
    let c0 = select(0u, 1u, mid0 < u32(lh));
    let mid = mid0 + u32(lh);
    let c = c0 + select(0u, 1u, mid < mid0);
    return gf64_reduce(U128((u64(mid) << 32u) | u64(u32(ll)), hh + ((lh >> 32u) << 1u) + u64(c)));
}

fn gf64_sbox(x: u64) -> u64 {
    let x2 = gf64_sqr(x);
    let x4 = gf64_sqr(x2);
    let x6 = gf64_mul(x4, x2);
    return gf64_mul(x6, x);
}

fn gf64_canon(a: u64) -> u64 {
    return a - select(0lu, P64, a >= P64);
}

// 4x4 MDS circ(2, 3, 1, 1) on one chunk, results left unreduced.
fn mds4(x0: u64, x1: u64, x2: u64, x3: u64) -> array<Acc, 4> {
    let t01 = acc_add(Acc(x0, 0u), x1);
    let t23 = acc_add(Acc(x2, 0u), x3);
    let t0123 = acc_add2(t01, t23);
    let t01123 = acc_add(t0123, x1);
    let t01233 = acc_add(t0123, x3);
    return array<Acc, 4>(
        acc_add2(t01123, t01),
        acc_add2(t01123, Acc(x2 << 1u, u32(x2 >> 63u))),
        acc_add2(t01233, t23),
        acc_add2(t01233, Acc(x0 << 1u, u32(x0 >> 63u)))
    );
}

// External linear layer: 4x4 MDS on each chunk, then circulant sums, plus the
// next round's constants. Additions are accumulated unreduced (at most 27
// carries) and folded once per output.
fn ext_layer64(state: ptr<function, array<u64, 12>>, rc_index: u32) {
    var y: array<Acc, 12>;
    for (var chunk = 0u; chunk < 3u; chunk++) {
        let o = chunk * 4u;
        let m = mds4((*state)[o], (*state)[o + 1u], (*state)[o + 2u], (*state)[o + 3u]);
        y[o] = m[0];
        y[o + 1u] = m[1];
        y[o + 2u] = m[2];
        y[o + 3u] = m[3];
    }
    for (var k = 0u; k < 4u; k++) {
        let s = acc_add2(acc_add2(y[k], y[k + 4u]), y[k + 8u]);
        (*state)[k] = acc_fold(acc_add(acc_add2(y[k], s), RC_EXT[rc_index][k]));
        (*state)[k + 4u] = acc_fold(acc_add(acc_add2(y[k + 4u], s), RC_EXT[rc_index][k + 4u]));
        (*state)[k + 8u] = acc_fold(acc_add(acc_add2(y[k + 8u], s), RC_EXT[rc_index][k + 8u]));
    }
}

// Internal linear layer: diagonal matrix plus full sum, plus the next round's
// constant on element 0 (the S-box output, which is summed last).
fn int_layer64(state: ptr<function, array<u64, 12>>, rc0: u64) {
    let s12 = acc_add(Acc((*state)[1], 0u), (*state)[2]);
    let s34 = acc_add(Acc((*state)[3], 0u), (*state)[4]);
    let s56 = acc_add(Acc((*state)[5], 0u), (*state)[6]);
    let s78 = acc_add(Acc((*state)[7], 0u), (*state)[8]);
    let s910 = acc_add(Acc((*state)[9], 0u), (*state)[10]);
    let s1234 = acc_add2(s12, s34);
    let s5678 = acc_add2(s56, s78);
    let s91011 = acc_add(s910, (*state)[11]);
    let sum = acc_add(acc_add2(acc_add2(s1234, s5678), s91011), (*state)[0]);
    (*state)[0] = gf64_mul_add((*state)[0], MDS_DIAG[0], acc_add(sum, rc0));
    (*state)[1] = gf64_mul_add((*state)[1], MDS_DIAG[1], sum);
    (*state)[2] = gf64_mul_add((*state)[2], MDS_DIAG[2], sum);
    (*state)[3] = gf64_mul_add((*state)[3], MDS_DIAG[3], sum);
    (*state)[4] = gf64_mul_add((*state)[4], MDS_DIAG[4], sum);
    (*state)[5] = gf64_mul_add((*state)[5], MDS_DIAG[5], sum);
    (*state)[6] = gf64_mul_add((*state)[6], MDS_DIAG[6], sum);
    (*state)[7] = gf64_mul_add((*state)[7], MDS_DIAG[7], sum);
    (*state)[8] = gf64_mul_add((*state)[8], MDS_DIAG[8], sum);
    (*state)[9] = gf64_mul_add((*state)[9], MDS_DIAG[9], sum);
    (*state)[10] = gf64_mul_add((*state)[10], MDS_DIAG[10], sum);
    (*state)[11] = gf64_mul_add((*state)[11], MDS_DIAG[11], sum);
}

fn sbox_lanes(state: ptr<function, array<u64, 12>>, lanes: u32) {
    for (var i = 0u; i < lanes; i++) {
        (*state)[i] = gf64_sbox((*state)[i]);
    }
}

fn add_rc(state: ptr<function, array<u64, 12>>, rc: array<u64, 12>) {
    for (var i = 0u; i < 12u; i++) {
        (*state)[i] = gf64_add((*state)[i], rc[i]);
    }
}

const RC_ZERO: array<u64, 12> = array<u64, 12>(0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu);
// Constants added by each external layer, which feeds the S-box that follows it:
// entries 0..3 precede the initial rounds, entry 4 carries the first internal
// round's constant on element 0, entries 5..7 precede terminal rounds 1..3.
const RC_EXT: array<array<u64, 12>, 9> = array<array<u64, 12>, 9>(
    RC_INITIAL[0], RC_INITIAL[1], RC_INITIAL[2], RC_INITIAL[3],
    array<u64, 12>(RC_INTERNAL[0], 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu),
    RC_TERMINAL[1], RC_TERMINAL[2], RC_TERMINAL[3], RC_ZERO
);

// One loop drives all 30 rounds so each layer is emitted once: step 0 is the
// initial external layer, steps 1..4 and 27..30 are external rounds, steps
// 5..26 are internal rounds. Round constants are added by the linear layer
// preceding each S-box (the last internal layer is followed by the first
// terminal round's constants).
fn permute64(state: ptr<function, array<u64, 12>>) {
    for (var k = 0u; k < 31u; k++) {
        let is_ext = k < 5u || k > 26u;
        if (k > 0u) {
            sbox_lanes(state, select(1u, 12u, is_ext));
        }
        if (is_ext) {
            ext_layer64(state, select(k, k - 22u, k > 26u));
        } else {
            int_layer64(state, RC_INTERNAL[k - 4u]);
            if (k == 26u) {
                add_rc(state, RC_TERMINAL[0]);
            }
        }
    }
}

fn bswap32(v: u32) -> u32 {
    return ((v & 0xFFu) << 24u) | ((v & 0xFF00u) << 8u) | ((v >> 8u) & 0xFF00u) | (v >> 24u);
}

@compute @workgroup_size(256)
fn mining_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (atomicLoad(&results[0]) != 0u) {
        return;
    }
    let thread_id = global_id.x;
    let total_threads = dispatch_config[0];
    let nonces_per_thread = dispatch_config[1];
    let total_nonces = dispatch_config[2];
    if (thread_id >= total_threads) {
        return;
    }
    let base_index = thread_id * nonces_per_thread;

    // Hoist uniform storage reads out of the nonce loop
    var mid: array<u64, 12>;
    for (var i = 0u; i < 12u; i++) {
        mid[i] = (u64(midstate[(2u * i + 1u) / 4u][(2u * i + 1u) % 4u]) << 32u) | u64(midstate[(2u * i) / 4u][(2u * i) % 4u]);
    }
    var tgt: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        tgt[i] = difficulty_target[(i) / 4u][(i) % 4u];
    }
    var nonce_base: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        nonce_base[i] = start_nonce[(i) / 4u][(i) % 4u];
    }

    for (var j = 0u; j < nonces_per_thread; j = j + 1u) {
        let logical_index = base_index + j;
        if (logical_index >= total_nonces) {
            break;
        }
        if (j > 0u && atomicLoad(&results[0]) != 0u) {
            return;
        }

        // The host guarantees a batch never carries into the high nonce half
        // (limbs 8..15), so only the low 256 bits are incremented here.
        var current_nonce: array<u32, 16>;
        let val0 = nonce_base[0];
        let sum0 = val0 + logical_index;
        current_nonce[0] = sum0;
        var carry = select(0u, 1u, sum0 < val0);
        for (var i = 1u; i < 8u; i++) {
            let val = nonce_base[i];
            let sum = val + carry;
            current_nonce[i] = sum;
            carry = select(0u, 1u, sum < val);
        }
        for (var i = 8u; i < 16u; i++) {
            current_nonce[i] = nonce_base[i];
        }

        // Resume the sponge from the precomputed midstate: absorb the low
        // nonce half, pad, squeeze twice (3 permutations instead of 5).
        var st = mid;
        for (var i = 0u; i < 8u; i++) {
            st[i] = gf64_add(st[i], u64(bswap32(current_nonce[7u - i])));
        }
        // The overwhelmingly common rejection needs only the first hash word.
        // Delay materializing the full result until that comparison passes.
        for (var phase = 0u; phase < 2u; phase++) {
            permute64(&st);
            if (phase == 0u) {
                st[0] = gf64_add(st[0], 1lu);
                st[1] = gf64_add(st[1], 1lu);
            }
        }
        let first = bswap32(u32(gf64_canon(st[0]) & EPS64));
        if (first > tgt[15]) {
            continue;
        }
        var hash_le: array<u32, 16>;
        var cmp = 0u;
        for (var i = 0u; i < 4u; i++) {
            let c = gf64_canon(st[i]);
            hash_le[15u - 2u * i] = bswap32(u32(c & EPS64));
            hash_le[14u - 2u * i] = bswap32(u32(c >> 32u));
        }
        for (var i = 0u; i < 8u; i++) {
            let h = hash_le[15u - i];
            let t = tgt[15u - i];
            if (h != t) {
                cmp = select(2u, 1u, h > t);
                break;
            }
        }
        if (cmp == 1u) {
            continue;
        }
        permute64(&st);
        for (var i = 0u; i < 4u; i++) {
            let c = gf64_canon(st[i]);
            hash_le[7u - 2u * i] = bswap32(u32(c & EPS64));
            hash_le[6u - 2u * i] = bswap32(u32(c >> 32u));
        }
        var below = cmp == 2u;
        if (!below) {
            for (var i = 0u; i < 8u; i++) {
                let h = hash_le[7u - i];
                let t = tgt[7u - i];
                if (h != t) {
                    below = h < t;
                    break;
                }
            }
        }

        if (below) {
            if (atomicExchange(&results[0], 1u) == 0u) {
                for (var i = 0u; i < 16u; i++) {
                    atomicStore(&results[1u + i], current_nonce[i]);
                    atomicStore(&results[17u + i], hash_le[i]);
                }
            }
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Compatibility layer: same API as mining.wgsl, backed by the u64 core above.
// Only used by the component test harness; the mining kernel never calls it.
// All outputs are canonical, matching the reference implementation.
// ---------------------------------------------------------------------------

struct GoldilocksField {
    limb0: u32,
    limb1: u32,
}

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

fn gf_pack(g: GoldilocksField) -> u64 {
    return (u64(g.limb1) << 32u) | u64(g.limb0);
}

fn gf_unpack(v: u64) -> GoldilocksField {
    let c = gf64_canon(v);
    return GoldilocksField(u32(c & EPS64), u32(c >> 32u));
}

fn gf_from_limbs(l0: u32, l1: u32) -> GoldilocksField {
    return GoldilocksField(l0, l1);
}

fn gf_zero() -> GoldilocksField {
    return GoldilocksField(0u, 0u);
}

fn gf_one() -> GoldilocksField {
    return GoldilocksField(1u, 0u);
}

fn gf_from_u32(val: u32) -> GoldilocksField {
    return GoldilocksField(val, 0u);
}

fn gf_from_u64_parts(low: u32, high: u32) -> GoldilocksField {
    return gf_unpack((u64(high) << 32u) | u64(low));
}

fn gf_from_const(val: array<u32, 2>) -> GoldilocksField {
    return gf_from_u64_parts(val[0], val[1]);
}

fn gf_add(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    return gf_unpack(gf64_add(gf_pack(a), gf_pack(b)));
}

fn gf_mul(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    return gf_unpack(gf64_mul(gf_pack(a), gf_pack(b)));
}

fn sbox(x: GoldilocksField) -> GoldilocksField {
    return gf_unpack(gf64_sbox(gf_pack(x)));
}

fn state_pack(state: ptr<function, array<GoldilocksField, 12>>, out: ptr<function, array<u64, 12>>) {
    for (var i = 0u; i < 12u; i++) {
        (*out)[i] = gf_pack((*state)[i]);
    }
}

fn state_unpack(v: ptr<function, array<u64, 12>>, state: ptr<function, array<GoldilocksField, 12>>) {
    for (var i = 0u; i < 12u; i++) {
        (*state)[i] = gf_unpack((*v)[i]);
    }
}

fn external_linear_layer(state: ptr<function, array<GoldilocksField, 12>>) {
    var st: array<u64, 12>;
    state_pack(state, &st);
    ext_layer64(&st, 8u);
    state_unpack(&st, state);
}

fn internal_linear_layer(state: ptr<function, array<GoldilocksField, 12>>) {
    var st: array<u64, 12>;
    state_pack(state, &st);
    int_layer64(&st, 0lu);
    state_unpack(&st, state);
}

fn poseidon2_permute(state: ptr<function, array<GoldilocksField, 12>>) {
    var st: array<u64, 12>;
    state_pack(state, &st);
    permute64(&st);
    state_unpack(&st, state);
}

fn bytes_to_field_elements(input: array<u32, 24>) -> array<GoldilocksField, 25> {
    var felts: array<GoldilocksField, 25>;
    for (var i = 0u; i < 24u; i++) {
        felts[i] = gf_from_u32(input[i]);
    }
    felts[24] = gf_one();
    return felts;
}

fn field_elements_to_bytes(felts: array<GoldilocksField, 4>) -> array<u32, 8> {
    var result: array<u32, 8>;
    for (var i = 0u; i < 4u; i++) {
        result[i * 2u] = felts[i].limb0;
        result[i * 2u + 1u] = felts[i].limb1;
    }
    return result;
}

fn poseidon2_hash_squeeze_twice(input: array<u32, 24>) -> array<u32, 16> {
    var st: array<u64, 12>;
    for (var i = 0u; i < 12u; i++) {
        st[i] = 0lu;
    }
    for (var chunk = 0u; chunk < 3u; chunk++) {
        for (var i = 0u; i < 8u; i++) {
            st[i] = gf64_add(st[i], u64(input[chunk * 8u + i]));
        }
        permute64(&st);
    }
    st[0] = gf64_add(st[0], 1lu);
    st[1] = gf64_add(st[1], 1lu);
    permute64(&st);

    var result: array<u32, 16>;
    for (var i = 0u; i < 4u; i++) {
        let c = gf64_canon(st[i]);
        result[2u * i] = u32(c & EPS64);
        result[2u * i + 1u] = u32(c >> 32u);
    }
    permute64(&st);
    for (var i = 0u; i < 4u; i++) {
        let c = gf64_canon(st[i]);
        result[8u + 2u * i] = u32(c & EPS64);
        result[8u + 2u * i + 1u] = u32(c >> 32u);
    }
    return result;
}

fn hash_squeeze_twice(input: array<u32, 24>) -> array<u32, 16> {
    return poseidon2_hash_squeeze_twice(input);
}
