use primitive_types::U512;
use qp_poseidon_core::poseidon2::INITIAL_EXTERNAL_CONSTANTS;
use qp_poseidon_core::{Goldilocks, Poseidon2};

pub use qp_poseidon_core::SPONGE_WIDTH;
pub use qpow_math::{get_nonce_hash, is_valid_nonce, mine_range};

/// Split a U512 into 16 little-endian u32 limbs (same layout the GPU kernels use).
pub fn u512_to_le_u32s(n: U512) -> [u32; 16] {
    let bytes = n.to_little_endian();
    let mut out = [0u32; 16];
    for i in 0..16 {
        out[i] = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    }
    out
}

/// Rebuild a U512 from 16 little-endian u32 limbs.
pub fn u512_from_le_u32s(limbs: [u32; 16]) -> U512 {
    let mut bytes = [0u8; 64];
    for i in 0..16 {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&limbs[i].to_le_bytes());
    }
    U512::from_little_endian(&bytes)
}

/// Midstate as 12 felts packed into 24 little-endian u32 limbs.
pub fn mining_midstate_u32s(header: [u8; 32], nonce_high_be: [u8; 32]) -> [u32; 24] {
    let mid = mining_midstate(header, nonce_high_be);
    let mut out = [0u32; 24];
    for (i, felt) in mid.iter().enumerate() {
        out[2 * i] = *felt as u32;
        out[2 * i + 1] = (*felt >> 32) as u32;
    }
    out
}

/// Poseidon2 state after the nonce-invariant part of the next permutation's
/// initial linear layer and first round constants.
///
/// The low 64 bits of `nonce_be` are deliberately excluded. A GPU kernel can
/// inject their sparse linear contribution per nonce, then resume immediately
/// before the first external-round S-box. Callers must split batches before the
/// low 64 bits carry.
pub fn mining_prestate_low64_u32s(header: [u8; 32], nonce_be: [u8; 64]) -> [u32; 24] {
    let prestate = mining_prestate_low64(header, nonce_be);
    let mut out = [0u32; 24];
    for (i, felt) in prestate.iter().enumerate() {
        out[2 * i] = *felt as u32;
        out[2 * i + 1] = (*felt >> 32) as u32;
    }
    out
}

/// Canonical-felt form of [`mining_prestate_low64_u32s`].
pub fn mining_prestate_low64(header: [u8; 32], nonce_be: [u8; 64]) -> [u64; SPONGE_WIDTH] {
    let mut state =
        mining_midstate(header, nonce_be[..32].try_into().unwrap()).map(Goldilocks::from_u64);

    // Absorb the fixed upper 192 bits of the nonce's low 256-bit half. The
    // final two words are the low 64 bits specialized by the GPU kernel.
    for (i, chunk) in nonce_be[32..56].chunks_exact(4).enumerate() {
        state[i] += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
    external_linear_layer(&mut state);
    for (felt, constant) in state.iter_mut().zip(INITIAL_EXTERNAL_CONSTANTS[0]) {
        *felt += Goldilocks::from_u64(constant);
    }
    state.map(|felt| felt.as_canonical_u64())
}

fn external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        let chunk: &mut [Goldilocks; 4] = chunk.try_into().unwrap();
        let t01 = chunk[0] + chunk[1];
        let t23 = chunk[2] + chunk[3];
        let t0123 = t01 + t23;
        let t01123 = t0123 + chunk[1];
        let t01233 = t0123 + chunk[3];
        chunk[3] = t01233 + chunk[0] + chunk[0];
        chunk[1] = t01123 + chunk[2] + chunk[2];
        chunk[0] = t01123 + t01;
        chunk[2] = t01233 + t23;
    }

    let sums: [Goldilocks; 4] =
        std::array::from_fn(|offset| (offset..SPONGE_WIDTH).step_by(4).map(|i| state[i]).sum());
    for (i, felt) in state.iter_mut().enumerate() {
        *felt += sums[i % 4];
    }
}

/// Sponge state (canonical u64 felts) after absorbing the 32-byte header and the
/// high 32 bytes of the big-endian nonce — the first two of the five Poseidon2
/// permutations of `get_nonce_hash`. This state is identical for every nonce in a
/// batch as long as incrementing the nonce never carries into its high 256 bits,
/// so GPU kernels can resume the sponge from here and skip 2 of 5 permutations.
pub fn mining_midstate(header: [u8; 32], nonce_high_be: [u8; 32]) -> [u64; SPONGE_WIDTH] {
    let poseidon2 = Poseidon2::new();
    let mut state = [Goldilocks::ZERO; SPONGE_WIDTH];
    for (i, chunk) in header.chunks_exact(4).enumerate() {
        state[i] += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
    poseidon2.permute_mut(&mut state);
    for (i, chunk) in nonce_high_be.chunks_exact(4).enumerate() {
        state[i] += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
    poseidon2.permute_mut(&mut state);
    state.map(|g| g.as_canonical_u64())
}

/// Hardcoded get_nonce_hash / midstate vectors for CPU and CUDA regression.
pub struct NonceHashKv {
    pub header: &'static str,
    pub nonce: &'static str,
    pub hash: &'static str,
    pub mid: [u64; 12],
}

pub const NONCE_HASH_KVS: &[NonceHashKv] = &[
    NonceHashKv {
        header: "0000000000000000000000000000000000000000000000000000000000000000",
        nonce: "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        hash: "8e64e3d8e0f38f882e8501f9e525df0a95d2e91e9cfc32c9248d756fb07780e2f8fdca2c5a54441e6fcd8d774a5f6aae72f36d1c76bc19f691a0d4f6c607e8cc",
        mid: [
            0x8646d336b5a0fccd,
            0x818ca59548916345,
            0x49475e8bc9c928bc,
            0xc469163bcdc4900b,
            0xe4421fafacd59087,
            0x633f49b998698d66,
            0x4a9067e3883e76da,
            0x193039e85ea14472,
            0x203d2d3ee6a8eba8,
            0xef5b9ce39bcff072,
            0xc16a3c8dd5680003,
            0xd77c52ab5bf7cf1f,
        ],
    },
    NonceHashKv {
        header: "0101010101010101010101010101010101010101010101010101010101010101",
        nonce: "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001",
        hash: "e4ef79db80b642a093d7e38e6a6daac7ec1cca7293ddbe3710b4e6781be4add0b93f7e3440626df5bb23b757436787ba8d3fd0a2652e387f8af5bd8a5cc2ce2f",
        mid: [
            0x788cf743676a5c85,
            0xdce1d727e1189b53,
            0x36e0e019f5a0bf3a,
            0x806487dd9c3bf83f,
            0xcb6d7300de1ab5fe,
            0x7dbe191fa64fc6b7,
            0x95dedc7fe99861be,
            0xb1a3885303f7467d,
            0x891eb002a34a0f17,
            0x4be4cf057a87208a,
            0xe9113e7dc541745f,
            0xfaeda908097d00f1,
        ],
    },
    NonceHashKv {
        header: "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        nonce: "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001234567890abcdef",
        hash: "7f6c410ce62e7fc54811ad3b8b92664ac0f8b1d2cd6eb163129251146e7ee34d27794aaae1b31eaaeb98ab835bae6a2b9ac4ad6f08c95903e1423f0b865816ff",
        mid: [
            0xf6fc848ab2dddc18,
            0xfd88d1f0ccda99fe,
            0x27cfa43b8ab9947a,
            0x6a505b37f02183d3,
            0x5f98f1d3524fb0c6,
            0xd6de4efd98793e2e,
            0xf3f205bd1e4d4526,
            0xc0c9046201e91dd6,
            0xcd2d28eef32b18de,
            0xa303ecbc8e98000c,
            0x3db245e7738f3aca,
            0xbc55f7553023863e,
        ],
    },
    NonceHashKv {
        header: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        nonce: "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ffffffffffffffff",
        hash: "822f70999ee517f610b51bbb79ba88bfbb19cb77b316df25d8aeb4c2682673555e9fc6a588ee96dcce4ff0d2b8d97f4aa89d0f46c97f9027b386d42b14be5100",
        mid: [
            0xd1fe93daa1f07409,
            0x0ea440f2b57a0ecf,
            0x3b7a1f3cb2d48e39,
            0x43a5a1255658d1b2,
            0xebe00ad8f0fe28c0,
            0xb5618e7c6d6072d1,
            0x158b7b19f80ec62c,
            0x329a91c8de5575c2,
            0xd1637d851a474212,
            0x2d927735708f3b7d,
            0x6219c7cf8ef5af51,
            0x802028c39ccd8f13,
        ],
    },
    NonceHashKv {
        header: "0707070707070707070707070707070707070707070707070707070707070707",
        nonce: "00000000000000000000000000000000000000000deadbeefcafe000000000000000000000000000000000000000000000000000000000001234567890abcdef",
        hash: "eab81cdb33db293aef7e1f10f85cb08d17bc197c284d46a25d9dc57cdb7ba878adf3f5c354318121876795aab08c0941afda21d835383faa2f139c26be31a30a",
        mid: [
            0x41885398f5b05a0d,
            0xb7f17cee3f70d242,
            0x307aecc5bca54c7e,
            0xb74164c0d6de7e91,
            0x0c0a9b27281c8181,
            0xd84ddf12fdddf0ee,
            0x88f8cdf552c0a15e,
            0x88da4cfbfbf92d22,
            0x8c344261349149db,
            0x0c0fd0d6a8444652,
            0x7a57b00736d58568,
            0x9ad40597f4fe2abc,
        ],
    },
];

/// Format a U512 in a human-readable way (scientific notation for large numbers).
pub fn format_u512(n: U512) -> String {
    if n.is_zero() {
        return "0".to_string();
    }
    let s = format!("{}", n);
    let len = s.len();
    if len <= 12 {
        s
    } else {
        format!("{}e{}", &s[..4], len - 1)
    }
}

/// Format a hash rate with appropriate units (H/s, KH/s, MH/s, GH/s).
pub fn format_hashrate(hashes_per_sec: f64) -> String {
    if hashes_per_sec >= 1_000_000_000.0 {
        format!("{:.2} GH/s", hashes_per_sec / 1_000_000_000.0)
    } else if hashes_per_sec >= 1_000_000.0 {
        format!("{:.2} MH/s", hashes_per_sec / 1_000_000.0)
    } else if hashes_per_sec >= 1_000.0 {
        format!("{:.2} KH/s", hashes_per_sec / 1_000.0)
    } else {
        format!("{:.2} H/s", hashes_per_sec)
    }
}

/// Job context for PoW mining with Poseidon2 hash_squeeze_twice
#[derive(Debug, Clone)]
pub struct JobContext {
    pub header: [u8; 32],
    pub difficulty: U512,
    pub target: U512,
}

impl JobContext {
    /// Build a new context from header and difficulty.
    ///
    /// # Panics
    ///
    /// Panics if `difficulty` is zero (division by zero in target calculation).
    /// Callers must validate that difficulty is non-zero before calling this function.
    pub fn new(header: [u8; 32], difficulty: U512) -> Self {
        // In Bitcoin-style PoW, target = max_target / difficulty
        let max_target = U512::MAX;
        let target = max_target / difficulty;

        JobContext {
            header,
            difficulty,
            target,
        }
    }
}

/// Initialize a worker with starting nonce (no special initialization needed for Bitcoin-style)
pub fn init_worker_nonce(start_nonce: U512) -> U512 {
    start_nonce
}

/// Advance nonce by one (simple increment for Bitcoin-style)
pub fn step_nonce(nonce: U512) -> U512 {
    nonce.saturating_add(U512::from(1u64))
}

/// Compute hash for the current nonce using Poseidon2 hash_squeeze_twice
pub fn hash_from_nonce(ctx: &JobContext, nonce: U512) -> U512 {
    let nonce_bytes = nonce.to_big_endian();
    qpow_math::get_nonce_hash(ctx.header, nonce_bytes)
}

/// Job-local CPU hasher that reuses the header/high-nonce sponge state.
///
/// The reference `hash_from_nonce` remains independent. This mining path skips
/// the final squeeze when the first half already exceeds the target, but
/// returns the full reference hash for every qualifying nonce.
pub struct MiningHasher {
    header: [u8; 32],
    target: U512,
    target_bytes: [u8; 64],
    poseidon: Poseidon2,
    cached: Option<([u8; 32], [Goldilocks; SPONGE_WIDTH])>,
}

impl MiningHasher {
    pub fn new(ctx: &JobContext) -> Self {
        Self {
            header: ctx.header,
            target: ctx.target,
            target_bytes: ctx.target.to_big_endian(),
            poseidon: Poseidon2::new(),
            cached: None,
        }
    }

    /// Return the full hash iff it is strictly below this job's target.
    /// Arbitrary nonce order and carries into the high half are supported.
    pub fn hash_if_valid(&mut self, nonce: U512) -> Option<U512> {
        let bytes = nonce.to_big_endian();
        let high: [u8; 32] = bytes[..32].try_into().unwrap();
        let mut state = match self.cached {
            Some((cached_high, state)) if cached_high == high => state,
            _ => {
                let state = mining_midstate(self.header, high).map(Goldilocks::from_u64);
                self.cached = Some((high, state));
                state
            }
        };
        for (felt, chunk) in state.iter_mut().zip(bytes[32..].chunks_exact(4)) {
            *felt += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
        }
        self.poseidon.permute_mut(&mut state);
        state[0] += Goldilocks::ONE;
        state[1] += Goldilocks::ONE;
        self.poseidon.permute_mut(&mut state);

        let mut hash_bytes = [0u8; 64];
        for (felt, chunk) in state.iter().zip(hash_bytes[..32].chunks_exact_mut(8)) {
            chunk.copy_from_slice(&felt.as_canonical_u64().to_le_bytes());
        }
        if hash_bytes[..32] > self.target_bytes[..32] {
            return None;
        }
        self.poseidon.permute_mut(&mut state);
        for (felt, chunk) in state.iter().zip(hash_bytes[32..].chunks_exact_mut(8)) {
            chunk.copy_from_slice(&felt.as_canonical_u64().to_le_bytes());
        }
        let hash = U512::from_big_endian(&hash_bytes);
        (hash < self.target).then_some(hash)
    }
}

/// Check if hash meets difficulty target
pub fn is_valid_hash(ctx: &JobContext, hash: U512) -> bool {
    hash < ctx.target
}

/// Mine a range of nonces starting from start_nonce
pub fn mine_nonce_range(ctx: &JobContext, start_nonce: U512, steps: u64) -> Option<(U512, U512)> {
    let start_nonce_bytes = start_nonce.to_big_endian();

    if let Some((nonce_bytes, hash)) =
        mine_range(ctx.header, start_nonce_bytes, steps, ctx.difficulty)
    {
        let nonce = U512::from_big_endian(&nonce_bytes);
        Some((nonce, hash))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mining_hasher_matches_reference_across_cache_changes() {
        let boundary = U512::one() << 256;
        let mut nonces = vec![U512::zero(), U512::one(), U512::MAX];
        for offset in 0..8u64 {
            nonces.push(boundary - U512::from(4u64) + U512::from(offset));
        }
        // Jump backwards and between high halves as well as incrementing.
        nonces.extend([U512::one(), U512::MAX, boundary, U512::zero()]);
        let mut seed = 0x517a9e37u64;
        for _ in 0..64 {
            let mut bytes = [0u8; 64];
            for chunk in bytes.chunks_exact_mut(8) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                chunk.copy_from_slice(&seed.to_le_bytes());
            }
            nonces.push(U512::from_big_endian(&bytes));
        }
        for header in [0u8, 17, 255] {
            for target in [U512::zero(), U512::one(), U512::MAX >> 1, U512::MAX] {
                let mut ctx = JobContext::new([header; 32], U512::one());
                ctx.target = target;
                let mut hasher = MiningHasher::new(&ctx);
                for &nonce in &nonces {
                    let hash = hash_from_nonce(&ctx, nonce);
                    assert_eq!(hasher.hash_if_valid(nonce), (hash < target).then_some(hash));
                }
            }
        }
    }

    #[test]
    fn mining_hasher_preserves_strict_full_hash_comparison() {
        for vector in NONCE_HASH_KVS {
            let nonce = U512::from_big_endian(&decode64(vector.nonce));
            let mut ctx = JobContext::new(decode32(vector.header), U512::one());
            let hash = U512::from_big_endian(&decode64(vector.hash));
            for target in [hash - U512::one(), hash, hash + U512::one()] {
                ctx.target = target;
                let mut hasher = MiningHasher::new(&ctx);
                // Both a cold cache and a reused midstate must behave identically.
                for _ in 0..2 {
                    assert_eq!(hasher.hash_if_valid(nonce), (hash < target).then_some(hash));
                }
            }
        }
    }

    #[test]
    fn test_job_context_creation() {
        let header = [1u8; 32];
        let difficulty = U512::from(1000u64);

        let ctx = JobContext::new(header, difficulty);

        assert_eq!(ctx.header, header);
        assert_eq!(ctx.difficulty, difficulty);
        assert_eq!(ctx.target, U512::MAX / difficulty);
    }

    #[test]
    fn test_nonce_stepping() {
        let start = U512::from(100u64);
        let next = step_nonce(start);

        assert_eq!(next, U512::from(101u64));
    }

    #[test]
    fn test_hash_computation() {
        let header = [1u8; 32];
        let difficulty = U512::from(1u64);
        let ctx = JobContext::new(header, difficulty);

        let nonce = U512::from(123u64);
        let hash1 = hash_from_nonce(&ctx, nonce);
        let hash2 = hash_from_nonce(&ctx, nonce);

        // Same input should produce same hash
        assert_eq!(hash1, hash2);

        // Hash should not be zero for non-zero nonce
        assert_ne!(hash1, U512::zero());
    }

    #[test]
    fn test_different_nonces_different_hashes() {
        let header = [2u8; 32];
        let difficulty = U512::from(1u64);
        let ctx = JobContext::new(header, difficulty);

        let nonce1 = U512::from(100u64);
        let nonce2 = U512::from(101u64);

        let hash1 = hash_from_nonce(&ctx, nonce1);
        let hash2 = hash_from_nonce(&ctx, nonce2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_validity_check() {
        let header = [3u8; 32];
        let easy_difficulty = U512::from(1u64);
        let ctx = JobContext::new(header, easy_difficulty);

        let nonce = U512::from(1u64);
        let (is_valid, hash) = is_valid_nonce(ctx.header, nonce.to_big_endian(), ctx.difficulty);

        // With very easy difficulty, should be valid
        assert!(is_valid);
        assert_ne!(hash, U512::zero());

        // Verify hash is actually below target
        assert!(hash < ctx.target);
    }

    #[test]
    fn test_target_calculation() {
        let header = [4u8; 32];
        let difficulty = U512::from(256u64);
        let ctx = JobContext::new(header, difficulty);

        let expected_target = U512::MAX / U512::from(256u64);
        assert_eq!(ctx.target, expected_target);
    }

    #[test]
    fn test_hash_matches_qpow_math() {
        // Test that our JobContext produces the same results as qpow_math directly
        let header = [1u8; 32];
        let nonce = U512::from(123u64);
        let difficulty = U512::from(1000u64);

        let ctx = JobContext::new(header, difficulty);
        let hash_ctx = hash_from_nonce(&ctx, nonce);

        let nonce_bytes = nonce.to_big_endian();
        let hash_direct = qpow_math::get_nonce_hash(header, nonce_bytes);

        assert_eq!(hash_ctx, hash_direct);
    }

    #[test]
    fn test_mine_range_functionality() {
        let header = [5u8; 32];
        let difficulty = U512::from(1u64); // Very easy
        let ctx = JobContext::new(header, difficulty);

        let start_nonce = U512::from(1u64);
        let result = mine_nonce_range(&ctx, start_nonce, 10);

        // With very easy difficulty, should find a solution quickly
        if let Some((found_nonce, found_hash)) = result {
            assert!(found_nonce >= start_nonce);
            assert!(found_nonce < start_nonce + U512::from(10u64));
            assert!(found_hash < ctx.target);
        }
        // If no solution found, that's also valid behavior
    }

    #[test]
    fn test_hard_difficulty_no_solution() {
        let header = [6u8; 32];
        let very_hard_difficulty = U512::MAX; // Impossible difficulty
        let ctx = JobContext::new(header, very_hard_difficulty);

        let start_nonce = U512::from(1u64);
        let result = mine_nonce_range(&ctx, start_nonce, 5);

        // With impossible difficulty, should not find solution
        assert!(result.is_none());
    }

    #[test]
    fn test_midstate_resumes_to_full_hash() {
        let header = [7u8; 32];
        let nonce = (U512::from(0xdeadbeefcafeu64) << 300) | U512::from(0x1234567890abcdefu64);
        let nonce_be = nonce.to_big_endian();

        let mut state =
            mining_midstate(header, nonce_be[..32].try_into().unwrap()).map(Goldilocks::from_u64);
        let poseidon2 = Poseidon2::new();
        for (i, chunk) in nonce_be[32..].chunks_exact(4).enumerate() {
            state[i] += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
        }
        poseidon2.permute_mut(&mut state);
        state[0] += Goldilocks::ONE;
        state[1] += Goldilocks::ONE;
        poseidon2.permute_mut(&mut state);

        let mut hash = [0u8; 64];
        for i in 0..4 {
            hash[i * 8..(i + 1) * 8].copy_from_slice(&state[i].as_canonical_u64().to_le_bytes());
        }
        poseidon2.permute_mut(&mut state);
        for i in 0..4 {
            hash[32 + i * 8..32 + (i + 1) * 8]
                .copy_from_slice(&state[i].as_canonical_u64().to_le_bytes());
        }

        assert_eq!(
            U512::from_big_endian(&hash),
            qpow_math::get_nonce_hash(header, nonce_be)
        );
    }

    fn decode32(s: &str) -> [u8; 32] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn decode64(s: &str) -> [u8; 64] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    #[test]
    fn nonce_hash_golden_vectors() {
        for (i, v) in NONCE_HASH_KVS.iter().enumerate() {
            let header = decode32(v.header);
            let nonce_be = decode64(v.nonce);
            let want_hash = decode64(v.hash);
            let got = qpow_math::get_nonce_hash(header, nonce_be);
            assert_eq!(
                got.to_big_endian(),
                want_hash,
                "kv {i}: get_nonce_hash mismatch"
            );
            let high: [u8; 32] = nonce_be[..32].try_into().unwrap();
            assert_eq!(
                mining_midstate(header, high),
                v.mid,
                "kv {i}: midstate mismatch"
            );
            assert_eq!(
                hash_from_nonce(
                    &JobContext::new(header, U512::from(1u64)),
                    U512::from_big_endian(&nonce_be)
                ),
                got
            );
        }
    }

    #[test]
    fn u512_limb_roundtrip() {
        let n = (U512::from(0xdeadbeefcafeu64) << 300) | U512::from(0x1234567890abcdefu64);
        assert_eq!(u512_from_le_u32s(u512_to_le_u32s(n)), n);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_zero_difficulty_panics() {
        // Zero difficulty causes division by zero in target calculation.
        // Callers MUST validate difficulty > 0 before calling JobContext::new.
        let header = [7u8; 32];
        let zero_difficulty = U512::zero();
        let _ctx = JobContext::new(header, zero_difficulty);
    }
}
