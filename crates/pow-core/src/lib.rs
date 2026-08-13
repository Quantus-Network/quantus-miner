use primitive_types::U512;
use qp_poseidon_core::serialization::digest_to_bytes;
use qp_poseidon_core::{Goldilocks, Poseidon2, POSEIDON2_OUTPUT};

pub use qp_poseidon_core::SPONGE_WIDTH;
pub use qpow_math::{get_nonce_hash, is_valid_nonce, mine_range};

/// Absorb 32 bytes into the sponge rate as 8 little-endian u32 felts, matching
/// `bytes_to_felts_iter`'s 4-bytes-per-felt encoding.
fn absorb_32(state: &mut [Goldilocks; SPONGE_WIDTH], bytes: &[u8; 32]) {
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        state[i] += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
}

/// First 4 felts of the state as 32 bytes, matching the squeeze side of
/// `hash_squeeze_twice`.
fn squeeze_bytes(state: &[Goldilocks; SPONGE_WIDTH]) -> [u8; 32] {
    digest_to_bytes(state[..POSEIDON2_OUTPUT].try_into().unwrap())
}

/// Sponge state (canonical u64 felts) after absorbing the 32-byte header and the
/// high 32 bytes of the big-endian nonce — the first two of the five Poseidon2
/// permutations of `get_nonce_hash`. This state is identical for every nonce in a
/// batch as long as incrementing the nonce never carries into its high 256 bits,
/// so GPU kernels can resume the sponge from here and skip 2 of 5 permutations.
pub fn mining_midstate(header: [u8; 32], nonce_high_be: [u8; 32]) -> [u64; SPONGE_WIDTH] {
    let poseidon2 = Poseidon2::new();
    let mut state = [Goldilocks::ZERO; SPONGE_WIDTH];
    absorb_32(&mut state, &header);
    poseidon2.permute_mut(&mut state);
    absorb_32(&mut state, &nonce_high_be);
    poseidon2.permute_mut(&mut state);
    state.map(|g| g.as_canonical_u64())
}

/// CPU mining context: precomputed sponge midstate plus the comparison target
/// split for early rejection.
///
/// `get_nonce_hash` runs 5 permutations per nonce: absorb header (1), absorb
/// nonce-high (2), absorb nonce-low (3), padding block (4), second squeeze (5).
/// Permutations 1-2 are constant while the nonce's high 256 bits are unchanged,
/// so they are precomputed once here. Permutation 5 only affects the low half
/// of the big-endian U512 hash, so when the first squeeze already exceeds the
/// target's high half the nonce is rejected after permutation 4 — the common
/// case is 2 permutations per nonce instead of 5. Returned hashes remain
/// byte-identical to `get_nonce_hash`.
pub struct MidstateContext {
    poseidon2: Poseidon2,
    midstate: [u64; SPONGE_WIDTH],
    target: U512,
    /// High 32 bytes of the big-endian target; compared against the first squeeze.
    target_high: [u8; 32],
}

impl MidstateContext {
    pub fn new(header: [u8; 32], target: U512, nonce_high_be: [u8; 32]) -> Self {
        let target_be = target.to_big_endian();
        let mut target_high = [0u8; 32];
        target_high.copy_from_slice(&target_be[..32]);
        Self {
            poseidon2: Poseidon2::new(),
            midstate: mining_midstate(header, nonce_high_be),
            target,
            target_high,
        }
    }

    /// Sponge state just before the first squeeze: resume from the midstate,
    /// absorb the low 32 bytes of the nonce, then the padding block
    /// (terminator felt and finalization marker land on state[0]/state[1]).
    fn squeeze_state(&self, nonce_low_be: [u8; 32]) -> [Goldilocks; SPONGE_WIDTH] {
        let mut state = self.midstate.map(Goldilocks::from_u64);
        absorb_32(&mut state, &nonce_low_be);
        self.poseidon2.permute_mut(&mut state);
        state[0] += Goldilocks::ONE;
        state[1] += Goldilocks::ONE;
        self.poseidon2.permute_mut(&mut state);
        state
    }

    /// Run the second squeeze permutation, fill the low half, return the hash.
    fn second_squeeze(&self, state: &mut [Goldilocks; SPONGE_WIDTH], hash: &mut [u8; 64]) -> U512 {
        self.poseidon2.permute_mut(state);
        hash[32..].copy_from_slice(&squeeze_bytes(state));
        U512::from_big_endian(hash)
    }

    /// Full 64-byte nonce hash as a U512; identical to `qpow_math::get_nonce_hash`.
    pub fn hash_nonce_low(&self, nonce_low_be: [u8; 32]) -> U512 {
        let mut state = self.squeeze_state(nonce_low_be);
        let mut hash = [0u8; 64];
        hash[..32].copy_from_slice(&squeeze_bytes(&state));
        self.second_squeeze(&mut state, &mut hash)
    }

    /// Return the full hash when this nonce meets the target, else `None`.
    /// The second squeeze permutation runs only when the first squeeze
    /// (the high half of the BE hash) cannot settle the comparison alone.
    pub fn check_nonce_low(&self, nonce_low_be: [u8; 32]) -> Option<U512> {
        let mut state = self.squeeze_state(nonce_low_be);
        let mut hash = [0u8; 64];
        hash[..32].copy_from_slice(&squeeze_bytes(&state));
        // Big-endian comparison: a greater high half means the hash is out.
        if hash[..32] > self.target_high[..] {
            return None;
        }
        let h = self.second_squeeze(&mut state, &mut hash);
        (h < self.target).then_some(h)
    }
}

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
    fn test_midstate_context_matches_get_nonce_hash() {
        // Cross-check the fast path against the reference over a spread of
        // headers and nonces (including nonces with non-zero high halves; the
        // carry *transition* itself is exercised in engine-cpu). The non-uniform
        // header pins byte order and felt indexing in the absorb, which uniform
        // [N; 32] headers cannot distinguish.
        let headers = [
            [0u8; 32],
            [1u8; 32],
            [0xffu8; 32],
            core::array::from_fn(|i| (i as u8) ^ 0x5a),
        ];
        let nonces = [
            U512::zero(),
            U512::from(1u64),
            U512::from(123u64),
            U512::from(u64::MAX),
            (U512::from(0xdeadbeefu64) << 256) | U512::from(0x1234567890abcdefu64),
            U512::MAX,
        ];
        for header in headers {
            for nonce in nonces {
                let nonce_be = nonce.to_big_endian();
                let ctx =
                    MidstateContext::new(header, U512::MAX, nonce_be[..32].try_into().unwrap());
                let fast = ctx.hash_nonce_low(nonce_be[32..].try_into().unwrap());
                let reference = get_nonce_hash(header, nonce_be);
                assert_eq!(fast, reference, "hash mismatch for nonce {nonce}");
            }
        }
    }

    #[test]
    fn test_check_nonce_low_matches_full_compare() {
        // Fuzz-style cross-check: the early-reject path must agree with
        // hashing the full 64 bytes and comparing against the target.
        let difficulties = [
            U512::from(1u64), // always valid
            U512::from(2u64), // ~50% valid
            U512::from(1_000_000u64),
            U512::MAX, // never valid (target = 1)
        ];
        for header_seed in 0..4u8 {
            let header = [header_seed; 32];
            for &difficulty in &difficulties {
                let target = U512::MAX / difficulty;
                let ctx = MidstateContext::new(header, target, [0u8; 32]);
                let mut found = 0u64;
                for n in 0..500u64 {
                    let nonce_be = U512::from(n).to_big_endian();
                    let result = ctx.check_nonce_low(nonce_be[32..].try_into().unwrap());
                    let reference = get_nonce_hash(header, nonce_be);
                    assert_eq!(
                        result.is_some(),
                        reference < target,
                        "validity mismatch for header {header_seed}, nonce {n}, difficulty {difficulty}"
                    );
                    if let Some(h) = result {
                        assert_eq!(h, reference, "hash mismatch for nonce {n}");
                        found += 1;
                    }
                }
                if difficulty == U512::one() {
                    assert_eq!(found, 500, "difficulty 1 must accept every nonce");
                }
                if difficulty == U512::MAX {
                    assert_eq!(found, 0, "target 1 must reject every nonce");
                }
            }
        }
    }

    #[test]
    fn test_check_nonce_low_boundary_targets() {
        // Targets derived from a real hash, so target_high == hash_high by
        // construction — the only case where the second squeeze decides.
        // Pins the strict inequalities on both compares and the target split.
        let header = [9u8; 32];
        for n in 0..64u64 {
            let nonce_be = U512::from(n).to_big_endian();
            let hi = nonce_be[..32].try_into().unwrap();
            let lo = nonce_be[32..].try_into().unwrap();
            let h = get_nonce_hash(header, nonce_be);
            for target in [h - U512::one(), h, h + U512::one()] {
                let got = MidstateContext::new(header, target, hi).check_nonce_low(lo);
                assert_eq!(got.is_some(), h < target, "nonce {n} target {target:x}");
                if let Some(v) = got {
                    assert_eq!(v, h);
                }
            }
        }
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
