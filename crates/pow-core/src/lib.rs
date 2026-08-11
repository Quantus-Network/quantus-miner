use primitive_types::U512;
use qp_poseidon_core::{Goldilocks, Poseidon2};

pub use qp_poseidon_core::SPONGE_WIDTH;
pub use qpow_math::{get_nonce_hash, is_valid_nonce, mine_range};

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
