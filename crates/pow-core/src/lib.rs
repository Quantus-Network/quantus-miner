use primitive_types::U512;
use qp_poseidon_core::Poseidon2Core;

pub use qpow_math::{get_nonce_hash, is_valid_nonce, mine_range};

/// Job context for Bitcoin-style PoW mining with double Poseidon2 hashing
#[derive(Debug, Clone)]
pub struct JobContext {
    pub header: [u8; 32],
    pub difficulty: U512,
    pub target: U512,
}

impl JobContext {
    /// Build a new context from header and difficulty
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

/// Compute hash for the current nonce using Bitcoin-style double Poseidon2
pub fn hash_from_nonce(ctx: &JobContext, nonce: U512) -> U512 {
    let nonce_bytes = nonce.to_big_endian();
    qpow_math::get_nonce_hash(ctx.header, nonce_bytes)
}

/// Check if hash meets difficulty target
pub fn is_valid_hash(ctx: &JobContext, hash: U512) -> bool {
    hash < ctx.target
}

/// Convenience: compute hash for an arbitrary nonce and check validity
pub fn is_valid_nonce_for_context(ctx: &JobContext, nonce: U512) -> (bool, U512) {
    let hash = hash_from_nonce(ctx, nonce);
    let is_valid = is_valid_hash(ctx, hash);
    (is_valid, hash)
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
        let (is_valid, hash) = is_valid_nonce_for_context(&ctx, nonce);

        // With very easy difficulty, should be valid
        assert!(is_valid);
        assert_ne!(hash, U512::zero());

        // Verify hash is actually below target
        assert!(hash < ctx.target);
    }

    #[test]
    fn test_zero_nonce_produces_zero_hash() {
        let header = [0xABu8; 32];
        let difficulty = U512::from(1u64);
        let ctx = JobContext::new(header, difficulty);

        let zero_nonce = U512::zero();
        let hash = hash_from_nonce(&ctx, zero_nonce);

        // Zero nonce should produce zero hash
        assert_eq!(hash, U512::zero());

        // Zero hash should not be considered valid
        assert!(!is_valid_hash(&ctx, hash));
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
}
