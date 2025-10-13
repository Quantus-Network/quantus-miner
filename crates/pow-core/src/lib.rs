#![cfg_attr(not(feature = "std"), no_std)]

//! Local QPoW math core that delegates to the existing `qpow-math` crate,
//! while providing additional scaffolding for optimized mining paths.
//!
//! This crate provides:
//! - Direct re-exports from `qpow-math` for compatibility
//! - `JobContext`: Precomputed constants for optimized mining
//! - Incremental helpers: `init_worker_y0`, `step_mul`, and `distance_from_y`
//!   to enable replacing per-nonce exponentiation with single modular multiplications.

extern crate alloc;

use core::ops::BitXor;
use primitive_types::U512;
use qp_poseidon_core::Poseidon2Core;

// Re-export commonly used functions from qpow-math for compatibility
pub use qpow_math::{
    get_nonce_distance, get_random_rsa, hash_to_group_bigint, hash_to_group_bigint_poseidon,
    is_coprime, is_prime, is_valid_nonce, mod_pow, mod_pow_next,
};

/// Precomputed context for a single mining job (header + threshold).
///
/// This enables an optimized path:
/// - Precompute (m, n) and target = H(m^(h + 0) mod n) once per job.
/// - For each worker, compute `y0 = m^(h + start_nonce) mod n` once.
/// - For each subsequent nonce, update y = (y * m) mod n (O(1) per step).
/// - Distance at a step is `target XOR H(y)`.
#[derive(Clone, Debug)]
pub struct JobContext {
    pub header: [u8; 32],
    pub header_int: U512,
    pub threshold: U512,

    pub m: U512,
    pub n: U512,
    pub target: U512,
}

impl JobContext {
    /// Build a new context by deriving (m, n) and target from the header.
    pub fn new(header: [u8; 32], threshold: U512) -> Self {
        let header_int = U512::from_big_endian(&header);
        let (m, n) = qpow_math::get_random_rsa(&header);
        let target = qpow_math::hash_to_group_bigint_poseidon(&header_int, &m, &n, &U512::zero());
        JobContext {
            header,
            header_int,
            threshold,
            m,
            n,
            target,
        }
    }
}

/// Compute y0 = m^(h + start_nonce) mod n for a worker's starting nonce.
///
/// This is the one-time exponentiation cost per worker/thread. Subsequent steps can
/// use `step_mul` to advance y with a single modular multiplication.
pub fn init_worker_y0(ctx: &JobContext, start_nonce: U512) -> U512 {
    let sum = ctx.header_int.saturating_add(start_nonce);
    qpow_math::mod_pow(&ctx.m, &sum, &ctx.n)
}

/// Advance y by one nonce: y <- y * m (mod n).
pub fn step_mul(ctx: &JobContext, y: U512) -> U512 {
    qpow_math::mod_pow_next(&y, &ctx.m, &ctx.n)
}

/// Compute distance for the current y: distance = target XOR Poseidon2_512(y)
pub fn distance_from_y(ctx: &JobContext, y: U512) -> U512 {
    let poseidon = Poseidon2Core::new();
    let hashed = U512::from_big_endian(&poseidon.hash_512(&y.to_big_endian()));
    ctx.target.bitxor(hashed)
}

/// Convenience: compute distance for an arbitrary nonce using the context.
pub fn distance_for_nonce(ctx: &JobContext, nonce: U512) -> U512 {
    let y = qpow_math::mod_pow(&ctx.m, &ctx.header_int.saturating_add(nonce), &ctx.n);
    distance_from_y(ctx, y)
}

/// Convenience: check if a distance is valid under the context's threshold.
pub fn is_valid_distance(ctx: &JobContext, distance: U512) -> bool {
    distance <= ctx.threshold
}

/// Poseidon2-512 over the big-endian bytes of input U512.
pub fn poseidon2_512(input: U512) -> U512 {
    let poseidon = Poseidon2Core::new();
    let bytes = input.to_big_endian();
    U512::from_big_endian(&poseidon.hash_512(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_matches_qpow_math() {
        // Test that our JobContext produces the same results as qpow_math directly
        let header = [1u8; 32];
        let nonce = [2u8; 64];

        let dist_qpow = qpow_math::get_nonce_distance(header, nonce);

        let ctx = JobContext::new(header, U512::from(123u64));
        let nonce_int = U512::from_big_endian(&nonce);
        let dist_ctx = distance_for_nonce(&ctx, nonce_int);

        assert_eq!(dist_qpow, dist_ctx);
    }

    #[test]
    fn incremental_step_matches_pow_plus_one() {
        let header = [3u8; 32];
        let threshold = U512::from(99999u64);
        let ctx = JobContext::new(header, threshold);

        let start = U512::from(1000u64);
        let y0 = init_worker_y0(&ctx, start);

        // Distance for start+1 computed two ways should match:
        // 1) Incremental: step once from y0
        // 2) Direct exponentiation with nonce = start+1
        let y1_inc = step_mul(&ctx, y0);
        let dist_inc = distance_from_y(&ctx, y1_inc);

        let direct = qpow_math::mod_pow(
            &ctx.m,
            &ctx.header_int.saturating_add(start + U512::one()),
            &ctx.n,
        );
        let dist_direct = distance_from_y(&ctx, direct);

        assert_eq!(dist_inc, dist_direct);
    }

    #[test]
    fn zero_nonce_is_zero_distance() {
        let header = [0xABu8; 32];
        let nonce = [0u8; 64];
        let d = qpow_math::get_nonce_distance(header, nonce);
        assert_eq!(d, U512::zero());
    }

    #[test]
    fn poseidon2_512_works() {
        let input = U512::from(12345u64);
        let result = poseidon2_512(input);
        assert_ne!(result, U512::zero());
        assert_ne!(result, input);
    }
}
