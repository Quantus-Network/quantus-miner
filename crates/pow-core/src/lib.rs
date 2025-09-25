#![cfg_attr(not(feature = "std"), no_std)]

// Local QPoW math core with a compatibility API mirroring the original `qpow-math` crate,
// plus new scaffolding for an optimized path (precompute + incremental evaluation).
//
// This crate intentionally provides:
// - `compat` API: Drop-in functions like `is_valid_nonce` and `get_nonce_distance`
// - `JobContext`: Precomputed constants (m, n, target, threshold) for a given header
// - Incremental helpers: `init_worker_y0`, `step_mul`, and `distance_from_y`
//   to enable replacing per-nonce exponentiation with a single modular multiplication.
//
// Notes:
// - Current implementation mirrors the reference algorithm using BigUint-based modular arithmetic.
// - Future work will gate accelerated paths (e.g., Montgomery, SIMD Poseidon2) behind features.

extern crate alloc;

use core::ops::BitXor;
use primitive_types::U512;
use qp_poseidon_core::Poseidon2Core;

#[cfg(feature = "std")]
use log::{debug, error};

pub mod compat {
    //! Compatibility layer that mirrors the original `qpow-math` crate API.

    use super::*;

    /// Check QPoW validity for a given `header` and `nonce` against `threshold`.
    ///
    /// Returns a boolean indicating validity. If you also need the computed distance,
    /// use `is_valid_nonce_with_distance`.
    pub fn is_valid_nonce(header: [u8; 32], nonce: [u8; 64], threshold: U512) -> bool {
        let (ok, _) = is_valid_nonce_with_distance(header, nonce, threshold);
        ok
    }

    /// Same as `is_valid_nonce`, but also returns the computed distance (U512).
    pub fn is_valid_nonce_with_distance(
        header: [u8; 32],
        nonce: [u8; 64],
        threshold: U512,
    ) -> (bool, U512) {
        if nonce == [0u8; 64] {
            #[cfg(feature = "std")]
            error!(
                "is_valid_nonce should not be called with 0 nonce, but was for header: {header:?}"
            );
            return (false, U512::zero());
        }

        let distance_achieved = get_nonce_distance(header, nonce);
        #[cfg(feature = "std")]
        debug!(target: "pow-core", "distance = {distance_achieved}..., threshold = {threshold}...");

        (distance_achieved <= threshold, distance_achieved)
    }

    /// Compute the QPoW distance for (header, nonce).
    ///
    /// distance = target XOR H(m^(h + nonce) mod n)
    /// where (m, n) are derived deterministically from the header, and H is Poseidon2-512.
    pub fn get_nonce_distance(header: [u8; 32], nonce: [u8; 64]) -> U512 {
        super::get_nonce_distance_impl(header, nonce)
    }

    /// Generate a pair (m, n) deterministically from the header.
    pub fn get_random_rsa(header: &[u8; 32]) -> (U512, U512) {
        super::get_random_rsa_impl(header)
    }

    /// Check if two numbers are coprime using the Euclidean algorithm.
    pub fn is_coprime(a: &U512, b: &U512) -> bool {
        super::is_coprime_impl(a, b)
    }

    /// Miller–Rabin primality test used by `get_random_rsa`.
    pub fn is_prime(n: &U512) -> bool {
        super::is_prime_impl(n)
    }

    /// Apply the reference "hash-to-group" function then Poseidon2-512.
    pub fn hash_to_group_bigint_sha(h: &U512, m: &U512, n: &U512, solution: &U512) -> U512 {
        super::hash_to_group_bigint_sha_impl(h, m, n, solution)
    }

    /// Reference hash-to-group function: computes m^(h + solution) mod n.
    pub fn hash_to_group_bigint(h: &U512, m: &U512, n: &U512, solution: &U512) -> U512 {
        super::hash_to_group_bigint_impl(h, m, n, solution)
    }

    /// Reference modular exponentiation via BigUint.
    pub fn mod_pow(base: &U512, exponent: &U512, modulus: &U512) -> U512 {
        super::mod_pow_impl(base, exponent, modulus)
    }

    /// Poseidon2-512 over the big-endian bytes of input U512.
    pub fn poseidon2_512(input: U512) -> U512 {
        super::poseidon2_512_impl(input)
    }
}

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
        let (m, n) = get_random_rsa_impl(&header);
        let target = hash_to_group_bigint_sha_impl(&header_int, &m, &n, &U512::zero());
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
    mod_pow_impl(&ctx.m, &sum, &ctx.n)
}

/// Advance y by one nonce: y <- y * m (mod n).
pub fn step_mul(ctx: &JobContext, y: U512) -> U512 {
    mod_mul_impl(&y, &ctx.m, &ctx.n)
}

/// Compute distance for the current y:
/// distance = target XOR Poseidon2_512(y)
pub fn distance_from_y(ctx: &JobContext, y: U512) -> U512 {
    let hashed = poseidon2_512_impl(y);
    ctx.target.bitxor(hashed)
}

/// Convenience: compute distance for an arbitrary nonce using the context.
pub fn distance_for_nonce(ctx: &JobContext, nonce: U512) -> U512 {
    let y = mod_pow_impl(&ctx.m, &ctx.header_int.saturating_add(nonce), &ctx.n);
    distance_from_y(ctx, y)
}

/// Convenience: check if a distance is valid under the context's threshold.
pub fn is_valid_distance(ctx: &JobContext, distance: U512) -> bool {
    distance <= ctx.threshold
}

/// Reference distance computation used by the compat layer.
fn get_nonce_distance_impl(header: [u8; 32], nonce: [u8; 64]) -> U512 {
    if nonce == [0u8; 64] {
        #[cfg(feature = "std")]
        debug!(target: "pow-core", "zero nonce");
        return U512::zero();
    }

    let (m, n) = get_random_rsa_impl(&header);
    let header_int = U512::from_big_endian(&header);
    let nonce_int = U512::from_big_endian(&nonce);

    let target = hash_to_group_bigint_sha_impl(&header_int, &m, &n, &U512::zero());
    let nonce_element = hash_to_group_bigint_sha_impl(&header_int, &m, &n, &nonce_int);

    let distance = target.bitxor(nonce_element);
    #[cfg(feature = "std")]
    debug!(target: "pow-core", "distance = {distance}");
    distance
}

/// Generates a pair of RSA-style numbers (m, n) deterministically from input header.
///
/// - m: 256-bit derived via Poseidon2-256(header)
/// - n: 512-bit derived via Poseidon2-512(header), iteratively rehashed until valid:
///   (odd, composite, coprime with m, and n > m)
fn get_random_rsa_impl(header: &[u8; 32]) -> (U512, U512) {
    let poseidon = Poseidon2Core::new();

    // m from Poseidon2-256
    let m_bytes = poseidon.hash_no_pad_bytes(header);
    let m = U512::from_big_endian(&m_bytes);

    // initial n from Poseidon2-512
    let mut n_bytes = poseidon.hash_512(&m_bytes);
    let mut n = U512::from_big_endian(&n_bytes);

    // Keep hashing until n satisfies constraints
    while n % 2u32 == U512::zero() || n <= m || !is_coprime_impl(&m, &n) || is_prime_impl(&n) {
        n_bytes = poseidon.hash_512(&n_bytes);
        n = U512::from_big_endian(&n_bytes);
    }

    (m, n)
}

/// Check if two numbers are coprime using the Euclidean algorithm.
fn is_coprime_impl(a: &U512, b: &U512) -> bool {
    let mut x = *a;
    let mut y = *b;

    while y != U512::zero() {
        let tmp = y;
        y = x % y;
        x = tmp;
    }

    x == U512::one()
}

/// Hash-to-group then Poseidon2-512.
///
/// Note: The reference calls `hash_to_group_bigint` followed by an additional Poseidon2-512.
fn hash_to_group_bigint_sha_impl(h: &U512, m: &U512, n: &U512, solution: &U512) -> U512 {
    let result = hash_to_group_bigint_impl(h, m, n, solution);
    poseidon2_512_impl(result)
}

/// Reference hash-to-group big-integer function (no chunk splitting).
/// Computes sum = h + solution; then y = m^sum mod n.
fn hash_to_group_bigint_impl(h: &U512, m: &U512, n: &U512, solution: &U512) -> U512 {
    let sum = h.saturating_add(*solution);
    mod_pow_impl(m, &sum, n)
}

/// Reference modular exponentiation using BigUint square-and-multiply.
fn mod_pow_impl(base: &U512, exponent: &U512, modulus: &U512) -> U512 {
    if *modulus == U512::zero() {
        panic!("Modulus cannot be zero");
    }

    use num_bigint::BigUint;
    use num_traits::{One, Zero};

    // Convert inputs to BigUint
    let mut base = BigUint::from_bytes_be(&base.to_big_endian());
    let mut exp = BigUint::from_bytes_be(&exponent.to_big_endian());
    let modulus = BigUint::from_bytes_be(&modulus.to_big_endian());

    // Initialize result as 1
    let mut result = BigUint::one();

    // Square-and-multiply algorithm
    while !exp.is_zero() {
        if exp.bit(0) {
            result = (result * &base) % &modulus;
        }
        base = (&base * &base) % &modulus;
        exp >>= 1;
    }

    U512::from_big_endian(&result.to_bytes_be())
}

/// Reference modular multiplication using BigUint, i.e., (a * b) mod n.
fn mod_mul_impl(a: &U512, b: &U512, modulus: &U512) -> U512 {
    use num_bigint::BigUint;

    if *modulus == U512::zero() {
        panic!("Modulus cannot be zero");
    }

    let a_bi = BigUint::from_bytes_be(&a.to_big_endian());
    let b_bi = BigUint::from_bytes_be(&b.to_big_endian());
    let n_bi = BigUint::from_bytes_be(&modulus.to_big_endian());

    let prod = (a_bi * b_bi) % n_bi;
    U512::from_big_endian(&prod.to_bytes_be())
}

/// Miller–Rabin primality test.
///
/// Deterministically selects k=32 bases hashed from `n` using Poseidon2-512 to
/// bound false-positive probability to ~1/2^64 for composites.
fn is_prime_impl(n: &U512) -> bool {
    if *n <= U512::one() {
        return false;
    }
    if *n == U512::from(2u32) || *n == U512::from(3u32) {
        return true;
    }
    if *n % U512::from(2u32) == U512::zero() {
        return false;
    }

    // write n-1 as d * 2^r
    let mut d = *n - U512::one();
    let mut r = 0u32;
    while d % U512::from(2u32) == U512::zero() {
        d /= U512::from(2u32);
        r += 1;
    }

    // Generate test bases deterministically from n using Poseidon2
    let mut bases = [U512::zero(); 32];
    let mut base_count = 0;
    let poseidon = Poseidon2Core::new();
    let mut counter = U512::zero();

    while base_count < 32 {
        // Hash n concatenated with counter
        let mut bytes = [0u8; 128];
        let n_bytes = n.to_big_endian();
        let counter_bytes = counter.to_big_endian();

        bytes[..64].copy_from_slice(&n_bytes);
        bytes[64..128].copy_from_slice(&counter_bytes);

        let poseidon_bytes = poseidon.hash_512(&bytes);

        // Use the hash to generate a base in [2, n-2]
        let hash = U512::from_big_endian(&poseidon_bytes);
        let base = (hash % (*n - U512::from(4u32))) + U512::from(2u32);
        bases[base_count] = base;
        base_count += 1;

        counter += U512::one();
    }

    'witness: for base in bases {
        let mut x = mod_pow_impl(&base, &d, n);

        if x == U512::one() || x == *n - U512::one() {
            continue 'witness;
        }

        // Square r-1 times
        for _ in 0..r - 1 {
            x = mod_pow_impl(&x, &U512::from(2u32), n);
            if x == *n - U512::one() {
                continue 'witness;
            }
            if x == U512::one() {
                return false;
            }
        }
        return false;
    }

    true
}

/// Poseidon2-512 over the big-endian bytes of input `U512`.
fn poseidon2_512_impl(input: U512) -> U512 {
    let poseidon = Poseidon2Core::new();
    let bytes = input.to_big_endian();
    U512::from_big_endian(&poseidon.hash_512(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    // removed unused helper u512_from_hex

    #[test]
    fn compat_distance_matches_context_distance() {
        // Synthetic header and nonce
        let header = [1u8; 32];
        let nonce = [2u8; 64];

        let dist_compat = compat::get_nonce_distance(header, nonce);

        let ctx = JobContext::new(header, U512::from(123u64));
        let nonce_int = U512::from_big_endian(&nonce);
        let dist_ctx = distance_for_nonce(&ctx, nonce_int);

        assert_eq!(dist_compat, dist_ctx);
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

        let direct = mod_pow_impl(
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
        let d = compat::get_nonce_distance(header, nonce);
        assert_eq!(d, U512::zero());
    }
}
