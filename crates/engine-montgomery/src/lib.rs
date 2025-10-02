#![deny(rust_2018_idioms)]
#![deny(unsafe_code)]

//! Montgomery-optimized CPU mining engine (scaffolding).
//!
//! Note to maintainers of this file:
//! The ADX/BMI2 Montgomery path uses inline-asm. To eliminate rare post-shift
//! mismatches caused by flag-edge ordering, we must export acc[8] and the OF/CF
//! carry bits into Rust locals via memory operands and perform the final fold
//! and the shift in Rust, not in asm. The edits below implement exactly that.
//!
//! Goals:
//! - Mirror the cpu-fast engine behavior and metrics (hash counts, progress cadence).
//! - Provide a drop-in engine selectable via `--engine cpu-montgomery`.
//! - Introduce a crypto-bigint based 512-bit fixed-width backend scaffold for future
//!   Montgomery multiplication/reduction.
//!
//! Current state:
//! - The search loop mirrors `engine-cpu`'s `FastCpuEngine` to ensure apples-for-apples
//!   metrics and correctness parity.
//! - A lightweight Montgomery scaffolding is included (conversions and parameter
//!   container) using `crypto-bigint`, ready to be integrated into the hot path.
//!
//! Next steps (planned):
//! - Replace the per-step `step_mul` with Montgomery domain multiplication:
//!   y_hat <- montgomery_mul(y_hat, m_hat, n, n') with y kept in Montgomery domain.
//! - Precompute `R`, `R^2`, and `n'` once per job/thread and transform inputs.
//! - Convert out of Montgomery before Poseidon2-512 distance computation.
//!
//! Important: We intentionally keep emissions (hash_count increments, control flow)
//! identical to `cpu-fast` so metrics are directly comparable when pitting
//! `--engine cpu-fast` vs `--engine cpu-montgomery`.

use core::cmp::Ordering;

use engine_cpu::EngineStatus;
use engine_cpu::{EngineCandidate as Candidate, EngineRange as Range, MinerEngine};
use pow_core::compat;
use pow_core::{init_worker_y0, is_valid_distance, JobContext};
use primitive_types::U512;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

/// Montgomery engine for CPU.
pub struct MontgomeryCpuEngine {
    #[allow(clippy::type_complexity)]
    cache: Mutex<HashMap<([u8; 64], [u8; 64]), Arc<mont_portable::MontCtx>>>,
}

impl Default for MontgomeryCpuEngine {
    fn default() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl MontgomeryCpuEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MinerEngine for MontgomeryCpuEngine {
    fn name(&self) -> &'static str {
        "cpu-montgomery"
    }

    fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext {
        // Build the standard pow-core context (m, n, target, etc).
        // Montgomery parameters are computed per search call for now; we can
        // later thread them through a custom context wrapper if needed.
        JobContext::new(header_hash, threshold)
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus {
        // NOTE: For initial integration we mirror the cpu-fast engine logic to ensure
        // metrics parity and correctness, while keeping the Montgomery scaffolding
        // ready for optimization in follow-ups.

        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        // One-time init per thread: y0 = m^(h + start_nonce) mod n
        let mut current = range.start;
        let y0 = init_worker_y0(ctx, current);
        let mut hash_count: u64 = 0;

        // Initialize or reuse per-job Montgomery params and residues from cache
        let mont = {
            let key = (ctx.m.to_big_endian(), ctx.n.to_big_endian());
            let mut guard = self.cache.lock().unwrap();
            if let Some(m) = guard.get(&key) {
                m.clone()
            } else {
                let m = Arc::new(mont_portable::MontCtx::from_ctx(ctx));
                guard.insert(key, m.clone());
                m
            }
        };
        let mut y_hat = mont.to_mont_u512(&y0);
        let m_hat = mont.m_hat;

        loop {
            // Cancellation check (fast and frequent as in cpu-fast)
            if cancel.load(AtomicOrdering::Relaxed) {
                return EngineStatus::Cancelled { hash_count };
            }

            // Compute distance from Montgomery accumulator: normalize then hash via pow-core
            let y_norm = mont.from_mont_u512(&y_hat);
            let distance = pow_core::distance_from_y(ctx, y_norm);
            hash_count = hash_count.saturating_add(1);

            if is_valid_distance(ctx, distance) {
                let work = current.to_big_endian();
                return EngineStatus::Found {
                    candidate: Candidate {
                        nonce: current,
                        work,
                        distance,
                    },
                    hash_count,
                    origin: engine_cpu::FoundOrigin::Cpu,
                };
            }

            // Advance or finish
            match current.cmp(&range.end) {
                Ordering::Less => {
                    // Incremental step in Montgomery domain: y_hat <- y_hat * m_hat (mod n)
                    y_hat = mont.mul(&y_hat, &m_hat);
                    current = current.saturating_add(U512::one());
                }
                _ => {
                    break EngineStatus::Exhausted { hash_count };
                }
            }
        }
    }
}

/// Montgomery scaffolding with crypto-bigint.
/// This module provides conversions and parameter containers that we can use
/// to wire up a fixed-width limb backend for 512-bit operations.
mod mont_portable {
    use super::*;

    // Montgomery context with portable CIOS 8x64 implementation (u128 intermediates).
    // Limbs are stored little-endian (limb 0 is least significant).
    type MulFn = fn(&[u64; 8], &[u64; 8], &[u64; 8], u64) -> [u64; 8];

    #[derive(Clone)]
    pub struct MontCtx {
        n: [u64; 8],
        n0_inv: u64,  // -n^{-1} mod 2^64
        r2: [u64; 8], // R^2 mod n
        pub m_hat: [u64; 8],
        mul_fn: MulFn,
    }

    impl MontCtx {
        pub fn from_ctx(ctx: &JobContext) -> Self {
            let n = u512_to_le(ctx.n);
            let n0_inv = mont_n0_inv(n[0]);
            let r2_u512 = compat::mod_pow(&U512::from(2u32), &U512::from(1024u32), &ctx.n);
            let r2 = u512_to_le(r2_u512);
            let m = u512_to_le(ctx.m);
            let (mul_fn, backend) = select_backend();
            log::info!(target: "miner", "cpu-montgomery backend selected: {backend}");
            #[cfg(feature = "metrics")]
            {
                metrics::set_engine_backend("cpu-montgomery", backend);
            }
            let m_hat = mul_fn(&m, &r2, &n, n0_inv);
            MontCtx {
                n,
                n0_inv,
                r2,
                m_hat,
                mul_fn,
            }
        }

        pub fn to_mont_u512(&self, x: &U512) -> [u64; 8] {
            let xl = u512_to_le(*x);
            (self.mul_fn)(&xl, &self.r2, &self.n, self.n0_inv)
        }

        #[allow(clippy::wrong_self_convention)]
        pub fn from_mont_u512(&self, x_hat: &[u64; 8]) -> U512 {
            let one = {
                let mut o = [0u64; 8];
                o[0] = 1;
                o
            };
            let norm_le = (self.mul_fn)(x_hat, &one, &self.n, self.n0_inv);
            let norm_be = le_to_be_bytes(&norm_le);
            U512::from_big_endian(&norm_be)
        }

        pub fn mul(&self, a_hat: &[u64; 8], b_hat: &[u64; 8]) -> [u64; 8] {
            (self.mul_fn)(a_hat, b_hat, &self.n, self.n0_inv)
        }

        // Test-only helpers to enable forcing a specific backend and to access limb-level conversions.
        // These are useful for property tests and backend A/B validations.
        #[cfg(test)]
        pub fn from_ctx_with_backend_tag(ctx: &JobContext, tag: &str) -> Self {
            let n = u512_to_le(ctx.n);
            let n0_inv = mont_n0_inv(n[0]);
            let r2_u512 = compat::mod_pow(&U512::from(2u32), &U512::from(1024u32), &ctx.n);
            let r2 = u512_to_le(r2_u512);
            let m = u512_to_le(ctx.m);

            // Choose mul_fn by tag; fall back to portable when not applicable or CPU features missing.
            let (mul_fn, _backend): (MulFn, &'static str) = match tag {
                "x86_64-bmi2-adx" | "bmi2-adx" => {
                    #[cfg(target_arch = "x86_64")]
                    {
                        if std::is_x86_feature_detected!("bmi2") {
                            // Temporarily route ADX to BMI2 for correctness while ADX refactor completes
                            (mont_mul_bmi2, "x86_64-bmi2")
                        } else {
                            (mont_mul_portable, "portable")
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        (mont_mul_portable, "portable")
                    }
                }
                "x86_64-bmi2" | "bmi2" => {
                    #[cfg(target_arch = "x86_64")]
                    {
                        if std::is_x86_feature_detected!("bmi2") {
                            (mont_mul_bmi2, "x86_64-bmi2")
                        } else {
                            (mont_mul_portable, "portable")
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        (mont_mul_portable, "portable")
                    }
                }
                "aarch64-umulh" | "umulh" => {
                    #[cfg(target_arch = "aarch64")]
                    {
                        (mont_mul_aarch64, "aarch64-umulh")
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        (mont_mul_portable, "portable")
                    }
                }
                _ => (mont_mul_portable, "portable"),
            };

            let m_hat = mul_fn(&m, &r2, &n, n0_inv);
            MontCtx {
                n,
                n0_inv,
                r2,
                m_hat,
                mul_fn,
            }
        }

        #[cfg(test)]
        pub fn to_mont_le_limbs(&self, x: &U512) -> [u64; 8] {
            (self.mul_fn)(&u512_to_le(*x), &self.r2, &self.n, self.n0_inv)
        }

        #[cfg(test)]
        #[allow(clippy::wrong_self_convention)]
        pub fn from_mont_le_limbs(&self, x_hat: &[u64; 8]) -> [u64; 8] {
            let one = {
                let mut o = [0u64; 8];
                o[0] = 1;
                o
            };
            (self.mul_fn)(x_hat, &one, &self.n, self.n0_inv)
        }
    }

    #[inline]
    fn u512_to_le(x: U512) -> [u64; 8] {
        let be = x.to_big_endian();
        let mut limbs = [0u64; 8];
        // Split BE into 8 chunks, then reverse to get LE limb order (least-significant first).
        for i in 0..8 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&be[i * 8..(i + 1) * 8]);
            limbs[i] = u64::from_be_bytes(bytes);
        }
        limbs.reverse();
        limbs
    }

    #[inline]
    fn le_to_be_bytes(limbs: &[u64; 8]) -> [u8; 64] {
        let mut out = [0u8; 64];
        for i in 0..8 {
            let chunk = limbs[7 - i].to_be_bytes();
            out[i * 8..(i + 1) * 8].copy_from_slice(&chunk);
        }
        out
    }

    // Compute n0_inv = -n[0]^{-1} mod 2^64 using Newton–Raphson (n[0] must be odd).
    #[inline]
    fn mont_n0_inv(n0: u64) -> u64 {
        // Compute inverse of n0 modulo 2^64
        let mut x = 1u64;
        // 6 iterations suffice for 64-bit modulus
        for _ in 0..6 {
            let t = x.wrapping_mul(n0);
            x = x.wrapping_mul(2u64.wrapping_sub(t));
        }
        x.wrapping_neg()
    }

    // Portable CIOS Montgomery multiplication: returns (a * b * R^{-1}) mod n
    #[inline]
    fn mont_mul_portable(a: &[u64; 8], b: &[u64; 8], n: &[u64; 8], n0_inv: u64) -> [u64; 8] {
        const MASK: u128 = 0xFFFF_FFFF_FFFF_FFFFu128;
        let mut acc = [0u128; 9];

        for &ai_u64 in a.iter().take(8) {
            // acc += ai * b
            let ai = ai_u64 as u128;
            let mut carry = 0u128;
            for j in 0..8 {
                let sum = acc[j] + ai * (b[j] as u128) + carry;
                acc[j] = sum & MASK;
                carry = sum >> 64;
            }
            acc[8] += carry;

            // m = (acc[0] * n0_inv) mod 2^64
            let m = ((acc[0] as u64).wrapping_mul(n0_inv)) as u128;

            // acc += m * n
            let mut carry2 = 0u128;
            for j in 0..8 {
                let sum = acc[j] + m * (n[j] as u128) + carry2;
                acc[j] = sum & MASK;
                carry2 = sum >> 64;
            }
            acc[8] += carry2;

            // shift acc right by one limb
            for j in 0..8 {
                acc[j] = acc[j + 1];
            }
            acc[8] = 0;
        }

        // Convert acc (little-endian limbs) to u64 array
        let mut res = [0u64; 8];
        for j in 0..8 {
            res[j] = acc[j] as u64;
        }

        // Conditional subtract modulus if res >= n
        if ge_le(&res, n) {
            sub_le_in_place(&mut res, n);
        }

        res
    }

    #[inline]
    fn ge_le(a: &[u64; 8], b: &[u64; 8]) -> bool {
        for i in (0..8).rev() {
            if a[i] != b[i] {
                return a[i] > b[i];
            }
        }
        true
    }

    #[inline]
    fn sub_le_in_place(a: &mut [u64; 8], b: &[u64; 8]) {
        let mut borrow: u128 = 0;
        for i in 0..8 {
            let ai = a[i] as u128;
            let bi = b[i] as u128;
            let tmp = (1u128 << 64) + ai - bi - borrow;
            a[i] = (tmp & 0xFFFF_FFFF_FFFF_FFFFu128) as u64;
            borrow = if tmp >> 64 == 0 { 1 } else { 0 };
        }
    }

    #[inline]
    fn select_backend() -> (MulFn, &'static str) {
        // Optional override via env:
        // MINER_MONT_BACKEND=portable|bmi2|bmi2-adx
        if let Ok(val) = std::env::var("MINER_MONT_BACKEND") {
            let forced = val.to_ascii_lowercase();

            #[cfg(target_arch = "x86_64")]
            {
                let bmi2 = std::is_x86_feature_detected!("bmi2");

                match forced.as_str() {
                    "portable" => {
                        log::warn!(target: "miner", "cpu-montgomery backend override: forced portable");
                        return (mont_mul_portable, "forced-portable");
                    }
                    "bmi2-adx" | "adx" => {
                        if bmi2 {
                            log::warn!(target: "miner", "cpu-montgomery backend override: forced x86_64-bmi2-adx");
                            return (mont_mul_bmi2_adx, "forced-x86_64-bmi2-adx");
                        } else {
                            log::warn!(target: "miner", "cpu-montgomery backend override requested bmi2-adx but BMI2 unavailable; falling back to x86_64-generic");
                            return (mont_mul_portable, "x86_64-generic");
                        }
                    }
                    "bmi2" => {
                        if bmi2 {
                            log::warn!(target: "miner", "cpu-montgomery backend override: forced x86_64-bmi2");
                            return (mont_mul_bmi2, "forced-x86_64-bmi2");
                        } else {
                            log::warn!(target: "miner", "cpu-montgomery backend override requested bmi2 but BMI2 unavailable; falling back to x86_64-generic");
                            return (mont_mul_portable, "x86_64-generic");
                        }
                    }
                    other => {
                        log::warn!(target: "miner", "cpu-montgomery backend override '{other}' is not recognized on x86_64; using auto-detect");
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                match forced.as_str() {
                    "portable" => {
                        log::warn!(target: "miner", "cpu-montgomery backend override: forced portable");
                        return (mont_mul_portable, "forced-portable");
                    }
                    // x86-only hints on aarch64 -> warn and ignore
                    "bmi2" | "bmi2-adx" | "adx" => {
                        log::warn!(target: "miner", "cpu-montgomery backend override '{}' not supported on aarch64; using auto-detect", forced);
                    }
                    other => {
                        log::warn!(target: "miner", "cpu-montgomery backend override '{}' is not recognized on aarch64; using auto-detect", other);
                    }
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                match forced.as_str() {
                    "portable" => {
                        log::warn!(target: "miner", "cpu-montgomery backend override: forced portable");
                        return (mont_mul_portable, "forced-portable");
                    }
                    other => {
                        log::warn!(target: "miner", "cpu-montgomery backend override '{}' not supported on this arch; using auto-detect", other);
                    }
                }
            }
        }

        // Auto-detection (default) per architecture
        #[cfg(target_arch = "x86_64")]
        {
            let bmi2 = std::is_x86_feature_detected!("bmi2");

            if bmi2 {
                (mont_mul_bmi2, "x86_64-bmi2")
            } else {
                (mont_mul_portable, "x86_64-generic")
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            (mont_mul_aarch64, "aarch64-umulh")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            (mont_mul_portable, "portable")
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[allow(unsafe_code)]
    fn mont_mul_bmi2(a: &[u64; 8], b: &[u64; 8], n: &[u64; 8], n0_inv: u64) -> [u64; 8] {
        // BMI2-optimized CIOS using MULX. Single carry chain with u128 accumulation.
        // Limbs: little-endian (limb 0 = least significant).
        use core::arch::x86_64::_mulx_u64;
        const MASK: u128 = 0xFFFF_FFFF_FFFF_FFFFu128;

        // 9-limb accumulator in u128 to simplify carries
        let mut acc = [0u128; 9];

        for &ai in a.iter().take(8) {
            // acc += ai * b
            let mut carry: u128 = 0;
            for j in 0..8 {
                let mut hi: u64 = 0;
                // lo = (ai * b[j])_lo, hi = (ai * b[j])_hi
                let lo = unsafe { _mulx_u64(ai, b[j], &mut hi) };
                let sum = acc[j] + (lo as u128) + carry;
                acc[j] = sum & MASK;
                carry = (sum >> 64) + (hi as u128);
            }
            acc[8] += carry;

            // m = (acc[0] * n0_inv) mod 2^64
            let m = (acc[0] as u64).wrapping_mul(n0_inv);

            // acc += m * n
            let mut carry2: u128 = 0;
            for j in 0..8 {
                let mut hi2: u64 = 0;
                let lo2 = unsafe { _mulx_u64(m, n[j], &mut hi2) };
                let sum2 = acc[j] + (lo2 as u128) + carry2;
                acc[j] = sum2 & MASK;
                carry2 = (sum2 >> 64) + (hi2 as u128);
            }
            acc[8] += carry2;

            // shift acc right by one limb (drop acc[0])
            for j in 0..8 {
                acc[j] = acc[j + 1];
            }
            acc[8] = 0;
        }

        // Convert acc to u64 limbs (little-endian)
        let mut res = [0u64; 8];
        for j in 0..8 {
            res[j] = acc[j] as u64;
        }

        // Conditional subtraction: if res >= n then res -= n
        if ge_le(&res, n) {
            sub_le_in_place(&mut res, n);
        }

        res
    }

    #[cfg(all(target_arch = "x86_64", feature = "adx-trace"))]
    #[inline]
    #[allow(unsafe_code)]
    #[allow(unused_variables, unused_mut)]
    fn mont_mul_bmi2_adx(a: &[u64; 8], b: &[u64; 8], n: &[u64; 8], n0_inv: u64) -> [u64; 8] {
        // Runtime feature check: fallback to BMI2 if ADX is not available.
        if !std::is_x86_feature_detected!("adx") {
            return mont_mul_bmi2(a, b, n, n0_inv);
        }

        // ADX refactor: two phases per iteration with Rust-side fold+shift.
        // Keep accumulator as u64 limbs to allow direct asm load/store.
        let mut acc: [u64; 9] = [0; 9];
        let acc_ptr = acc.as_mut_ptr();
        let b_ptr = b.as_ptr();
        let n_ptr = n.as_ptr();

        for i in 0..8 {
            let ai = a[i];

            // Phase A: acc += ai * b (MULX + ADCX/ADOX), export acc8/of/cf to Rust
            let mut acc8_a: u64;
            let mut of_a: u64;
            let mut cf_a: u64;
            unsafe {
                core::arch::asm!(
                    // rdx = ai for MULX
                    "mov rdx, {ai}",

                    // Clear both carry chains
                    "xor r8d, r8d",
                    "adcx r8, r8",
                    "adox r8, r8",

                    // j = 0..7 with dual chains (CF->ADCX, OF->ADOX)
                    "mulx r9, r10, qword ptr [r13 + 0]",
                    "mov r11, qword ptr [r12 + 0]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 0], r11",
                    "mov r11, qword ptr [r12 + 8]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 8], r11",

                    "mulx r9, r10, qword ptr [r13 + 8]",
                    "mov r11, qword ptr [r12 + 8]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 8], r11",
                    "mov r11, qword ptr [r12 + 16]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 16], r11",

                    "mulx r9, r10, qword ptr [r13 + 16]",
                    "mov r11, qword ptr [r12 + 16]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 16], r11",
                    "mov r11, qword ptr [r12 + 24]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 24], r11",

                    "mulx r9, r10, qword ptr [r13 + 24]",
                    "mov r11, qword ptr [r12 + 24]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 24], r11",
                    "mov r11, qword ptr [r12 + 32]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 32], r11",

                    "mulx r9, r10, qword ptr [r13 + 32]",
                    "mov r11, qword ptr [r12 + 32]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 32], r11",
                    "mov r11, qword ptr [r12 + 40]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 40], r11",

                    "mulx r9, r10, qword ptr [r13 + 40]",
                    "mov r11, qword ptr [r12 + 40]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 40], r11",
                    "mov r11, qword ptr [r12 + 48]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 48], r11",

                    "mulx r9, r10, qword ptr [r13 + 48]",
                    "mov r11, qword ptr [r12 + 48]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 48], r11",
                    "mov r11, qword ptr [r12 + 56]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 56], r11",

                    "mulx r9, r10, qword ptr [r13 + 56]",
                    "mov r11, qword ptr [r12 + 56]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 56], r11",
                    "mov r11, qword ptr [r12 + 64]",
                    "adox r11, r9",

                    // Export acc8 and flags
                    "seto  al",
                    "setc  dl",
                    "mov   {acc8_out}, r11",
                    "movzx {of_out},  al",
                    "movzx {cf_out},  dl",

                    ai         = in(reg) ai,
                    in("r12") acc_ptr,
                    in("r13") b_ptr,
                    acc8_out   = lateout(reg) acc8_a,
                    of_out     = lateout(reg) of_a,
                    cf_out     = lateout(reg) cf_a,
                    out("r8") _, out("r9") _, out("r10") _, out("r11") _,
                    out("rax") _, out("rdx") _,
                    options(nostack)
                );
            }

            // Rust fold for Phase A
            acc[8] = acc8_a
                .wrapping_add((of_a & 1) as u64)
                .wrapping_add((cf_a & 1) as u64);

            // ---------------------------
            // Phase B: m = acc[0] * n0_inv; acc += m * n
            // ---------------------------
            let m = acc[0].wrapping_mul(n0_inv);
            let mut acc8_b: u64;
            let mut of_b: u64;
            let mut cf_b: u64;
            unsafe {
                core::arch::asm!(
                    // Place m in rdx for MULX
                    "mov rdx, {m}",

                    // Clear both carry chains
                    "xor r8d, r8d",
                    "adcx r8, r8",
                    "adox r8, r8",

                    // j = 0..7
                    "mulx r9, r10, qword ptr [r14 + 0]",
                    "mov r11, qword ptr [r12 + 0]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 0], r11",
                    "mov r11, qword ptr [r12 + 8]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 8], r11",

                    "mulx r9, r10, qword ptr [r14 + 8]",
                    "mov r11, qword ptr [r12 + 8]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 8], r11",
                    "mov r11, qword ptr [r12 + 16]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 16], r11",

                    "mulx r9, r10, qword ptr [r14 + 16]",
                    "mov r11, qword ptr [r12 + 16]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 16], r11",
                    "mov r11, qword ptr [r12 + 24]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 24], r11",

                    "mulx r9, r10, qword ptr [r14 + 24]",
                    "mov r11, qword ptr [r12 + 24]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 24], r11",
                    "mov r11, qword ptr [r12 + 32]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 32], r11",

                    "mulx r9, r10, qword ptr [r14 + 32]",
                    "mov r11, qword ptr [r12 + 32]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 32], r11",
                    "mov r11, qword ptr [r12 + 40]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 40], r11",

                    "mulx r9, r10, qword ptr [r14 + 40]",
                    "mov r11, qword ptr [r12 + 40]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 40], r11",
                    "mov r11, qword ptr [r12 + 48]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 48], r11",

                    "mulx r9, r10, qword ptr [r14 + 48]",
                    "mov r11, qword ptr [r12 + 48]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 48], r11",
                    "mov r11, qword ptr [r12 + 56]",
                    "adox r11, r9",
                    "mov qword ptr [r12 + 56], r11",

                    "mulx r9, r10, qword ptr [r14 + 56]",
                    "mov r11, qword ptr [r12 + 56]",
                    "adcx r11, r10",
                    "mov qword ptr [r12 + 56], r11",
                    "mov r11, qword ptr [r12 + 64]",
                    "adox r11, r9",

                    // Export acc8 and flags
                    "seto  al",
                    "setc  dl",
                    "mov   {acc8_out}, r11",
                    "movzx {of_out},  al",
                    "movzx {cf_out},  dl",

                    in("r12") acc_ptr,
                    in("r14") n_ptr,
                    m          = in(reg) m,
                    acc8_out   = lateout(reg) acc8_b,
                    of_out     = lateout(reg) of_b,
                    cf_out     = lateout(reg) cf_b,
                    out("r8") _, out("r9") _, out("r10") _, out("r11") _,
                    lateout("rax") _,
                    options(nostack)
                );
            }

            // Rust fold for Phase B
            acc[8] = acc8_b
                .wrapping_add((of_b & 1) as u64)
                .wrapping_add((cf_b & 1) as u64);

            // Shift
            acc.copy_within(1..=8, 0);
            acc[8] = 0;
        }

        // Conditional subtraction
        let mut res = [0u64; 8];
        res.copy_from_slice(&acc[0..8]);
        if ge_le(&res, n) {
            sub_le_in_place(&mut res, n);
        }
        res
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[allow(unsafe_code)]
    fn mont_mul_bmi2_adx_states(
        a: &[u64; 8],
        b: &[u64; 8],
        n: &[u64; 8],
        n0_inv: u64,
    ) -> [[u64; 8]; 8] {
        use core::arch::x86_64::_mulx_u64;
        const MASK: u128 = 0xFFFF_FFFF_FFFF_FFFFu128;

        let mut acc = [0u128; 9];
        let mut states = [[0u64; 8]; 8];

        for i in 0..8 {
            let ai = a[i];

            // Phase A
            let mut carry: u128 = 0;
            for j in 0..8 {
                let mut hi: u64 = 0;
                let lo = unsafe { _mulx_u64(ai, b[j], &mut hi) };
                let sum = acc[j] + (lo as u128) + carry;
                acc[j] = sum & MASK;
                carry = (sum >> 64) + (hi as u128);
            }
            acc[8] = acc[8].wrapping_add(carry);

            // Phase B
            let m = (acc[0] as u64).wrapping_mul(n0_inv);
            let mut carry2: u128 = 0;
            for j in 0..8 {
                let mut hi2: u64 = 0;
                let lo2 = unsafe { _mulx_u64(m, n[j], &mut hi2) };
                let sum2 = acc[j] + (lo2 as u128) + carry2;
                acc[j] = sum2 & MASK;
                carry2 = (sum2 >> 64) + (hi2 as u128);
            }
            acc[8] = acc[8].wrapping_add(carry2);

            // Shift
            for j in 0..8 {
                acc[j] = acc[j + 1];
            }
            acc[8] = 0;

            // Record state (big-endian)
            for j in 0..8 {
                states[i][j] = acc[7 - j] as u64;
            }
        }

        states
    }

    #[cfg(all(target_arch = "x86_64", feature = "adx-trace"))]
    #[inline]
    #[allow(unsafe_code)]
    fn mont_mul_bmi2_adx_boundaries_single(
        a: &[u64; 8],
        b: &[u64; 8],
        n: &[u64; 8],
        n0_inv: u64,
        iter: usize,
    ) -> ([u64; 8], [u64; 8]) {
        use core::arch::x86_64::_mulx_u64;
        const MASK: u128 = 0xFFFF_FFFF_FFFF_FFFFu128;

        debug_assert!(iter < 8);
        let mut acc = [0u128; 9];

        // Replay to start of iter
        for i in 0..iter {
            let ai = a[i];
            // Phase A
            let mut carry: u128 = 0;
            for j in 0..8 {
                let mut hi: u64 = 0;
                let lo = unsafe { _mulx_u64(ai, b[j], &mut hi) };
                let sum = acc[j] + (lo as u128) + carry;
                acc[j] = sum & MASK;
                carry = (sum >> 64) + (hi as u128);
            }
            acc[8] = acc[8].wrapping_add(carry);
            // Phase B
            let m = (acc[0] as u64).wrapping_mul(n0_inv);
            let mut carry2: u128 = 0;
            for j in 0..8 {
                let mut hi2: u64 = 0;
                let lo2 = unsafe { _mulx_u64(m, n[j], &mut hi2) };
                let sum2 = acc[j] + (lo2 as u128) + carry2;
                acc[j] = sum2 & MASK;
                carry2 = (sum2 >> 64) + (hi2 as u128);
            }
            acc[8] = acc[8].wrapping_add(carry2);
            // Shift
            for j in 0..8 {
                acc[j] = acc[j + 1];
            }
            acc[8] = 0;
        }

        // mul boundary at iter (before shift)
        let ai = a[iter];
        let mut acc_mul = acc;
        let mut carry_m: u128 = 0;
        for j in 0..8 {
            let mut hi: u64 = 0;
            let lo = unsafe { _mulx_u64(ai, b[j], &mut hi) };
            let sum = acc_mul[j] + (lo as u128) + carry_m;
            acc_mul[j] = sum & MASK;
            carry_m = (sum >> 64) + (hi as u128);
        }
        acc_mul[8] = acc_mul[8].wrapping_add(carry_m);
        let mut mul_be = [0u64; 8];
        for j in 0..8 {
            mul_be[7 - j] = acc_mul[j] as u64;
        }

        // red boundary at iter (before shift)
        let m = (acc_mul[0] as u64).wrapping_mul(n0_inv);
        let mut acc_red = acc_mul;
        let mut carry_r: u128 = 0;
        for j in 0..8 {
            let mut hi2: u64 = 0;
            let lo2 = unsafe { _mulx_u64(m, n[j], &mut hi2) };
            let sum2 = acc_red[j] + (lo2 as u128) + carry_r;
            acc_red[j] = sum2 & MASK;
            carry_r = (sum2 >> 64) + (hi2 as u128);
        }
        acc_red[8] = acc_red[8].wrapping_add(carry_r);
        let mut red_be = [0u64; 8];
        for j in 0..8 {
            red_be[7 - j] = acc_red[j] as u64;
        }

        (mul_be, red_be)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    #[allow(unsafe_code)]
    fn mont_mul_aarch64(a: &[u64; 8], b: &[u64; 8], n: &[u64; 8], n0_inv: u64) -> [u64; 8] {
        const MASK: u128 = 0xFFFF_FFFF_FFFF_FFFFu128;
        let mut acc = [0u128; 9];
        for &ai in a.iter().take(8) {
            // acc += ai * b
            let mut carry: u128 = 0;
            for j in 0..8 {
                // low 64-bit product
                let lo = ai.wrapping_mul(b[j]);
                // high 64-bit product via UMULH intrinsic
                let hi = ((ai as u128) * (b[j] as u128)) >> 64;
                let sum = acc[j] + (lo as u128) + carry;
                acc[j] = sum & MASK;
                carry = (sum >> 64) + hi;
            }
            acc[8] += carry;

            // m = (acc[0] * n0_inv) mod 2^64
            let m = (acc[0] as u64).wrapping_mul(n0_inv);

            // acc += m * n
            let mut carry2: u128 = 0;
            for j in 0..8 {
                let lo2 = m.wrapping_mul(n[j]);
                let hi2 = ((m as u128) * (n[j] as u128)) >> 64;
                let sum2 = acc[j] + (lo2 as u128) + carry2;
                acc[j] = sum2 & MASK;
                carry2 = (sum2 >> 64) + hi2;
            }
            acc[8] += carry2;

            // shift acc right by one limb
            for j in 0..8 {
                acc[j] = acc[j + 1];
            }
            acc[8] = 0;
        }

        // Convert acc to result limbs
        let mut res = [0u64; 8];
        for j in 0..8 {
            res[j] = acc[j] as u64;
        }

        // Conditional subtraction: if res >= n then res -= n
        if ge_le(&res, n) {
            sub_le_in_place(&mut res, n);
        }
        res
    }

    #[cfg(test)]
    mod prop_tests {
        use super::*;
        use crate::MontgomeryCpuEngine;
        use engine_cpu::{EngineStatus, FastCpuEngine, MinerEngine};
        use pow_core::{init_worker_y0, step_mul, JobContext};
        use primitive_types::U512;
        use std::sync::atomic::AtomicBool;

        fn u64x8_le_to_u512(le: &[u64; 8]) -> U512 {
            let be = le_to_be_bytes(le);
            U512::from_big_endian(&be)
        }

        fn make_ctx_with_header_byte(byte: u8) -> JobContext {
            let mut header = [0u8; 32];
            header.fill(byte);
            let threshold = U512::MAX;
            JobContext::new(header, threshold)
        }

        #[test]
        fn montgomery_portable_mul_matches_step_mul() {
            // Validate that mont_mul agrees with pow_core's step_mul across many steps.
            let ctx = make_ctx_with_header_byte(0x5Au8);
            let mont = MontCtx::from_ctx_with_backend_tag(&ctx, "portable");

            let start = U512::from(12345u64);
            let mut y_ref = init_worker_y0(&ctx, start);
            let mut y_hat = mont.to_mont_le_limbs(&y_ref);
            let m_hat = mont.m_hat;

            // Walk 128 steps, comparing at each step
            for _ in 0..128 {
                // Reference path: y <- y * m (mod n) via pow_core BigUint
                y_ref = step_mul(&ctx, y_ref);

                // Montgomery path: y_hat <- y_hat * m_hat (mod n), convert out
                y_hat = mont.mul(&y_hat, &m_hat);
                let y_hat_norm_le = mont.from_mont_le_limbs(&y_hat);
                let y_mont = u64x8_le_to_u512(&y_hat_norm_le);

                assert_eq!(y_ref, y_mont, "montgomery mul mismatch vs step_mul");
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn montgomery_bmi2_equivalence_to_portable_when_available() {
            // On x86_64, ensure bmi2 path produces identical results as portable for the same (a,b).
            // On non-x86_64 this test still runs but both tags fall back to portable.
            let ctx = make_ctx_with_header_byte(0x3Cu8);
            let mont_port = MontCtx::from_ctx_with_backend_tag(&ctx, "portable");
            let mont_bmi2 = MontCtx::from_ctx_with_backend_tag(&ctx, "bmi2");

            let start = U512::from(999u64);
            let mut y_ref = init_worker_y0(&ctx, start);

            let mut y_hat_port = mont_port.to_mont_le_limbs(&y_ref);
            let mut y_hat_bmi2 = mont_bmi2.to_mont_le_limbs(&y_ref);

            let m_hat_port = mont_port.m_hat;
            let m_hat_bmi2 = mont_bmi2.m_hat;

            for _ in 0..64 {
                // advance reference so values change per-iteration
                y_ref = step_mul(&ctx, y_ref);

                y_hat_port = mont_port.mul(&y_hat_port, &m_hat_port);
                y_hat_bmi2 = mont_bmi2.mul(&y_hat_bmi2, &m_hat_bmi2);

                let y_port = u64x8_le_to_u512(&mont_port.from_mont_le_limbs(&y_hat_port));
                let y_bmi2 = u64x8_le_to_u512(&mont_bmi2.from_mont_le_limbs(&y_hat_bmi2));

                assert_eq!(y_port, y_bmi2, "bmi2 path mismatch with portable");
            }
        }

        #[test]
        fn engine_end_to_end_matches_cpu_fast_on_small_range() {
            // End-to-end parity check against cpu-fast over a small inclusive range.
            let header = [0x11u8; 32];
            let threshold = U512::MAX;
            let ctx = JobContext::new(header, threshold);

            let range = crate::Range {
                start: U512::from(0u64),
                end: U512::from(500u64),
            };
            let cancel = AtomicBool::new(false);

            let mont = MontgomeryCpuEngine::new();
            let fast = FastCpuEngine::new();

            let s_m = mont.search_range(&ctx, range.clone(), &cancel);
            let s_f = fast.search_range(&ctx, range.clone(), &cancel);

            match (s_m, s_f) {
                (
                    EngineStatus::Found {
                        candidate: cm,
                        hash_count: hm,
                        origin: _,
                    },
                    EngineStatus::Found {
                        candidate: cf,
                        hash_count: hf,
                        origin: _,
                    },
                ) => {
                    assert_eq!(cm.nonce, cf.nonce, "nonce mismatch");
                    assert_eq!(cm.distance, cf.distance, "distance mismatch");
                    assert_eq!(hm, hf, "hash_count mismatch");
                }
                (
                    EngineStatus::Exhausted { hash_count: hm },
                    EngineStatus::Exhausted { hash_count: hf },
                ) => {
                    assert_eq!(hm, hf, "hash_count mismatch on Exhausted");
                }
                (m, f) => panic!("expected matching status, got mont={m:?}, fast={f:?}"),
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[test]
        fn montgomery_aarch64_equivalence_to_portable_when_available() {
            // On aarch64, ensure UMULH/ADCS path matches portable. On other arches this test
            // still runs but both tags fall back to portable.
            let ctx = make_ctx_with_header_byte(0x77u8);
            let mont_port = MontCtx::from_ctx_with_backend_tag(&ctx, "portable");
            let mont_arm = MontCtx::from_ctx_with_backend_tag(&ctx, "aarch64-umulh");

            let start = U512::from(4242u64);
            let mut y_ref = init_worker_y0(&ctx, start);

            let mut y_hat_port = mont_port.to_mont_le_limbs(&y_ref);
            let mut y_hat_arm = mont_arm.to_mont_le_limbs(&y_ref);

            let m_hat_port = mont_port.m_hat;
            let m_hat_arm = mont_arm.m_hat;

            for _ in 0..64 {
                // advance reference so values change per-iteration
                y_ref = step_mul(&ctx, y_ref);

                y_hat_port = mont_port.mul(&y_hat_port, &m_hat_port);
                y_hat_arm = mont_arm.mul(&y_hat_arm, &m_hat_arm);

                let y_port = u64x8_le_to_u512(&mont_port.from_mont_le_limbs(&y_hat_port));
                let y_arm = u64x8_le_to_u512(&mont_arm.from_mont_le_limbs(&y_hat_arm));

                assert_eq!(y_port, y_arm, "aarch64 umulh path mismatch with portable");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn make_ctx() -> JobContext {
        let header = [1u8; 32];
        let threshold = U512::MAX; // permissive threshold for "found" parity test
        JobContext::new(header, threshold)
    }

    #[test]
    fn montgomery_engine_matches_fast_engine_on_small_range() {
        let ctx = make_ctx();

        let range = Range {
            start: U512::from(0u64),
            end: U512::from(100u64),
        };

        let cancel = AtomicBool::new(false);

        let mont = MontgomeryCpuEngine::new();
        let fast = engine_cpu::FastCpuEngine::new();

        let m_status = mont.search_range(&ctx, range.clone(), &cancel);
        let f_status = fast.search_range(&ctx, range.clone(), &cancel);

        match (m_status, f_status) {
            (
                EngineStatus::Found {
                    candidate: m_cand,
                    hash_count: m_hashes,
                    origin: _,
                },
                EngineStatus::Found {
                    candidate: f_cand,
                    hash_count: f_hashes,
                    origin: _,
                },
            ) => {
                assert_eq!(
                    m_cand.nonce, f_cand.nonce,
                    "engines disagreed on winning nonce"
                );
                assert_eq!(
                    m_cand.distance, f_cand.distance,
                    "engines disagreed on distance"
                );
                assert_eq!(m_hashes, f_hashes, "engines disagreed on hash_count");
            }
            (m, f) => panic!("expected Found/Found, got montgomery={m:?}, fast={f:?}"),
        }
    }

    #[test]
    fn engine_returns_exhausted_when_no_solution_in_range() {
        // Very strict threshold to avoid solutions in a tiny range.
        let header = [2u8; 32];
        let threshold = U512::zero();
        let ctx = JobContext::new(header, threshold);

        let range = Range {
            start: U512::from(1u64),
            end: U512::from(1000u64),
        };

        let cancel = AtomicBool::new(false);
        let eng = MontgomeryCpuEngine::new();

        let status = eng.search_range(&ctx, range.clone(), &cancel);
        match status {
            EngineStatus::Exhausted { hash_count } => {
                // Inclusive range length = end - start + 1
                let expected = (range.end - range.start + U512::one()).as_u64();
                assert_eq!(hash_count, expected, "hash_count should equal range length");
            }
            other => panic!("expected Exhausted, got {other:?}"),
        }
    }

    #[test]
    fn engine_respects_immediate_cancellation() {
        let ctx = make_ctx();
        let range = Range {
            start: U512::from(0u64),
            end: U512::from(1_000_000u64),
        };
        let cancel = AtomicBool::new(true); // cancelled before starting
        let eng = MontgomeryCpuEngine::new();

        let status = eng.search_range(&ctx, range, &cancel);
        match status {
            EngineStatus::Cancelled { hash_count } => {
                assert_eq!(hash_count, 0, "expected no work when cancelled immediately");
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }
}
