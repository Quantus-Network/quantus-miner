# engine-montgomery

Montgomery-optimized CPU mining engine for the Quantus External Miner.

This crate implements the same public `MinerEngine` trait as the baseline and fast CPU engines, while replacing the per-nonce modular multiplication with a fixed-width 8×64-bit Montgomery implementation. It is selectable via the CLI flag:

- `--engine cpu-montgomery`

The engine mirrors `cpu-fast` control flow and metrics emission so that apples-to-apples comparisons can be made between engines.

Highlights:
- Fixed-width 512-bit arithmetic using 8×64-bit limbs.
- Portable CIOS (Coarsely Integrated Operand Scanning) Montgomery multiply with `u128` intermediates.
- Direct Poseidon2 over big-endian limbs to avoid intermediate big-integer conversions.
- Per-job precompute cache (Montgomery params and `m_hat`) to reduce setup overhead.
- Runtime backend selection for microarchitecture-optimized kernels (x86_64 BMI2-only and BMI2+ADX; aarch64 UMULH).
- Metrics label for backend selection to aid dashboards and A/B analysis.

---

## Algorithm Overview

QPoW distance per nonce is computed as:

1) y update (group accumulation):
   - y0 = m^(h + start_nonce) mod n (one-time per worker).
   - y_{k+1} = (y_k * m) mod n (one multiply per nonce).

2) Distance = target XOR Poseidon2_512(y), where y is encoded as 64 big-endian bytes.

The engine focuses on optimizing step (1) with Montgomery multiplication and reducing overhead in step (2) by hashing directly over the final big-endian representation of y (no extra big-int intermediates).

---

## Montgomery Arithmetic (512-bit, 8×64 limbs)

We implement a 512-bit Montgomery field backed by 8 little-endian 64-bit limbs. For modulus `n` (odd composite, constrained by pow-core), we precompute:

- `n0_inv = -n^{-1} mod 2^64` (using Newton–Raphson).
- `R = 2^(64*8) mod n` (implicit via representation).
- `R^2 mod n` (computed once per job using the existing big-integer reference).
- `to_mont(x) = x * R mod n = mont_mul(x, R^2)`.
- `from_mont(x̂) = x̂ * 1 mod n = mont_mul(x̂, 1)`.

The core multiply is a portable CIOS Montgomery reduction:

```
- acc <- 0
- For each limb i in a:
    acc += a[i] * b
    m = (acc[0] * n0_inv) mod 2^64
    acc += m * n
    acc = acc >> 64 (shift down one limb)
- If acc >= n: acc -= n
- return acc
```

Where `acc` is a 9-limb `u128` accumulator to simplify carries.

This yields a single multiply+reduce per nonce with a branchless inner loop (except final conditional subtract).

---

## Hashing Strategy

- We keep `y` in Montgomery domain during iteration to minimize transforms.
- Before Poseidon2, we convert the residue `ŷ` back to the normal domain via `mont_mul(ŷ, 1)` and serialize as big-endian 64 bytes.
- We reuse a single `Poseidon2Core` hasher per search call and `hash_512()` each iteration to reduce construction overhead.

This preserves consensus behavior while avoiding unnecessary big-int allocations or conversions.

---

## Backend Selection and Optimizations

At runtime, the engine selects a Montgomery multiply backend based on CPU and environment:

- Portable (default/fallback)
  - `mont_mul_portable`: pure-Rust, `u128`-based CIOS.
  - Available everywhere.

- x86_64 (runtime detected)
  - BMI2-only (`_mulx_u64`):
    - `mont_mul_bmi2`: uses BMI2 MULX to get 128-bit products efficiently.
    - Single carry chain (easier to validate; broadly available on newer CPUs).
  - BMI2+ADX:
    - `mont_mul_bmi2_adx`: implemented using MULX + ADCX/ADOX dual carry chains for higher ILP.

- aarch64
  - UMULH/ADCS:
    - `mont_mul_aarch64`: implemented using UMULH for high halves and ADCS-style accumulation via 64-bit ops to reduce dependency on `u128` where beneficial.
    - Default backend on Apple Silicon/macOS and Linux ARM64.

You can override backend selection for testing:

- `MINER_MONT_BACKEND=portable|bmi2|bmi2-adx|umulh`

The engine logs the selected backend (and exports it via metrics) at job start. Unsupported overrides safely fall back with a clear warning.

---

## Metrics

The engine emits the same per-job and per-thread metrics as other engines, but adds a backend info gauge:

- `miner_engine_backend{engine="cpu-montgomery", backend="<name>"} = 1`

This makes it easy to pivot in Grafana by backend.

All other metrics (hash rates, progress chunking cadence, counters) are identical to `cpu-fast` for apples-to-apples comparisons.

---

## Correctness and Tests

We keep the portable CIOS path as the ground-truth reference for optimized kernels, and we cross-check against pow-core’s BigUint-based implementations.

Property tests included:

- Portable Montgomery vs reference incremental multiply:
  - `from_mont(mul(to_mont(y), to_mont(m))) == step_mul(y)` across multiple steps.

- BMI2 vs Portable:
  - For randomized sequences, `bmi2` backend must match `portable` exactly at each step.
  - On non-x86_64 platforms, the test still runs but both backends fall back to `portable`.

- aarch64 UMULH vs Portable:
  - For randomized sequences, `aarch64-umulh` must match `portable` exactly at each step.
  - On non-aarch64 platforms, both tags fall back to `portable`.

- End-to-end parity:
  - `cpu-montgomery` vs `cpu-fast` on a small inclusive range (distance and winner parity; identical hash_count accounting).

We recommend running the property tests on machines with and without BMI2/ADX and on aarch64 to cover all optimized code paths.

---

## Safety

- The crate uses `#![deny(unsafe_code)]`.
- `unsafe` is scoped only to tiny backend functions:
  - x86_64: `_mulx_u64` and inline asm for ADCX/ADOX dual carry chains.
  - aarch64: restricted intrinsics (e.g., UMULH) behind a small boundary.
- All other code remains safe Rust.
- The portable path is always available as a fallback for correctness/regression checks.

---

## Performance Notes

- Relative gains depend on how much time the miner spends in Poseidon2 vs modular multiply.
- Direct-hash-from-residue + precompute caching already yields a measurable uplift over `cpu-fast`.
- The BMI2-only path should improve throughput on supporting x86_64 hardware.
- BMI2+ADX typically produces the highest gains on the multiply itself (often 1.5–2.0×), with end-to-end uplift bounded by Poseidon2 share per nonce.
- aarch64 UMULH/ADCS brings similar relative gains on Apple Silicon and other ARM64 platforms.

To minimize orchestration overhead in the service:
- Increase `--progress-chunk-ms` (e.g., 3000–5000) on both engines when comparing, to reduce update traffic and context switching.

---

## Runtime and CI Tips

- Selecting the engine:
  - `quantus-miner --engine cpu-montgomery`

- For A/B:
  - Keep workers and chunking identical across instances.
  - If testing backends explicitly:
    - `MINER_MONT_BACKEND=bmi2` or `bmi2-adx` on capable x86_64 hardware
    - `MINER_MONT_BACKEND=umulh` on aarch64
    - Check logs and metrics for the selected backend label.

- Observability:
  - Ensure metrics exporter is enabled (`--metrics-port ...`) for dashboards.
  - Filter or group by `engine="cpu-montgomery"` and backend metric to compare microarchitectural paths.

---

## Roadmap

- [x] Portable 8×64 CIOS (u128)
- [x] Per-job precompute cache (`n0_inv`, `R^2 mod n`, `m_hat`)
- [x] Direct Poseidon2 from normalized big-endian bytes
- [x] Backend selection with log + metric
- [x] x86_64 BMI2-only MULX kernel
- [x] x86_64 BMI2+ADX (MULX + ADCX/ADOX) dual carry chain
- [x] aarch64 UMULH/ADCS kernel (macOS/Linux ARM64)
- [ ] Optional benchmark micro-harness (ns/op for mont_mul backends)
- [ ] Extend tests with more randomized vectors and edge-case sweeps

---

## Design Rationale

- Keep the interface and metrics identical to `cpu-fast` so that any performance deltas reflect algorithmic/microarchitectural improvements rather than service overhead.
- Keep a portable, well-reviewed core (CIOS) as a correctness reference.
- Add microarchitecture-optimized kernels behind runtime dispatch and an env override for safe, controlled rollouts.
- Log and export backend selection so A/B comparisons and regressions are easy to track.

---

## Contributing

- Changes to backends should include:
  - Property tests vs portable.
  - End-to-end parity checks vs `cpu-fast`.
  - A note in this README describing the optimization and any preconditions (e.g., required CPU features).
- Keep unsafe code minimal, private, and well-commented.
- Prefer small, focused PRs for each backend/optimization to simplify review and bisecting.
