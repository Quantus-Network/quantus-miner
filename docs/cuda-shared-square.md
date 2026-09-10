# Experimental CUDA shared prestate and fused square

The `engine-cuda/experimental-shared-square` feature combines a shared 12-word
prestate, an inline PTX square/reduction sequence, and a 1024-thread mining block
with a 640-block cap. It is disabled by default. The diagnostic hash kernel keeps
its 256-thread launch. The shared-memory barrier precedes all per-thread exits.

This is an RTX 3080 Ti experiment, not an automatically selected device policy.
The fixed cap was evaluated on an 80-SM device; other devices need separate
validation before enabling or generalizing it. The square preserves the existing
three partial products and reducer wrap behavior. CPU candidate verification,
nonce traversal, and rejected-candidate continuation remain in place.

## Observations

Official v4.2.0 (`c1cf0a3`), RTX 3080 Ti 12GB, Ubuntu 22.04,
driver 580.95.05, NVRTC 12.1, Rust 1.95.0. Offline exhaustive searches,
32,000,000 nonces per batch, 64 warmups and 101 measured batches per process.
The A/B/B/A process medians were:

| Process | MH/s |
|---|---:|
| Baseline A1 | 371.756644 |
| Candidate B1 | 385.788302 |
| Candidate B2 | 382.356088 |
| Baseline A2 | 356.028211 |

Averaging process medians gives 363.892428 versus 384.072195 MH/s (+5.55%).
This is an observation, not a statistically established speedup: automatic GPU
clocks could not be locked, and baseline drift was substantial. Separate kernel
microbenchmarks observed +4.46%. Neither result establishes effective pool
shares, net earnings, or performance on other NVIDIA devices.

These measurements used the equivalent ungated prototype. After packaging it as
a Cargo feature, both default and enabled release CUDA test suites passed all
10 tests with CUDA required. The enabled all-targets check also passed.
Do not combine this feature with other unmeasured experiments.

## Reproduction

Require a real CUDA device instead of silently accepting skipped tests:

```sh
QUANTUS_REQUIRE_CUDA=1 cargo test --release -p engine-cuda -- --test-threads=1
QUANTUS_REQUIRE_CUDA=1 cargo test --release -p engine-cuda --features experimental-shared-square -- --test-threads=1
cargo build --release -p engine-cuda --example hashrate --target-dir target/cuda-base
cargo build --release -p engine-cuda --example hashrate --features experimental-shared-square --target-dir target/cuda-shared-square
target/cuda-base/release/examples/hashrate 32000000 101
target/cuda-shared-square/release/examples/hashrate 32000000 101
target/cuda-shared-square/release/examples/hashrate 32000000 101
target/cuda-base/release/examples/hashrate 32000000 101
```

The benchmark uses a zero target and checks exhaustive nonce counts. It never
connects to a node, wallet, or pool. Record clocks, temperature, power and any
other GPU activity when reproducing results.
