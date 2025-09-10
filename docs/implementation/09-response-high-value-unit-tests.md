# 09 Response: High‑value unit tests (engine parity, cancellation, service validation/HTTP, chunking)

This iteration adds targeted tests that materially improve correctness confidence for the core CPU mining engines and the miner service API, while intentionally avoiding test areas that are non-critical to mining validity or likely to be flaky in CI.

---

## Summary of changes

- engine-cpu
  - Baseline vs Fast engine parity on a deterministic small range (permissive threshold).
  - Exhausted path correctness under a strict threshold (hash_count equals inclusive range length).
  - Immediate cancellation respected (pre-set cancel flag yields zero work).

- miner-service
  - Request validator coverage across all error branches.
  - HTTP endpoints basic behavior using in-process `warp::test::request`:
    - `GET /result/{unknown}` -> 404
    - `POST /cancel/{unknown}` -> 404
    - `POST /mine` (valid) -> 200 OK (Accepted)
    - `POST /mine` (duplicate) -> 409 Conflict
  - Chunked mining progress/complete signaling:
    - At least one progress update with `hash_count > 0`.
    - Robust to multiple progress messages; drains until a final completion event is observed.

Skipped by design (out-of-scope for core correctness):
- Manipulator engine tests (testnet-only tooling).
- Cpuset parser tests (not essential to mining validity).
- Metrics scrape tests (Grafana observability considered sufficient at this stage).

---

## Tests added (overview)

- engine-cpu
  - `baseline_and_fast_engines_find_same_candidate_on_small_range`
    - Asserts both engines produce the same winner (nonce, distance) and `hash_count`.
  - `engine_returns_exhausted_when_no_solution_in_range`
    - Uses `U512::zero()` threshold to make solutions effectively impossible; verifies `hash_count == (end - start + 1)`.
  - `engine_respects_immediate_cancellation`
    - Pre-set cancellation flag; expects `Cancelled` with `hash_count == 0`.

- miner-service
  - `validate_mining_request_rejects_bad_inputs`
    - Table-driven coverage of all validator error branches:
      - Empty `job_id`
      - `mining_hash` wrong length / invalid hex
      - Non-decimal `distance_threshold`
      - `nonce_start`/`nonce_end` wrong length / invalid hex
      - `nonce_start > nonce_end`
  - `http_endpoints_handle_basic_flows`
    - 404/404/200/409 responses for result/cancel/mine/mine-duplicate flow.
  - `chunked_mining_sends_progress_and_completion`
    - Confirms progress updates (`completed=false`, `hash_count>0`) and eventual completion (`completed=true`).
    - Uses a strict threshold and small range; drains messages to handle multiple progress events deterministically.

---

## Outcomes and coverage impact

- Engine parity:
  - Confirms the `cpu-fast` incremental path is consistent with the `cpu-baseline` reference for small deterministic ranges.
- Edge behavior:
  - Exhausted and Cancelled states are covered and validated with precise `hash_count` expectations.
- Service robustness:
  - Input validation error branches are exercised, reducing risk of acceptance of malformed jobs.
  - HTTP behavior for common flows (unknown, duplicate) is verified without requiring a live network.
- Chunking telemetry:
  - Confirms chunked mining emits progress and reliably signals completion, aligning with the metrics/reporting model.

---

## Validation

- Toolchain/tests:
  - `cargo +stable test --locked --workspace` passes (green).
  - `cargo clippy --workspace --all-targets` clean.
  - `taplo fmt --check` clean.

- Determinism:
  - No wall-clock sleeps in assertions; tests are robust to multiple progress updates by draining until completion.
  - Strict thresholds and small ranges avoid probabilistic outcomes.

---

## Compatibility and behavior

- No changes to engine or service semantics.
- Tests do not require network access or systemd; they are purely in-process.
- Tests avoid the manipulator engine to keep scope focused on core mining validity.

---

## Follow-ups (deferred)

- Manipulator-specific tests (throttling behavior and job_index increment patterns).
- Metrics scrape tests via registry encoding (e.g., TextEncoder) if needed later.
- End-to-end integration tests of exporter routes (optional; would require test harness that tolerates networking).

---

## Artifacts

- New unit tests within:
  - `crates/engine-cpu/src/lib.rs` (engine parity, exhausted, cancellation)
  - `crates/miner-service/src/lib.rs` (validator, HTTP basics, chunking behavior)

---

## Outcome

The added tests strengthen confidence that:
- `cpu-fast` equals `cpu-baseline` on core behavior,
- engines correctly handle `Exhausted` and `Cancelled` states,
- the service rejects malformed inputs, behaves correctly for basic HTTP flows,
- and chunked mining updates progress and completion deterministically.

These tests improve reviewer confidence in the correctness and maintainability of the core mining path without expanding scope into non-critical or potentially flaky areas.