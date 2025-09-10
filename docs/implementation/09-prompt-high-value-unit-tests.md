# 09 Prompt: High‑value unit tests (engine parity, cancellation, service validation/HTTP, chunking)

This iteration adds targeted unit tests that materially improve correctness confidence without expanding scope into ancillary tooling. We focus on the core mining/engine path and the service’s request validation and HTTP behavior.

## Background

- We recently stabilized the toolchain (stable + clippy + taplo) and expanded runtime/ops capabilities (metrics, cpuset visibility, systemd units).
- Prior tests covered miner lifecycle basics, partitioning, and select pow-core invariants.
- Reviewers requested clearer evidence that:
  - Baseline vs Fast CPU engines yield the same results on the same work.
  - Engines properly signal Exhausted and honor cancellation.
  - Service request validation and HTTP endpoints behave correctly.
  - Chunked mining emits progress and completion signals reliably.

## Objectives

Add unit tests that:
1) Verify engine parity and edge conditions
   - Baseline vs Fast engines produce identical winner (nonce, distance) and hash_count on a small deterministic range.
   - Exhausted path returns exact range length as hash_count when threshold is too strict to find solutions.
   - Cancellation flag is honored immediately (no work performed).

2) Validate miner-service behavior
   - Request validator rejects malformed inputs across all error branches.
   - HTTP handlers return correct statuses for valid/invalid/double-submit flows.
   - Chunked mining emits at least one progress update (non-zero hash_count) and a final completion event.

Non-goals (explicitly excluded here):
- Manipulator engine tests (testnet tooling for unusual circumstances; outside core mining validity).
- Cpuset parser tests (not essential to mining core correctness).
- Metrics scrape tests (Grafana visibility/usage considered sufficient validation for this phase).

## Scope of Work

A) engine-cpu tests
- parity: Baseline and Fast engines agree on solution and hash_count in a deterministic small range under a permissive threshold.
- exhausted: Strict threshold over a small range produces Exhausted with hash_count equal to inclusive range length.
- cancel: Pre-set cancellation flag yields Cancelled with zero work.

B) miner-service tests
- validation: Table-driven coverage of validate_mining_request() error paths:
  - empty job_id
  - mining_hash length/hex errors
  - distance_threshold not decimal
  - nonce_start/nonce_end length/hex errors
  - nonce_start > nonce_end
- http: warp-based tests for:
  - GET /result (unknown) -> 404
  - POST /cancel (unknown) -> 404
  - POST /mine (valid) -> 200 OK (Accepted)
  - POST /mine (duplicate) -> 409 Conflict
- chunking: mine_range_with_engine() sends at least one progress update (completed=false, hash_count>0) and eventually a final completion message (completed=true); robust to multiple progress updates.

## Acceptance Criteria

- Tests compile and pass on stable:
  - cargo test --locked --workspace
- No new clippy/taplo regressions:
  - cargo clippy --workspace --all-targets (clean)
  - taplo fmt --check (clean)
- Engine-cpu tests:
  - Parity test asserts equal nonce, distance, hash_count (Found/Found).
  - Exhausted test asserts hash_count equals inclusive range length.
  - Cancel test asserts Cancelled with zero work when cancel is pre-set.
- Miner-service tests:
  - Validator test hits each failure branch with descriptive assertions.
  - HTTP test verifies 404/404/200/409 flows as specified.
  - Chunking test drains messages until completed=true and asserts at least one non-zero progress first.

## Validation Plan

- Run the full test suite under stable with --locked to validate reproducibility.
- Confirm logs stay quiet (no unexpected panics/warnings) and tests are deterministic.
- Ensure tests do not rely on real network or systemd; use in-process warp testing and direct function calls.

## Risks and Mitigations

- Risk: Flaky tests due to implicit timing/assumptions in chunking.
  - Mitigation: Use deterministic small ranges and drain messages until completion; avoid wall-clock sleeps in assertions.
- Risk: False positives in “exhausted” tests if thresholds not strict enough.
  - Mitigation: Use U512::zero() threshold and small ranges to make solutions effectively impossible.
- Risk: Overfitting tests to current implementation details.
  - Mitigation: Assert on externally visible behavior (status enums, counts, responses) rather than internal steps.

## Deliverables

- New unit tests in:
  - engine-cpu:
    - Baseline vs Fast parity
    - Exhausted path correctness
    - Immediate cancellation behavior
  - miner-service:
    - Request validation failure coverage
    - HTTP endpoint basics (result/cancel/mine/duplicate)
    - Chunked progress + completion signaling
- CI passes unchanged (stable toolchain; clippy/taplo/tests clean).

## Notes

- This iteration deliberately defers manipulator-specific tests, cpuset parser tests, and metrics scrape tests to keep scope focused on core mining/system API correctness and avoid test flakiness or non-critical coverage.