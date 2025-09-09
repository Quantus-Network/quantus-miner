# 07 Prompt: Move to Stable, Clippy/Taplo Hygiene, and GPU Engine Enum

This prompt follows from 06 (Manipulator + Difficulty Overlay). It focuses on getting the repository clean and future-proofed for CI by standardizing the toolchain, enforcing static checks, and aligning engine naming for upcoming GPU work.

## Background

- We have mature CPU engines (baseline and fast), metrics, and a throttling engine.
- CI currently runs Taplo (TOML formatter) checks; we also want to rely on stable Rust and Clippy hygiene.
- The engine naming convention uses CPU prefixes; we will soon add GPU offloading. We want to preserve that naming and add GPU variants in a forward-compatible way.

## Objectives

1) Standardize the toolchain on stable
- Switch the repo’s toolchain to stable (rust-toolchain: channel = "stable").
- Keep useful components in the toolchain (clippy, rustfmt).
- Remove the unused wasm target from toolchain configuration.

2) Enforce Clippy hygiene across the workspace
- Run cargo clippy --workspace --all-targets and fix warnings (prefer inline format args, etc.).
- Keep Cpu-prefixed CLI variants (e.g., CpuFast) to preserve clarity for future GPU engines; suppress the enum_variant_names lint locally (not globally).
- Avoid code “simplifications” that would reduce maintainability; target idiomatic fixes only.

3) Ensure Taplo formatting checks pass
- Taplo is already configured via taplo.toml.
- Run taplo fmt and taplo fmt --check to ensure all TOML files comply (workspace Cargo.toml and all crate Cargo.toml files).
- Accept any Taplo-driven reordering or indentation changes as authoritative.

4) Add GPU engine variants (forward-compatible placeholders)
- Extend CLI engine enum with gpu-cuda and gpu-opencl variants.
- Extend service EngineSelection with GpuCuda and GpuOpenCl.
- If a user selects a GPU engine, fail fast at runtime with a clear error message and non-zero exit, explaining the engine is unimplemented and to use CPU engines for now.
- Update CLI help text to indicate GPU options are unimplemented.

5) Fix warnings and tighten tests
- Address any outstanding compiler warnings in tests (unused imports/helpers, useless comparisons).
- Ensure cargo test --locked --workspace passes cleanly on stable.

6) CI alignment (Taplo + Clippy + fmt + tests)
- Ensure CI pipeline covers:
  - taplo fmt --check
  - cargo fmt -- --check (if used)
  - cargo clippy --workspace --all-targets -- -D warnings
  - cargo test --locked --workspace
- Confirm the pipeline uses stable toolchain.

## Scope of Work

- Toolchain:
  - Update rust-toolchain to stable and remove unused wasm target(s).
- Code hygiene:
  - Apply clippy-driven edits (inline format args, remove redundant locals/fields, fix docs indentation, remove unused test helpers/imports).
  - Preserve CPU engine variant names and add #[allow(clippy::enum_variant_names)] on the CLI enum.
- GPU placeholders:
  - CLI: add GpuCuda, GpuOpencl variants with descriptive help; maintain Cpu-prefixed variants.
  - Service: add EngineSelection::GpuCuda, ::GpuOpenCl; in run(), return a descriptive error when selected.
- Taplo:
  - Format all TOML files to pass taplo fmt --check.
- Tests:
  - Ensure all unit tests pass on stable.
- Documentation:
  - Add this prompt (07-prompt-stable-clippy-taplo-and-gpu-enum.md).
  - Add a corresponding response document (07-response-stable-clippy-taplo-and-gpu-enum.md) summarizing changes applied and their outcomes.

## Acceptance Criteria

- rust-toolchain uses stable with clippy and rustfmt; no wasm targets remain unless justified.
- cargo test --locked --workspace passes on stable with zero warnings from rustc.
- cargo clippy --workspace --all-targets returns clean (ideally with -D warnings in CI).
- taplo fmt --check passes.
- CLI supports engine values:
  - cpu-baseline, cpu-fast, cpu-chain-manipulator (existing, fully functional)
  - gpu-cuda, gpu-opencl (new, unimplemented; selecting them yields a clear error and non-zero exit)
- Service run(config) surfaces a meaningful error message (and exit) for unimplemented engines.
- Documentation pair (07 prompt/response) added under docs/implementation.

## Non-Goals

- Implementing the actual GPU engines.
- Changing engine behavior or metrics pipelines beyond hygiene/UX improvements.
- Introducing new config options unrelated to the scope above.

## Risks and Mitigations

- Risk: Clippy or Taplo changes could clash with future code-gen or formatting preferences.
  - Mitigation: Keep changes minimal and idiomatic; follow Taplo’s configuration as the source of truth.
- Risk: Users expect GPU to work once they see options.
  - Mitigation: Explicitly label GPU engines as unimplemented in CLI help and fail with clear runtime errors.

## Suggested Developer Workflow

1) Toolchain
- Update rust-toolchain to stable, keeping clippy/rustfmt, and remove wasm target(s).

2) Hygiene & Formatting
- cargo fmt (if used)
- cargo clippy --workspace --all-targets
- Fix issues (focus on inline format args, remove unused imports, doc indentation).
- taplo fmt, then taplo fmt --check

3) Tests & Runtime
- cargo test --locked --workspace
- Run the CLI:
  - quantus-miner --engine cpu-fast (runs fine)
  - quantus-miner --engine gpu-cuda (clear error message and non-zero exit)

4) CI
- Ensure pipeline includes taplo fmt --check, clippy -D warnings, and tests on stable.

## Deliverables

- Code changes:
  - Updated rust-toolchain
  - Clippy-driven fixes across crates
  - GPU engine placeholders in CLI and service with descriptive runtime errors
  - Taplo-formatted TOMLs
- Documentation:
  - docs/implementation/07-prompt-stable-clippy-taplo-and-gpu-enum.md (this file)
  - docs/implementation/07-response-stable-clippy-taplo-and-gpu-enum.md (companion response)
- CI (if updated in this iteration):
  - Workflow(s) set to stable, with taplo fmt --check, clippy, fmt check, and tests

## References

- 06-prompt-manipulator-and-difficulty-overlay.md
- 06-response-manipulator-and-difficulty-overlay.md
- taplo.toml (TOML formatting rules)