# Agents and Authors Guide

This document defines the authoring workflow and process conventions for proposal/response documentation and how commits and PRs must link to those docs. It exists to make PR review fast, predictable, and auditable.

If you’re reviewing a PR, the canonical technical narrative lives in:
- This process guide (agents.md) is the single source of truth for authoring and process now.
- Commits and PRs should include clear, self-contained descriptions and link to relevant code diffs.

If you’re authoring changes or operating as an agent, follow this guide.

---

## Purpose

- Keep changes small, explicit, and justified.
- Capture the plan (prompt) and the actual implementation (response) for each iterative step.
- Ensure every commit and PR references the relevant doc(s), so reviewers have full context without hunting.

---

## Where things live

- This process guide: /agents.md (for authors/agents and reviewers)
- Repo overview and user-facing guidance: /README.md

---

## Numbering and naming conventions

Each iteration is a prompt/response pair using a two-digit sequence:

- 08-prompt-<short-topic>.md
- 08-response-<short-topic>.md

Conventions:
- Two-digit sequence numbers (00, 01, …, 10, 11, …) in chronological order.
- short-topic is lowercase, words separated by hyphens.
Remove standalone iteration documents. Capture sufficient context directly in commit messages and PR descriptions.

Commit message guidance:
- Title: imperative, concise
- Body: motivation, high-level changes, key code paths, risks/mitigations
- Footer: links to key diffs or files (if helpful)

---

## Authoring workflow

1) Plan the change
- Capture background, objectives, scope, non-goals, risks/mitigations, and validation in the PR description template (see below).
- Prefer concise planning; link to tracking issues when relevant.

2) Implement the change
- Commit in small, reviewable steps.
- Keep messages crisp; reference the PR and key deltas.

3) Document the outcome
- Update the PR description with deviations from plan, validation results, and any follow-ups.

PR description template (copy/paste into GitHub):
- Overview
- What changed (by area/crate)
- Validation (tests, clippy, taplo, manual checks)
- Risks and mitigations
- Follow-ups (if any)

---

## Commit and PR linkage

Every commit must reference the iteration doc. Use this template:

Commit message template
- Title: imperative, max ~72 chars
- Body: details and motivation
- Footer: links to the prompt/response

Example:
Fix clippy lints and inline format args across miner-service

- Inline {var} format args in logs
- Remove redundant locals/fields
- No behavior changes

Docs:
- Prompt: docs/implementation/07-prompt-stable-clippy-taplo-and-gpu-enum.md
- Response: docs/implementation/07-response-stable-clippy-taplo-and-gpu-enum.md

PR description template
- What and why (summary)
- Validation (what you ran, results)
- Risks and mitigations
- Links:
  - Prompt: docs/implementation/NN-prompt-<topic>.md
  - Response: docs/implementation/NN-response-<topic>.md

---

## Local validation checklist (pre-PR)

Use the stable toolchain.

- Toolchain and lints
  - rustup show active-toolchain -> stable
  - cargo fmt --all -- --check
  - cargo clippy --workspace --all-targets -- -D warnings

- TOML formatting (Taplo)
  - taplo fmt --check

- Build and tests
  - cargo build --workspace --locked
  - cargo test --workspace --locked

- Runtime sanity (typical)
  - cargo run -p miner-cli -- --engine cpu-fast --workers <n>
  - Check logs and metrics if enabled (see below)

Notes:
- We standardize on stable (rust-toolchain sets channel = "stable").
- We keep components = ["clippy", "rustfmt"].
- Taplo formatting is authoritative per taplo.toml.

---

## CI expectations (what reviewers look for)

- Stable toolchain
- taplo fmt --check
- cargo fmt --all -- --check
- cargo clippy --workspace --all-targets -- -D warnings
- cargo test --workspace --locked

If PRs introduce new feature flags or targets, document how CI should build them, or gate them until CI can.

---

## Engine selection policy (naming and runtime)

Naming:
- CPU engines use Cpu-prefixed variants in the CLI to avoid future collisions and clarify behavior:
  - cpu-baseline, cpu-fast, cpu-chain-manipulator
- GPU placeholders are exposed for UX and planning but currently unimplemented:
  - gpu-cuda, gpu-opencl

Runtime behavior:
- If a GPU engine is selected, the service logs a clear error and exits non-zero:
  - “engine 'gpu-cuda' is not implemented yet; use cpu-fast or cpu-baseline.”
- Reviewers expect this behavior to remain until real GPU engines are implemented.

---

## Metrics and observability conventions

- Metrics are feature-gated (metrics feature) and can be toggled at runtime with --metrics-port.
- Per-job and per-thread hash-rate gauges:
  - Prefer remove-on-end semantics to avoid scrape-timing artifacts.
  - Per-thread EMAs are summed to derive per-job hash rate.
- When changing metrics:
  - Update prompt/response docs with metric names and label sets.
  - Consider dashboards and alerting downstream.

---

## Style and lint conventions

- Use inline format args in log macros, e.g., "value = {value}".
- Prefer saturating arithmetic for U512 ranges; avoid panics on partitioning.
- Keep tests deterministic; avoid flakiness due to implicit ranges or 0 nonces.
- Apply rustfmt and follow Clippy guidance unless it harms readability/intent.
- TOML is formatted via Taplo per taplo.toml (tabs, key ordering for workspace.dependencies, etc.).

---

## Change management notes

- Avoid renaming CLI options unless clearly justified; when renaming, provide an alias window and document migration in the response doc.
- When adding features (e.g., new engines), default to conservative runtime behavior and explicit logs on unsupported paths.
- For system-level advice (affinity, priorities), document briefly in README and/or example files (e.g., examples/systemd), and link from the PR description if relevant.

---

## FAQ

Q: Why do we keep Cpu-prefixed names in the CLI?
- To prevent ambiguity once GPU engines land and to keep the mental model consistent. We suppress the enum_variant_names lint locally in the CLI.

Q: Why stable and not nightly?
- Consistency and CI portability. Stable + Clippy + Taplo cover our current needs.

Q: Why remove series on job end rather than setting 0?
- To avoid scrape-timing artifacts and enable cleaner aggregations. This is intentional and should be maintained.

Q: Where do I put process/authoring details?
- In this file (/agents.md). This is the single reference for authoring and process guidance.

---

## Quick links

- Process guide (this file): /agents.md
- Workspace README: /README.md