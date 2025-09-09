# Implementation docs index

This directory is intended for PR reviewers as the definitive context for evaluating changes. Each entry captures a discrete iteration as a “prompt” (plan/spec) and a matching “response” (implementation and outcomes). Commits and PRs should link to the relevant entry; authoring/process guidance lives in /agents.md.

Conventions
- Numbering: entries are prefixed with a two‑digit sequence (00, 01, …) to indicate order.
- Pairing: most entries come as a prompt/response pair. Some early entries may be standalone context docs.
- Scope: each entry should be narrowly scoped and actionable to ease review and onboarding.

Table of contents
- 00
  - [00-cpu-algorithm-and-optimisation-roadmap.md](./00-cpu-algorithm-and-optimisation-roadmap.md)
- 01
  - [01-prompt-metrics-toggle-and-soc-restructure.md](./01-prompt-metrics-toggle-and-soc-restructure.md)
  - [01-response-metrics-toggle-and-soc-restructure.md](./01-response-metrics-toggle-and-soc-restructure.md)
- 02
  - [02-prompt-cpu-fast-engine-and-metrics.md](./02-prompt-cpu-fast-engine-and-metrics.md)
  - [02-response-cpu-fast-engine-and-metrics.md](./02-response-cpu-fast-engine-and-metrics.md)
- 03
  - [03-prompt-engine-comparability.md](./03-prompt-engine-comparability.md)
  - [03-response-engine-comparability.md](./03-response-engine-comparability.md)
- 04
  - [04-prompt-metrics-chunking-and-dashboards.md](./04-prompt-metrics-chunking-and-dashboards.md)
  - [04-response-metrics-chunking-and-dashboards.md](./04-response-metrics-chunking-and-dashboards.md)
- 05
  - [05-prompt-workers-and-dashboards-sync.md](./05-prompt-workers-and-dashboards-sync.md)
  - [05-response-workers-and-dashboards-sync.md](./05-response-workers-and-dashboards-sync.md)
- 06
  - [06-prompt-manipulator-and-difficulty-overlay.md](./06-prompt-manipulator-and-difficulty-overlay.md)
  - [06-response-manipulator-and-difficulty-overlay.md](./06-response-manipulator-and-difficulty-overlay.md)
- 07
  - [07-prompt-stable-clippy-taplo-and-gpu-enum.md](./07-prompt-stable-clippy-taplo-and-gpu-enum.md)
  - [07-response-stable-clippy-taplo-and-gpu-enum.md](./07-response-stable-clippy-taplo-and-gpu-enum.md)

For authors and agents
Authoring workflow, conventions, and commit/PR linkage requirements are documented in /agents.md.

Cross‑references
- Authors and agents: see /agents.md for authoring workflow and process details. The repo README.md also links to review-critical docs.
- Workspace structure, crate boundaries, and feature gates are described inline in prompt/response docs.
- Dashboards and metrics: see entries 04–07 for metrics chunking, per‑thread/job gauges, and exporter behavior.

Last updated
- 07: Stable toolchain, Clippy/Taplo hygiene, GPU engine enum placeholders with user‑friendly errors.