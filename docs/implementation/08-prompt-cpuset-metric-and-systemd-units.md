# 08 Prompt: Cpuset visibility, effective CPU metric, and systemd units

This prompt builds on the prior iterations (01–07) and addresses two observability improvements around CPU capacity, plus operational packaging via systemd units and drop-in overrides.

## Background

- The miner auto-detects “effective CPUs” for the current process, preferring cgroup v2 cpuset (cpuset.cpus.effective), then cgroup v1, else falls back to all logical CPUs.
- When `--workers` is omitted, the miner defaults to ~50% of effective CPUs (clamped to [1, effective_cpus - 1]), which is desirable for hosts shared with a node.
- Reviewers/operators want the effective cpuset to be explicit in logs and dashboards to explain worker clamping and aid instance comparisons.
- Operators also need production-grade systemd artifacts to run the miner as a service with sensible CPU affinity defaults for shared vs. dedicated machines.

## Objectives

1) Cpuset visibility in logs
- Log the detected cpuset mask at miner startup (debug level).
- Log a helpful message when no cpuset is detected (e.g., “using full logical CPU count”).

2) Effective CPU metric
- Emit a Prometheus gauge recording the detected effective CPU count (cpuset-aware).
- Name: `miner_effective_cpus` (type: IntGauge).
- Only emit when the `metrics` feature and `--metrics-port` are enabled.

3) Systemd units and overrides
- Provide a production-ready `quantus-miner.service` unit with:
  - Journald logging, sane security hardening, and environment-driven configuration.
  - Stable `ExecStart` using environment variables (keeps deployment simple and auditable).
- Provide two drop-in override examples:
  - `10-shared-hardware.conf`: conservative CPU affinity/scheduling for co-located node + miner.
  - `20-dedicated-hardware.conf`: aggressive CPU affinity/scheduling for dedicated mining hosts.
- Include a `readme.md` explaining installation, overrides, environment variables, and validation.

## Scope of Work

A) Miner-service: cpuset logging and metric
- Add a helper to detect `(effective_cpus, cpuset_mask)`:
  - cgroup v2: `/sys/fs/cgroup/cpuset.cpus.effective`
  - cgroup v1: `/sys/fs/cgroup/cpuset/cpuset.cpus`
  - fallback: `num_cpus::get()`
- At startup in `run(config)`:
  - Log `Detected cpuset mask: {mask}` at debug when present; else log a “no cpuset” debug line.
  - If `metrics` feature is active, call `metrics::set_effective_cpus(effective_cpus as i64)`.

B) Metrics crate: new gauge and setter
- Add `miner_effective_cpus` (IntGauge) to the global registry.
- Expose `pub fn set_effective_cpus(n: i64)` that sets the gauge.

C) Systemd packaging
- Add `examples/systemd/quantus-miner.service` with:
  - User/Group separation (recommended `quantus` system user).
  - `StateDirectory` and `WorkingDirectory` setup under `/var/lib/quantus-miner`.
  - `EnvironmentFile` support (`/etc/default/quantus-miner`, `/etc/sysconfig/quantus-miner`).
  - Sensible security hardening (`NoNewPrivileges`, `ProtectSystem`, etc.).
  - Optional resource controls commented with guidance.
- Add overrides:
  - `examples/systemd/overrides/10-shared-hardware.conf`:
    - Example `CPUAffinity` binding a subset of CPUs (e.g., 4–7 on an 8-way).
    - `Nice` and `CPUWeight` tuned so the node remains responsive.
  - `examples/systemd/overrides/20-dedicated-hardware.conf`:
    - Full-CPU binding (or inherit cpuset), higher CPU weight, negative `Nice` for priority.
    - Relaxed `TasksMax` / `MemoryMax` for dedicated use.
- Add `examples/systemd/readme.md`:
  - Installation, enablement, environment variable reference, override usage.
  - Validation tips (journald logs, `taskset`, cpuset file checks, metrics curl).

## Non-goals

- Changing default worker allocation (remains ~50% of effective CPUs when unspecified).
- Modifying engine behavior or metrics beyond adding `miner_effective_cpus`.
- Implementing GPU engines or changing CLI flags.
- Introducing new runtime feature flags beyond the existing `metrics`.

## Acceptance Criteria

- On startup:
  - Debug log includes either “Detected cpuset mask: {mask}” or “No cpuset mask detected; using full logical CPU count.”
- Metrics:
  - When metrics are enabled, `/metrics` includes `miner_effective_cpus` with the effective CPU count.
  - Metric naming and type match this prompt.
- Systemd:
  - `examples/systemd/quantus-miner.service` installs cleanly and runs the miner.
  - `examples/systemd/overrides/10-shared-hardware.conf` and `20-dedicated-hardware.conf` are syntactically valid and documented.
  - `examples/systemd/readme.md` contains clear, step-by-step guidance.
- CI:
  - `taplo fmt --check` passes.
  - `cargo clippy --workspace --all-targets` is clean on stable.
  - `cargo test --locked --workspace` passes on stable.

## Validation Plan

- Unit/runtime checks:
  - Run miner with/without cpuset constraints and verify the debug log line.
  - With metrics enabled, curl `/metrics` and assert `miner_effective_cpus` presence and value.
- Systemd:
  - Install the service and each override on a test host/VM.
  - Verify `systemctl status`, journald logs, and `taskset -cp $(pidof quantus-miner)` reflect expectations.
  - Confirm `/sys/fs/cgroup/cpuset.cpus.effective` (if present) aligns with worker clamping and logs.
- Formatting & lints:
  - Run Taplo, Clippy, and tests on stable, ensuring no regressions.

## Risks and Mitigations

- Risk: CPUAffinity in overrides not a subset of cpuset; service fails to bind as intended.
  - Mitigation: README calls this out; instructs operators to verify with `taskset` and cpuset files.
- Risk: Metrics feature toggled off—no effective CPU gauge exported.
  - Mitigation: This is intentional. Document that metrics must be enabled for the gauge to appear.
- Risk: Different distros place env files in different paths.
  - Mitigation: Unit includes both Debian/Ubuntu and RHEL family `EnvironmentFile` lines (optional).

## Deliverables

- Code:
  - Cpuset detection + debug log; effective CPU gauge emission (guarded by `metrics`).
  - New `miner_effective_cpus` gauge and setter in metrics crate.
- Artifacts:
  - `examples/systemd/quantus-miner.service`
  - `examples/systemd/overrides/10-shared-hardware.conf`
  - `examples/systemd/overrides/20-dedicated-hardware.conf`
  - `examples/systemd/readme.md`
- CI:
  - All existing checks pass unchanged on stable (Clippy, Taplo, tests).