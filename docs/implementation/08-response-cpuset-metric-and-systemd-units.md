# 08 Response: Cpuset visibility, effective CPU metric, and systemd units

This document summarizes the work completed for iteration 08. It delivers:
- Explicit cpuset visibility at startup via debug logs
- A Prometheus gauge for the miner’s effective CPU capacity
- Production-ready systemd unit files and drop-in overrides for shared and dedicated hardware

---

## Summary of changes

### 1) Miner startup: cpuset detection and logging
- Added a helper that returns both:
  - effective_cpus: logical CPU count visible to the process (cpuset-aware)
  - cpuset_mask: the string mask (if available)
- Detection order:
  1) cgroup v2: /sys/fs/cgroup/cpuset.cpus.effective
  2) cgroup v1: /sys/fs/cgroup/cpuset/cpuset.cpus
  3) Fallback: num_cpus::get()
- Startup behavior (debug logs):
  - If a mask is detected: “Detected cpuset mask: {mask}”
  - Otherwise: “No cpuset mask detected; using full logical CPU count”
- Worker defaults and clamping remain unchanged:
  - When --workers is omitted, default is ~50% of effective CPUs (clamped to [1, effective-1]).
  - If --workers exceeds effective CPUs, it’s clamped down with a warning.

Why this matters:
- Operators and reviewers can immediately see why the miner may clamp --workers.
- Makes resource capacity explicit for shared hosts and containerized deployments.

### 2) Metrics: miner_effective_cpus gauge
- Introduced a new Prometheus IntGauge: miner_effective_cpus
- Set once at startup with the detected effective CPU count.
- Emitted only when the metrics feature is enabled and the exporter runs (via --metrics-port).
- Use cases:
  - Dashboards: display and trend effective CPU capacity per instance.
  - Alerts: detect unexpected cpuset changes (e.g., misconfiguration or orchestrator drift).

### 3) Systemd units and operational packaging
- Added a baseline service unit with journald logging, security hardening, and environment-driven configuration:
  - examples/systemd/quantus-miner.service
    - Stable ExecStart using environment variables (keeps deployments simple).
    - StateDirectory/WorkingDirectory handling.
    - Optional EnvironmentFile support (/etc/default/quantus-miner and /etc/sysconfig/quantus-miner).
    - Reasonable security defaults (NoNewPrivileges, ProtectSystem, etc.).
- Added two drop-in override examples (install under /etc/systemd/system/quantus-miner.service.d/*.conf):
  - examples/systemd/overrides/10-shared-hardware.conf
    - Conservative CPUAffinity and scheduling for co-located node + miner hosts.
    - Keeps node responsive via Nice/CPUWeight settings.
  - examples/systemd/overrides/20-dedicated-hardware.conf
    - Aggressive CPU usage for dedicated mining hosts (full CPUAffinity, high CPUWeight, negative Nice).
    - Maximizes throughput while staying within standard scheduling policy.
- Added operational documentation:
  - examples/systemd/readme.md
    - Installation steps, environment variable reference, override usage, validation tips (journald, taskset, cpuset file checks, metrics curl).

Why this matters:
- Provides production-grade service artifacts out-of-the-box.
- Codifies best practices for both shared and dedicated deployments.
- Reduces support friction with clear operational guidance and validation commands.

---

## Validation

- Logging:
  - Verified cpuset detection messages at miner startup:
    - With cgroup v2 mask present: “Detected cpuset mask: X-Y,Z…”
    - Without cpuset: “No cpuset mask detected; using full logical CPU count”
- Metrics:
  - With metrics enabled (--metrics-port), miner_effective_cpus is present and reflects the detected effective CPU count.
- Systemd:
  - quantus-miner.service installs cleanly; service starts and logs as expected.
  - 10-shared-hardware.conf and 20-dedicated-hardware.conf are syntactically valid and apply intended CPUAffinity/Nice/CPUWeight settings.
  - taskset -cp $(pidof quantus-miner) matches affinity settings; cpuset effective mask is respected.
- Tooling and tests:
  - cargo +stable test --locked --workspace: green.
  - cargo clippy --workspace --all-targets: clean.
  - taplo fmt --check: clean.

---

## Compatibility and behavior

- No change to default worker allocation policy:
  - Still ~50% of effective CPUs when --workers is omitted (lower bound 1, upper bound effective-1).
- No changes to engine behavior, CLI flags, or API.
- Metrics emission for miner_effective_cpus occurs only when the metrics feature/exporter are enabled.

---

## Known limitations and follow-ups

- miner_effective_cpus is set at startup; if cpusets change at runtime, the metric won’t update until restart. If needed, we can add periodic polling in a future iteration.
- We do not manage or enforce cpusets; the unit/overrides assume either kernel- or orchestrator-level cpuset policy is in place when required.

---

## Artifacts

- Code:
  - Cpuset detection and debug logging at startup.
  - Effective CPU gauge emission (miner_effective_cpus) guarded by metrics feature.
- Metrics:
  - New IntGauge: miner_effective_cpus
  - Setter: metrics::set_effective_cpus(i64)
- Systemd:
  - examples/systemd/quantus-miner.service
  - examples/systemd/overrides/10-shared-hardware.conf
  - examples/systemd/overrides/20-dedicated-hardware.conf
  - examples/systemd/readme.md

---

## Outcome

We improved operational transparency (cpuset logging + effective CPU gauge) and packaged production-ready systemd units and overrides. This makes deployments more predictable and auditable, especially on shared hosts, while keeping defaults conservative and backwards-compatible.