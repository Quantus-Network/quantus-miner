# Systemd unit files for Quantus External Miner

This directory contains a production-friendly systemd unit and drop-in overrides to run the Quantus External Miner as a managed Linux service.

Contents
- quantus-miner.service
  - A baseline unit file that runs the miner with journald logging and sensible security hardening.
  - Uses environment variables for configuration (preferred for stable ExecStart).
- overrides/
  - 10-shared-hardware.conf
    - Conservative CPU affinity and scheduling for hosts that run both a node and the miner.
    - Ensures the node remains responsive under load.
  - 20-dedicated-hardware.conf
    - Aggressive CPU affinity and scheduling for hosts dedicated to mining.
    - Maximizes miner throughput.

Prerequisites
- The quantus-miner binary installed at /usr/local/bin/quantus-miner (or adjust ExecStart).
- A service account (recommended):
  - sudo useradd --system --no-create-home --shell /usr/sbin/nologin quantus

Install (unit)
1) Copy the service file
   sudo install -D -m 0644 quantus-miner.service /etc/systemd/system/quantus-miner.service

2) Create a writable working directory (managed by systemd via StateDirectory)
   sudo install -d -o quantus -g quantus /var/lib/quantus-miner

3) (Optional) Provide environment variables
   - Debian/Ubuntu:   sudoedit /etc/default/quantus-miner
   - RHEL/CentOS/Fed: sudoedit /etc/sysconfig/quantus-miner

   Common variables (examples):
   MINER_ENGINE=cpu-fast
   MINER_PORT=9833
   MINER_METRICS_PORT=9900         # enable Prometheus exporter
   MINER_WORKERS=4                 # leave unset to use default (50% of effective CPUs)
   MINER_PROGRESS_CHUNK_MS=2000
   # Throttling engine (cpu-chain-manipulator) knobs:
   # MINER_MANIP_SOLVED_BLOCKS=0
   # MINER_MANIP_BASE_DELAY_NS=500000
   # MINER_MANIP_STEP_BATCH=10000
   # MINER_MANIP_THROTTLE_CAP=0
   # Extra CLI flags (kept stable ExecStart):
   # EXTRA_MINER_FLAGS="--some-future-flag value"

4) Enable and start
   sudo systemctl daemon-reload
   sudo systemctl enable --now quantus-miner.service

Install (overrides)
- Drop-in overrides live at: /etc/systemd/system/quantus-miner.service.d/*.conf
- Start with one of the examples and adjust CPU lists and weights to your host.

Shared hardware (node + miner)
- Use the conservative override:
  sudo install -D -m 0644 overrides/10-shared-hardware.conf /etc/systemd/system/quantus-miner.service.d/10-shared-hardware.conf
  sudo systemctl daemon-reload
  sudo systemctl restart quantus-miner

Dedicated hardware (miner only)
- Use the aggressive override:
  sudo install -D -m 0644 overrides/20-dedicated-hardware.conf /etc/systemd/system/quantus-miner.service.d/20-dedicated-hardware.conf
  sudo systemctl daemon-reload
  sudo systemctl restart quantus-miner

Configuration reference (environment variables)
- MINER_ENGINE
  - cpu-fast (default), cpu-baseline, cpu-chain-manipulator
  - gpu-cuda, gpu-opencl are placeholders; selecting them fails with a clear error.
- MINER_PORT
  - HTTP API port (default 9833)
- MINER_METRICS_PORT
  - Enable Prometheus exporter when set (e.g., 9900). If unset, metrics exporter is disabled.
- MINER_WORKERS
  - Worker threads (logical CPUs). If unset, defaults to ~50% of effective CPUs (clamped to [1, effective-1]).
- MINER_PROGRESS_CHUNK_MS
  - Target milliseconds for per-thread progress updates (default 2000ms).
- Throttling engine (cpu-chain-manipulator) knobs
  - MINER_MANIP_SOLVED_BLOCKS, MINER_MANIP_BASE_DELAY_NS, MINER_MANIP_STEP_BATCH, MINER_MANIP_THROTTLE_CAP
- EXTRA_MINER_FLAGS
  - Optional extra CLI flags appended to ExecStart.

CPU affinity, cpusets, and workers
- The miner detects the effective CPU capacity (logical CPUs) visible to the process by preferring cgroup v2 cpuset (cpuset.cpus.effective), falling back to v1, else using all logical CPUs.
- At startup (debug level), the miner logs the detected cpuset mask (if any).
- A Prometheus gauge miner_effective_cpus is emitted (when metrics are enabled) with the effective count for dashboards/alerts.
- If --workers (or MINER_WORKERS) exceeds effective CPUs, it is clamped and a warning is logged.
- If omitted, the miner defaults to ~50% of effective CPUs (but always at least 1 and less than or equal to effective-1).
- When pinning CPUAffinity at the systemd level:
  - Ensure CPUAffinity is a subset of the cgroup cpuset mask.
  - Consider setting MINER_WORKERS to match the number of CPUs in the affinity mask if you want full utilization, or rely on the default 50% policy.

Security hardening (in the unit)
- NoNewPrivileges=true
- ProtectSystem=full
- ProtectHome=true
- PrivateTmp=true
- RestrictNamespaces=true
- LockPersonality=true
- RestrictSUIDSGID=true
- SystemCallFilter=@system-service
Adjust or relax as needed for your environment.

Validation and troubleshooting
- Check service status and logs:
  journalctl -u quantus-miner -f
- Verify CPU affinity:
  pid=$(pidof quantus-miner)
  taskset -cp "$pid"
- Verify cpuset mask (cgroup v2):
  cat /sys/fs/cgroup/cpuset.cpus.effective
- Metrics:
  - If MINER_METRICS_PORT is set, curl http://127.0.0.1:<port>/metrics
  - Look for miner_effective_cpus and per-job/thread metrics.
- Common pitfalls:
  - ExecStart path wrong (ensure /usr/local/bin/quantus-miner exists and is executable).
  - Service user/group missing (create quantus or adjust User/Group).
  - CPUAffinity not a subset of the cgroup cpuset (adjust cpuset or affinity).
  - MINER_ENGINE set to gpu-* (currently unimplemented, exits with clear error).
  - Insufficient permissions to write WorkingDirectory (systemd StateDirectory creates /var/lib/quantus-miner with correct ownership).

Operational tips
- For shared machines: prefer 10-shared-hardware.conf and leave MINER_WORKERS unset (defaults to ~50%).
- For dedicated machines: use 20-dedicated-hardware.conf and set MINER_WORKERS to the number of CPUs in CPUAffinity (or omit CPUAffinity to inherit cpuset).
- Use RUST_LOG=info,miner=debug temporarily to verify startup detection (cpuset mask, effective CPUs) and to observe mining loop behavior; then turn back down to reduce log volume.

Support
- Repository: https://github.com/Quantus-Network/quantus-external-miner
- See docs/implementation/ for iteration-specific design and operational notes (metrics, throttling, partitioning, etc.).