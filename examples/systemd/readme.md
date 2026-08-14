# Systemd unit files for Quantus External Miner

This directory contains a production-friendly systemd unit and drop-in overrides to run the Quantus External Miner as a managed Linux service.

Contents
- quantus-miner.service
  - A service unit file that runs the miner with journald logging and sensible security hardening.
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
- The node's miner auth token and TLS cert fingerprint (required since the node
  authenticates miners). The node creates both on first start with
  --miner-listen-port and logs their paths (default:
  `<base-path>/chains/<chain>/miner-auth-token` and
  `<base-path>/chains/<chain>/miner-tls-cert-sha256`). Copy them where the
  service can read them — the unit sets ProtectHome=true, so paths under /home
  or /root are not readable:
  - sudo install -d -m 0755 /etc/quantus-miner
  - sudo install -m 0640 -o root -g quantus \
      "<base-path>/chains/<chain>/miner-auth-token" /etc/quantus-miner/miner-auth-token
  - sudo install -m 0644 \
      "<base-path>/chains/<chain>/miner-tls-cert-sha256" /etc/quantus-miner/miner-tls-cert-sha256

Install (unit)
1) Copy the service file
   sudo install -D -m 0644 quantus-miner.service /etc/systemd/system/quantus-miner.service

2) Create a writable working directory (managed by systemd via StateDirectory)
   sudo install -d -o quantus -g quantus /var/lib/quantus-miner

3) Provide environment variables (auth vars are REQUIRED)
   - Debian/Ubuntu:   sudoedit /etc/default/quantus-miner
   - RHEL/CentOS/Fed: sudoedit /etc/sysconfig/quantus-miner

   Common variables (examples). Note: systemd EnvironmentFile= does not strip
   inline comments — keep each assignment on its own line with nothing after
   the value.
   MINER_NODE_ADDR=127.0.0.1:9833
   # Required: node auth token + TLS cert pin (see Prerequisites)
   MINER_AUTH_TOKEN_FILE=/etc/quantus-miner/miner-auth-token
   MINER_TLS_CERT_SHA256_FILE=/etc/quantus-miner/miner-tls-cert-sha256
   # CPU worker threads; leave unset to auto-detect (~50% of available CPUs)
   MINER_CPU_WORKERS=4
   MINER_GPU_DEVICES=0
   # Prometheus exporter port (the exporter is always on; this only changes the port)
   MINER_METRICS_PORT=9900
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
- MINER_NODE_ADDR
  - Address of the node's miner QUIC endpoint (default 127.0.0.1:9833).
- MINER_AUTH_TOKEN_FILE (required, or MINER_AUTH_TOKEN inline)
  - Path to a copy of the node's miner-auth-token file. The file variant is
    preferred so the secret stays off the command line and out of `systemctl show`.
- MINER_TLS_CERT_SHA256_FILE (required, or MINER_TLS_CERT_SHA256 inline)
  - Path to a copy of the node's miner-tls-cert-sha256 file (the fingerprint is
    also printed in the node's startup logs).
- MINER_CPU_WORKERS
  - CPU worker threads. If unset, auto-detected.
- MINER_GPU_DEVICES
  - Number of GPU devices to use. If unset, auto-detected.
- MINER_GPU_BATCH_SIZE / MINER_CPU_BATCH_SIZE
  - Nonces/hashes per cancellation check (defaults 1000000 / 10000).
- MINER_GPU_THROTTLE_MS
  - Delay between GPU batches in milliseconds (default 0 = no throttle).
- MINER_METRICS_PORT
  - Prometheus exporter port (default 9900). The exporter is ALWAYS on and
    binds plaintext HTTP on all interfaces (0.0.0.0); this variable only
    changes the port — there is no disable or loopback-only option. Firewall
    the port or restrict it to your monitoring network.
- MINER_ALLOW_INTEGRATED
  - Allow integrated GPUs even when discrete GPUs are present.
- EXTRA_MINER_FLAGS
  - Optional extra CLI flags appended to ExecStart.

CPU affinity, cpusets, and workers
- The miner counts available CPUs via the process CPU affinity mask (num_cpus),
  so a systemd CPUAffinity= setting (or a cgroup cpuset) is reflected in the count.
- If MINER_CPU_WORKERS is unset, the miner uses ~50% of the detected CPUs
  (at least 1) and logs the choice at startup ("Auto-detected N CPU workers").
- An explicit MINER_CPU_WORKERS value is used as-is — it is NOT clamped to the
  affinity mask, so an oversized value oversubscribes the pinned CPUs.
- A Prometheus gauge miner_effective_cpus is emitted with the detected count for dashboards/alerts.
- When pinning CPUAffinity at the systemd level:
  - Ensure CPUAffinity is a subset of the cgroup cpuset mask.
  - Set MINER_CPU_WORKERS to the number of CPUs in the affinity mask for full
    utilization, or leave it unset for the ~50% default.

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
Note: the Prometheus exporter always listens on 0.0.0.0:<MINER_METRICS_PORT>
(default 9900) — the unit cannot disable it or bind it to loopback. Firewall
the port if the host is reachable from untrusted networks.

Validation and troubleshooting
- Check service status and logs:
  journalctl -u quantus-miner -f
- Verify CPU affinity:
  pid=$(pidof quantus-miner)
  taskset -cp "$pid"
- Verify cpuset mask (cgroup v2):
  cat /sys/fs/cgroup/cpuset.cpus.effective
- Metrics (always on, default port 9900):
  - curl http://127.0.0.1:${MINER_METRICS_PORT:-9900}/metrics
  - Look for miner_effective_cpus and per-job/thread metrics.
- Common pitfalls:
  - ExecStart path wrong (ensure /usr/local/bin/quantus-miner exists and is executable).
  - Service user/group missing (create quantus or adjust User/Group).
  - Miner exits immediately with "miner auth token required" / "TLS cert fingerprint required":
    set MINER_AUTH_TOKEN_FILE and MINER_TLS_CERT_SHA256_FILE in the env file (see Prerequisites).
  - Auth/TLS files unreadable: ProtectHome=true blocks /home and /root; copy the
    files to /etc/quantus-miner/ and make them readable by the quantus user.
  - Node rejects the miner after the node's base path changed or its credential
    files (miner-auth-token, miner-tls-cert.der/-key.der) were deleted and
    regenerated: re-copy both files and restart. (A plain purge-chain does NOT
    rotate them — it removes only the database, so the same token and cert are
    reloaded on the next start.)
  - CPUAffinity not a subset of the cgroup cpuset (adjust cpuset or affinity).
  - Insufficient permissions to write WorkingDirectory (systemd StateDirectory creates /var/lib/quantus-miner with correct ownership).

Operational tips
- For shared machines: prefer 10-shared-hardware.conf and leave MINER_CPU_WORKERS unset (auto-detect).
- For dedicated machines: use 20-dedicated-hardware.conf and set MINER_CPU_WORKERS to the number of CPUs in CPUAffinity (or omit CPUAffinity to inherit cpuset).
- Use RUST_LOG=info,miner=debug temporarily to verify startup detection (worker auto-detection, GPU discovery) and to observe mining loop behavior; then turn back down to reduce log volume.

Support
- Repository: https://github.com/Quantus-Network/quantus-miner
