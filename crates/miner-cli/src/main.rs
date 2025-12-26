use clap::{Parser, Subcommand};
use engine_cpu::{EngineRange, MinerEngine};
use miner_service::{run, ServiceConfig};
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Run the mining service (default behavior)
    Serve {
        /// Port number to listen on for the miner HTTP API
        #[arg(short, long, env = "MINER_PORT", default_value_t = 9833)]
        port: u16,

        /// Number of CPU worker threads to use for mining (None = auto-detect)
        #[arg(long = "cpu-workers", env = "MINER_CPU_WORKERS")]
        cpu_workers: Option<usize>,

        /// Number of GPU devices to use for mining (None = auto-detect)
        #[arg(long = "gpu-devices", env = "MINER_GPU_DEVICES")]
        gpu_devices: Option<usize>,

        /// Optional Prometheus metrics exporter port; if omitted, metrics are disabled
        #[arg(long, env = "MINER_METRICS_PORT")]
        metrics_port: Option<u16>,

        /// Enable verbose logging (shows debug info, progress details, etc.)
        #[arg(short, long, env = "MINER_VERBOSE")]
        verbose: bool,

        /// How often to report mining progress (in milliseconds).
        /// Smaller values give more frequent updates but slightly reduce performance.
        #[arg(long = "progress-interval-ms", env = "MINER_PROGRESS_INTERVAL_MS")]
        progress_interval_ms: Option<u64>,

        /// Size of work chunks to process before reporting progress (number of hashes).
        /// If omitted, uses engine-specific defaults (200K for CPU, 100M for GPU).
        #[arg(long = "chunk-size", env = "MINER_CHUNK_SIZE")]
        chunk_size: Option<u64>,

        /// For cpu-chain-manipulator: start throttle index at this many solved blocks
        /// to "pick up where we left off" after restarts.
        #[arg(long = "manip-solved-blocks", env = "MINER_MANIP_SOLVED_BLOCKS")]
        manip_solved_blocks: Option<u64>,

        /// For cpu-chain-manipulator: base sleep per batch in nanoseconds (default 500_000 ns)
        #[arg(long = "manip-base-delay-ns", env = "MINER_MANIP_BASE_DELAY_NS")]
        manip_base_delay_ns: Option<u64>,

        /// For cpu-chain-manipulator: number of nonce attempts between sleeps (default 10_000)
        #[arg(long = "manip-step-batch", env = "MINER_MANIP_STEP_BATCH")]
        manip_step_batch: Option<u64>,

        /// For cpu-chain-manipulator: optional cap on solved-blocks throttle index
        #[arg(long = "manip-throttle-cap", env = "MINER_MANIP_THROTTLE_CAP")]
        manip_throttle_cap: Option<u64>,

        /// Telemetry endpoints (repeat --telemetry-endpoint or comma-separated)
        #[arg(long = "telemetry-endpoint", env = "MINER_TELEMETRY_ENDPOINTS", value_delimiter = ',', num_args = 0.., value_name = "URL")]
        telemetry_endpoints: Option<Vec<String>>,

        /// Enable or disable telemetry explicitly
        #[arg(long = "telemetry-enabled", env = "MINER_TELEMETRY_ENABLED")]
        telemetry_enabled: Option<bool>,

        /// Telemetry verbosity level (0..=4 typical)
        #[arg(long = "telemetry-verbosity", env = "MINER_TELEMETRY_VERBOSITY")]
        telemetry_verbosity: Option<u8>,

        /// Interval seconds for system.interval messages
        #[arg(
            long = "telemetry-interval-secs",
            env = "MINER_TELEMETRY_INTERVAL_SECS"
        )]
        telemetry_interval_secs: Option<u64>,

        /// Default association: chain name
        #[arg(long = "telemetry-chain", env = "MINER_TELEMETRY_CHAIN")]
        telemetry_chain: Option<String>,

        /// Default association: genesis hash (hex)
        #[arg(long = "telemetry-genesis", env = "MINER_TELEMETRY_GENESIS")]
        telemetry_genesis: Option<String>,

        /// Default association: node telemetry id
        #[arg(long = "telemetry-node-id", env = "MINER_TELEMETRY_NODE_ID")]
        telemetry_node_id: Option<String>,

        /// Default association: node libp2p peer id
        #[arg(long = "telemetry-node-peer-id", env = "MINER_TELEMETRY_NODE_PEER_ID")]
        telemetry_node_peer_id: Option<String>,

        /// Default association: node name
        #[arg(long = "telemetry-node-name", env = "MINER_TELEMETRY_NODE_NAME")]
        telemetry_node_name: Option<String>,

        /// Default association: node version
        #[arg(long = "telemetry-node-version", env = "MINER_TELEMETRY_NODE_VERSION")]
        telemetry_node_version: Option<String>,
    },
    /// Run a quick benchmark of the mining engines
    Benchmark {
        /// Number of CPU workers to use for benchmark
        #[arg(long = "cpu-workers", env = "MINER_CPU_WORKERS")]
        cpu_workers: Option<usize>,

        /// Number of GPU devices to use for benchmark
        #[arg(long = "gpu-devices", env = "MINER_GPU_DEVICES")]
        gpu_devices: Option<usize>,

        /// Benchmark duration in seconds (default: 10)
        #[arg(short, long, default_value_t = 10)]
        duration: u64,

        /// Enable verbose logging during benchmark
        #[arg(short, long, env = "MINER_VERBOSE")]
        verbose: bool,
    },
}

/// Quantus External Miner CLI
#[derive(Parser, Debug)]
#[command(author, version = option_env!("MINER_VERSION").unwrap_or(env!("CARGO_PKG_VERSION")), about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
}

#[allow(clippy::enum_variant_names)]
#[tokio::main]
async fn main() {
    let args = Args::parse();

    match args.command.unwrap_or(Command::Serve {
        port: 9833,
        cpu_workers: None,
        gpu_devices: None,
        metrics_port: None,
        verbose: false,
        progress_interval_ms: None,
        chunk_size: None,
        manip_solved_blocks: None,
        manip_base_delay_ns: None,
        manip_step_batch: None,
        manip_throttle_cap: None,
        telemetry_endpoints: None,
        telemetry_enabled: None,
        telemetry_verbosity: None,
        telemetry_interval_secs: None,
        telemetry_chain: None,
        telemetry_genesis: None,
        telemetry_node_id: None,
        telemetry_node_peer_id: None,
        telemetry_node_name: None,
        telemetry_node_version: None,
    }) {
        Command::Serve {
            port,
            cpu_workers,
            gpu_devices,
            metrics_port,
            verbose,
            progress_interval_ms,
            chunk_size,
            manip_solved_blocks,
            manip_base_delay_ns,
            manip_step_batch,
            manip_throttle_cap,
            telemetry_endpoints,
            telemetry_enabled,
            telemetry_verbosity,
            telemetry_interval_secs,
            telemetry_chain,
            telemetry_genesis,
            telemetry_node_id,
            telemetry_node_peer_id,
            telemetry_node_name,
            telemetry_node_version,
        } => {
            run_serve_command(
                port,
                cpu_workers,
                gpu_devices,
                metrics_port,
                verbose,
                progress_interval_ms,
                chunk_size,
                manip_solved_blocks,
                manip_base_delay_ns,
                manip_step_batch,
                manip_throttle_cap,
                telemetry_endpoints,
                telemetry_enabled,
                telemetry_verbosity,
                telemetry_interval_secs,
                telemetry_chain,
                telemetry_genesis,
                telemetry_node_id,
                telemetry_node_peer_id,
                telemetry_node_name,
                telemetry_node_version,
            )
            .await;
        }
        Command::Benchmark {
            cpu_workers,
            gpu_devices,
            duration,
            verbose,
        } => {
            run_benchmark_command(cpu_workers, gpu_devices, duration, verbose).await;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_serve_command(
    port: u16,
    cpu_workers: Option<usize>,
    gpu_devices: Option<usize>,
    metrics_port: Option<u16>,
    verbose: bool,
    progress_interval_ms: Option<u64>,
    chunk_size: Option<u64>,
    manip_solved_blocks: Option<u64>,
    manip_base_delay_ns: Option<u64>,
    manip_step_batch: Option<u64>,
    manip_throttle_cap: Option<u64>,
    telemetry_endpoints: Option<Vec<String>>,
    telemetry_enabled: Option<bool>,
    telemetry_verbosity: Option<u8>,
    telemetry_interval_secs: Option<u64>,
    telemetry_chain: Option<String>,
    telemetry_genesis: Option<String>,
    telemetry_node_id: Option<String>,
    telemetry_node_peer_id: Option<String>,
    telemetry_node_name: Option<String>,
    telemetry_node_version: Option<String>,
) {
    // Initialize logger early to capture startup messages.
    // If RUST_LOG is not set, default to appropriate level based on verbose flag
    if std::env::var("RUST_LOG").is_err() {
        let log_level = if verbose {
            "debug,miner=debug,gpu_engine=debug,engine_cpu=debug"
        } else {
            "info,miner=info,gpu_engine=info"
        };
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();

    // Telemetry CLI passthrough to env for miner-service bootstrap
    if let Some(eps) = telemetry_endpoints.as_ref() {
        if !eps.is_empty() {
            std::env::set_var("MINER_TELEMETRY_ENDPOINTS", eps.join(","));
        }
    }
    if let Some(v) = telemetry_enabled {
        std::env::set_var("MINER_TELEMETRY_ENABLED", if v { "1" } else { "0" });
    }
    if let Some(v) = telemetry_verbosity {
        std::env::set_var("MINER_TELEMETRY_VERBOSITY", v.to_string());
    }
    if let Some(v) = telemetry_interval_secs {
        std::env::set_var("MINER_TELEMETRY_INTERVAL_SECS", v.to_string());
    }
    if let Some(v) = telemetry_chain.as_ref() {
        std::env::set_var("MINER_TELEMETRY_CHAIN", v);
    }
    if let Some(v) = telemetry_genesis.as_ref() {
        std::env::set_var("MINER_TELEMETRY_GENESIS", v);
    }
    if let Some(v) = telemetry_node_id.as_ref() {
        std::env::set_var("MINER_TELEMETRY_NODE_ID", v);
    }
    if let Some(v) = telemetry_node_peer_id.as_ref() {
        std::env::set_var("MINER_TELEMETRY_NODE_PEER_ID", v);
    }
    if let Some(v) = telemetry_node_name.as_ref() {
        std::env::set_var("MINER_TELEMETRY_NODE_NAME", v);
    }
    if let Some(v) = telemetry_node_version.as_ref() {
        std::env::set_var("MINER_TELEMETRY_NODE_VERSION", v);
    }

    // Log effective configuration (concise; see ServiceConfig Display)
    log::info!("Starting external miner service...");

    let config = ServiceConfig {
        port,
        cpu_workers,
        gpu_devices,
        metrics_port,
        progress_interval_ms,
        chunk_size,
        manip_solved_blocks,
        manip_base_delay_ns,
        manip_step_batch,
        manip_throttle_cap,
    };
    log::info!("Effective config: {config}");

    if let Err(e) = run(config).await {
        log::error!("Miner service terminated with error: {e:?}");
        std::process::exit(1);
    }
}

async fn run_benchmark_command(
    cpu_workers: Option<usize>,
    gpu_devices: Option<usize>,
    duration: u64,
    verbose: bool,
) {
    // Initialize logger early to capture startup messages.
    if std::env::var("RUST_LOG").is_err() {
        let log_level = if verbose {
            "debug,miner=debug,gpu_engine=debug,engine_cpu=debug"
        } else {
            "info,miner=info,gpu_engine=warn,engine_cpu=info"
        };
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();

    let effective_cpu_workers = cpu_workers.unwrap_or_else(num_cpus::get);
    let effective_gpu_devices = gpu_devices.unwrap_or(0);
    let total_workers = effective_cpu_workers + effective_gpu_devices;

    println!("🚀 Quantus Miner Benchmark");
    println!("==========================");
    println!("Configuration:");
    println!(
        "  CPU Workers: {} (Available Cores: {})",
        effective_cpu_workers,
        num_cpus::get()
    );
    println!("  GPU Devices: {}", effective_gpu_devices);
    println!("Duration: {} seconds", duration);
    println!("Total Workers: {}", total_workers);

    // Create engines based on configuration
    let cpu_engine = if effective_cpu_workers > 0 {
        Some(Arc::new(engine_cpu::FastCpuEngine::new()) as Arc<dyn MinerEngine>)
    } else {
        None
    };

    let gpu_engine = if effective_gpu_devices > 0 {
        Some(Arc::new(engine_gpu::GpuEngine::new()) as Arc<dyn MinerEngine>)
    } else {
        None
    };

    if total_workers == 0 {
        eprintln!("Error: No workers specified");
        std::process::exit(1);
    }
    println!();

    // Run benchmark
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let benchmark_start = Instant::now();

    let benchmark_range = EngineRange {
        start: U512::from(0u64),
        end: U512::from(100_000_000u64), // 100M nonces
    };

    let mut header = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut header);
    let difficulty = U512::MAX; // High difficulty - no solutions expected

    // We need a context from *some* engine. They should be compatible (same job).
    // Just pick one.
    let ref_engine = cpu_engine.as_ref().or(gpu_engine.as_ref()).unwrap();
    let ctx = ref_engine.prepare_context(header, difficulty);

    println!("⛏️  Starting benchmark...");

    // Spawn worker threads
    let mut handles = Vec::new();
    let total_hashes_arc = Arc::new(std::sync::Mutex::new(0u64));

    // Chunk sizes
    let cpu_chunk = 10_000u64;
    // GPU chunk size from 1 - 10M is good
    let gpu_chunk = 1_000_000u64;

    if effective_gpu_devices > 0 {
        if let Some(gpu_engine_arc) = &gpu_engine {
            // We can't easily get device count from MinerEngine trait without downcasting or extending trait.
            // But we know it's GpuEngine here.
            if let Some(gpu_ptr) = gpu_engine_arc
                .as_any()
                .downcast_ref::<engine_gpu::GpuEngine>()
            {
                let device_count = gpu_ptr.device_count();
                if effective_gpu_devices > device_count {
                    eprintln!(
                        "❌ ERROR: You have requested {} GPU devices but only {} device(s) are available.",
                        effective_gpu_devices, device_count
                    );
                    eprintln!("   Please check your --gpu-devices setting or GPU hardware.");
                    std::process::exit(1);
                }
            }
        }
    }

    for worker_id in 0..total_workers {
        // Determine type and engine
        let (engine, nonces_per_worker) = if worker_id < effective_cpu_workers {
            (
                cpu_engine.as_ref().expect("CPU engine missing").clone(),
                cpu_chunk,
            )
        } else {
            (
                gpu_engine.as_ref().expect("GPU engine missing").clone(),
                gpu_chunk,
            )
        };

        let ctx = ctx.clone();
        let cancel_flag = cancel_flag.clone();
        let total_hashes = total_hashes_arc.clone();

        let handle = thread::spawn(move || {
            // We'll just define a range based on worker ID and a large multiplier
            // so they don't overlap easily, though it doesn't matter for pure hashrate.
            // Use a large stride to separate workers.
            let stride = U512::from(1_000_000_000_000u64);
            let worker_start = benchmark_range
                .start
                .saturating_add(U512::from(worker_id as u64).saturating_mul(stride));

            let worker_range = EngineRange {
                start: worker_start,
                end: worker_start
                    .saturating_add(U512::from(nonces_per_worker))
                    .saturating_sub(U512::from(1u64)),
            };

            let mut worker_hashes = 0u64;

            loop {
                if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }

                let result = engine.search_range(&ctx, worker_range.clone(), &cancel_flag);

                match result {
                    engine_cpu::EngineStatus::Found { hash_count, .. }
                    | engine_cpu::EngineStatus::Exhausted { hash_count }
                    | engine_cpu::EngineStatus::Cancelled { hash_count } => {
                        worker_hashes += hash_count;
                        *total_hashes.lock().unwrap() += hash_count;
                    }
                    engine_cpu::EngineStatus::Running { .. } => {}
                }

                // Check if we've exceeded the time limit
                if benchmark_start.elapsed() >= Duration::from_secs(duration) {
                    break;
                }
            }

            engine_gpu::GpuEngine::clear_worker_resources();

            worker_hashes
        });

        handles.push(handle);
    }

    // Wait for duration or interrupt
    let mut last_update = Instant::now();

    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;

        if benchmark_start.elapsed() >= Duration::from_secs(duration) {
            cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            break;
        }

        // Update progress every second
        if last_update.elapsed() >= Duration::from_secs(1) {
            let current_hashes = *total_hashes_arc.lock().unwrap();
            let elapsed = benchmark_start.elapsed().as_secs_f64();

            if current_hashes > 0 {
                let hash_rate = current_hashes as f64 / elapsed;
                let hash_rate_str = if hash_rate >= 1_000_000.0 {
                    format!("{:.1}M", hash_rate / 1_000_000.0)
                } else if hash_rate >= 1_000.0 {
                    format!("{:.1}K", hash_rate / 1_000.0)
                } else {
                    format!("{:.0}", hash_rate)
                };
                println!("⏱️  {:.1}s - {} H/s", elapsed, hash_rate_str);
            } else if effective_gpu_devices > 0 {
                println!("⏱️  {:.1}s - processing...", elapsed);
            } else {
                println!("⏱️  {:.1}s - starting...", elapsed);
            }

            last_update = Instant::now();
        }
    }

    // Wait for all threads to finish
    for handle in handles {
        let _ = handle.join();
    }

    let total_elapsed = benchmark_start.elapsed();
    let final_hashes = *total_hashes_arc.lock().unwrap();
    let avg_hash_rate = final_hashes as f64 / total_elapsed.as_secs_f64();

    println!();
    println!("📊 Benchmark Results");
    println!("===================");
    println!("Total time: {:.2} seconds", total_elapsed.as_secs_f64());
    println!("Total hashes: {}", final_hashes);

    let hash_rate_str = if avg_hash_rate >= 1_000_000.0 {
        format!("{:.2}M H/s", avg_hash_rate / 1_000_000.0)
    } else if avg_hash_rate >= 1_000.0 {
        format!("{:.2}K H/s", avg_hash_rate / 1_000.0)
    } else {
        format!("{:.0} H/s", avg_hash_rate)
    };

    println!("Average hash rate: {}", hash_rate_str);

    if total_workers > 1 {
        let per_worker_rate = avg_hash_rate / total_workers as f64;
        let per_worker_str = if per_worker_rate >= 1_000_000.0 {
            format!("{:.2}M H/s", per_worker_rate / 1_000_000.0)
        } else if per_worker_rate >= 1_000.0 {
            format!("{:.2}K H/s", per_worker_rate / 1_000.0)
        } else {
            format!("{:.0} H/s", per_worker_rate)
        };
        println!(
            "Per-worker rate: {} (across {} workers)",
            per_worker_str, total_workers
        );
    }

    println!("✅ Benchmark completed successfully!");
}
