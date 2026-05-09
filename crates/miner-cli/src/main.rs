use clap::{Parser, Subcommand, ValueEnum};
use engine_cpu::{AtomicBoolCancelCheck, EngineRange, MinerEngine};
use miner_service::{
    run,
    zk_aggregation::{ClaimStrategy, ZkAggregationConfig},
    ServiceConfig,
};
use primitive_types::U512;
use rand::RngCore;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// CLI defaults
const DEFAULT_GPU_BATCH_SIZE: u64 = 1_000_000;
const DEFAULT_CPU_BATCH_SIZE: u64 = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliClaimStrategy {
    Oldest,
    RewardDensity,
}

impl From<CliClaimStrategy> for ClaimStrategy {
    fn from(value: CliClaimStrategy) -> Self {
        match value {
            CliClaimStrategy::Oldest => ClaimStrategy::Oldest,
            CliClaimStrategy::RewardDensity => ClaimStrategy::RewardDensity,
        }
    }
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the mining service
    Serve {
        /// Address of the node to connect to
        #[arg(long, env = "MINER_NODE_ADDR", default_value = "127.0.0.1:9833")]
        node_addr: std::net::SocketAddr,

        /// Number of CPU worker threads to use for mining (default: auto-detect)
        #[arg(long = "cpu-workers", env = "MINER_CPU_WORKERS")]
        cpu_workers: Option<usize>,

        /// Number of GPU devices to use for mining (default: auto-detect)
        #[arg(long = "gpu-devices", env = "MINER_GPU_DEVICES")]
        gpu_devices: Option<usize>,

        /// GPU batch size in nonces - controls how often GPU checks for cancellation
        #[arg(long = "gpu-batch-size", env = "MINER_GPU_BATCH_SIZE", default_value_t = DEFAULT_GPU_BATCH_SIZE)]
        gpu_batch_size: u64,

        /// CPU batch size in hashes - controls how often CPU checks for cancellation
        #[arg(long = "cpu-batch-size", env = "MINER_CPU_BATCH_SIZE", default_value_t = DEFAULT_CPU_BATCH_SIZE)]
        cpu_batch_size: u64,

        /// Port for Prometheus metrics HTTP endpoint (default: 9900)
        #[arg(
            long = "metrics-port",
            env = "MINER_METRICS_PORT",
            default_value_t = 9900
        )]
        metrics_port: u16,

        /// Enable verbose logging
        #[arg(short, long, env = "MINER_VERBOSE")]
        verbose: bool,

        /// Enable delegated ZK L1 aggregation worker
        #[arg(long = "enable-zk-aggregation", env = "MINER_ENABLE_ZK_AGGREGATION")]
        enable_zk_aggregation: bool,

        /// Chain RPC endpoint used by the ZK aggregation watcher
        #[arg(
            long = "node-rpc",
            env = "MINER_NODE_RPC",
            default_value = "ws://127.0.0.1:9944"
        )]
        node_rpc: String,

        /// Aggregation account address
        #[arg(long = "aggregation-account", env = "MINER_AGGREGATION_ACCOUNT")]
        aggregation_account: Option<String>,

        /// Aggregation signing key or keystore path
        #[arg(long = "aggregation-key", env = "MINER_AGGREGATION_KEY")]
        aggregation_key: Option<String>,

        /// Directory containing generated ZK proving/verifier artifacts
        #[arg(long = "zk-bins-dir", env = "MINER_ZK_BINS_DIR")]
        zk_bins_dir: Option<PathBuf>,

        /// Number of dedicated ZK aggregation workers
        #[arg(long = "zk-workers", env = "MINER_ZK_WORKERS", default_value_t = 1)]
        zk_workers: usize,

        /// Maximum active bonded ZK aggregation jobs
        #[arg(
            long = "max-active-zk-jobs",
            env = "MINER_MAX_ACTIVE_ZK_JOBS",
            default_value_t = 1
        )]
        max_active_zk_jobs: usize,

        /// Minimum aggregation reward required before claiming a bundle
        #[arg(
            long = "min-aggregation-reward",
            env = "MINER_MIN_AGGREGATION_REWARD",
            default_value_t = 0
        )]
        min_aggregation_reward: u128,

        /// Miner bond amount reserved when claiming a ZK aggregation bundle
        #[arg(
            long = "zk-miner-bond",
            env = "MINER_ZK_MINER_BOND",
            default_value_t = 0
        )]
        zk_miner_bond: u128,

        /// Bundle claiming strategy
        #[arg(
            long = "claim-strategy",
            env = "MINER_CLAIM_STRATEGY",
            value_enum,
            default_value = "oldest"
        )]
        claim_strategy: CliClaimStrategy,

        /// Validate opportunities without claiming or proving
        #[arg(long = "dry-run-zk-aggregation", env = "MINER_DRY_RUN_ZK_AGGREGATION")]
        dry_run_zk_aggregation: bool,
    },

    /// Run a quick benchmark of the mining engines
    Benchmark {
        /// Number of CPU workers to use for benchmark
        #[arg(long = "cpu-workers", env = "MINER_CPU_WORKERS")]
        cpu_workers: Option<usize>,

        /// Number of GPU devices to use for benchmark
        #[arg(long = "gpu-devices", env = "MINER_GPU_DEVICES")]
        gpu_devices: Option<usize>,

        /// GPU batch size in nonces - controls how often GPU checks for cancellation
        #[arg(long = "gpu-batch-size", env = "MINER_GPU_BATCH_SIZE", default_value_t = DEFAULT_GPU_BATCH_SIZE)]
        gpu_batch_size: u64,

        /// CPU batch size in hashes - controls how often CPU checks for cancellation
        #[arg(long = "cpu-batch-size", env = "MINER_CPU_BATCH_SIZE", default_value_t = DEFAULT_CPU_BATCH_SIZE)]
        cpu_batch_size: u64,

        /// Benchmark duration in seconds (default: 10)
        #[arg(short, long, default_value_t = 10)]
        duration: u64,

        /// Enable verbose logging
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

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let Some(command) = args.command else {
        eprintln!("Error: No command provided. Use 'serve' to start mining (defaults to local node at 127.0.0.1:9833).");
        eprintln!("Example: quantus-miner serve --node-addr 127.0.0.1:9833");
        std::process::exit(1);
    };

    match command {
        Command::Serve {
            node_addr,
            cpu_workers,
            gpu_devices,
            gpu_batch_size,
            cpu_batch_size,
            metrics_port,
            verbose,
            enable_zk_aggregation,
            node_rpc,
            aggregation_account,
            aggregation_key,
            zk_bins_dir,
            zk_workers,
            max_active_zk_jobs,
            min_aggregation_reward,
            zk_miner_bond,
            claim_strategy,
            dry_run_zk_aggregation,
        } => {
            init_logger(verbose);

            log::info!("Starting external miner service...");

            // Start metrics HTTP server
            if let Err(e) = metrics::start_http_exporter(metrics_port).await {
                log::error!("Failed to start metrics exporter: {e:?}");
                std::process::exit(1);
            }
            log::info!(
                "Metrics available at http://0.0.0.0:{}/metrics",
                metrics_port
            );

            let config = ServiceConfig {
                node_addr,
                cpu_workers,
                gpu_devices,
                gpu_batch_size,
                cpu_batch_size,
                zk_aggregation: enable_zk_aggregation.then_some(ZkAggregationConfig {
                    node_rpc,
                    aggregation_account,
                    aggregation_key,
                    zk_bins_dir,
                    workers: zk_workers,
                    max_active_jobs: max_active_zk_jobs,
                    min_aggregation_reward,
                    miner_bond: zk_miner_bond,
                    claim_strategy: claim_strategy.into(),
                    dry_run: dry_run_zk_aggregation,
                }),
            };

            if let Err(e) = run(config).await {
                log::error!("Miner service terminated with error: {e:?}");
                std::process::exit(1);
            }
        }

        Command::Benchmark {
            cpu_workers,
            gpu_devices,
            gpu_batch_size,
            cpu_batch_size,
            duration,
            verbose,
        } => {
            init_logger(verbose);
            run_benchmark(
                cpu_workers,
                gpu_devices,
                gpu_batch_size,
                cpu_batch_size,
                duration,
            )
            .await;
        }
    }
}

fn init_logger(verbose: bool) {
    if std::env::var("RUST_LOG").is_err() {
        let log_level = if verbose {
            "debug,miner=debug,gpu_engine=debug,engine_cpu=debug"
        } else {
            "info,miner=info,gpu_engine=info"
        };
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();
}

async fn run_benchmark(
    cpu_workers: Option<usize>,
    gpu_devices: Option<usize>,
    gpu_batch_size: u64,
    cpu_batch_size: u64,
    duration: u64,
) {
    let effective_cpu_workers = cpu_workers.unwrap_or_else(num_cpus::get);

    // Initialize GPU engine
    let (gpu_engine, effective_gpu_devices) =
        match miner_service::resolve_gpu_configuration(gpu_devices, gpu_batch_size) {
            Ok((engine, count)) => (engine, count),
            Err(e) => {
                eprintln!("❌ ERROR: {}", e);
                std::process::exit(1);
            }
        };

    let total_workers = effective_cpu_workers + effective_gpu_devices;

    println!("🚀 Quantus Miner Benchmark");
    println!("==========================");
    println!(
        "CPU Workers: {} (Available: {})",
        effective_cpu_workers,
        num_cpus::get()
    );
    println!("GPU Devices: {}", effective_gpu_devices);
    println!("Duration: {} seconds", duration);
    println!();

    if total_workers == 0 {
        eprintln!("Error: No workers specified");
        std::process::exit(1);
    }

    // Create CPU engine
    let cpu_engine: Option<Arc<dyn MinerEngine>> = if effective_cpu_workers > 0 {
        Some(Arc::new(engine_cpu::FastCpuEngine::new(cpu_batch_size)))
    } else {
        None
    };

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let benchmark_start = Instant::now();

    // Random header hash for benchmark
    let mut header = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut header);
    let difficulty = U512::MAX; // High difficulty - no solutions expected

    let ref_engine = cpu_engine.as_ref().or(gpu_engine.as_ref()).unwrap();
    let ctx = ref_engine.prepare_context(header, difficulty);

    println!("⛏️  Starting benchmark...");

    // Spawn worker threads
    let mut handles = Vec::new();
    let total_hashes = Arc::new(std::sync::Mutex::new(0u64));

    let cpu_chunk = 10_000u64;
    let gpu_chunk = 1_000_000u64;

    for worker_id in 0..total_workers {
        let (engine, nonces_per_batch) = if worker_id < effective_cpu_workers {
            (cpu_engine.as_ref().unwrap().clone(), cpu_chunk)
        } else {
            (gpu_engine.as_ref().unwrap().clone(), gpu_chunk)
        };

        let ctx = ctx.clone();
        let cancel = cancel_flag.clone();
        let hashes = total_hashes.clone();
        let start = benchmark_start;

        let handle = thread::spawn(move || {
            let stride = U512::from(1_000_000_000_000u64);
            let worker_start = U512::from(worker_id as u64).saturating_mul(stride);
            let worker_range = EngineRange {
                start: worker_start,
                end: worker_start
                    .saturating_add(U512::from(nonces_per_batch))
                    .saturating_sub(U512::from(1u64)),
            };

            loop {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }

                let cancel_check = AtomicBoolCancelCheck(&cancel);
                let result = engine.search_range(&ctx, worker_range.clone(), &cancel_check);

                match result {
                    engine_cpu::EngineStatus::Found { hash_count, .. }
                    | engine_cpu::EngineStatus::Exhausted { hash_count }
                    | engine_cpu::EngineStatus::Cancelled { hash_count } => {
                        *hashes.lock().unwrap() += hash_count;
                    }
                    engine_cpu::EngineStatus::Running { .. } => {}
                }

                if start.elapsed() >= Duration::from_secs(duration) {
                    break;
                }
            }

            engine_gpu::GpuEngine::clear_worker_resources();
        });

        handles.push(handle);
    }

    // Progress updates
    let mut last_update = Instant::now();

    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;

        if benchmark_start.elapsed() >= Duration::from_secs(duration) {
            cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            break;
        }

        if last_update.elapsed() >= Duration::from_secs(1) {
            let current = *total_hashes.lock().unwrap();
            let elapsed = benchmark_start.elapsed().as_secs_f64();
            if current > 0 {
                let rate = current as f64 / elapsed;
                println!("⏱️  {:.1}s - {} H/s", elapsed, format_hash_rate(rate));
            }
            last_update = Instant::now();
        }
    }

    // Wait for threads
    for handle in handles {
        let _ = handle.join();
    }

    let total_elapsed = benchmark_start.elapsed();
    let final_hashes = *total_hashes.lock().unwrap();
    let avg_rate = final_hashes as f64 / total_elapsed.as_secs_f64();

    println!();
    println!("📊 Benchmark Results");
    println!("===================");
    println!("Total time: {:.2}s", total_elapsed.as_secs_f64());
    println!("Total hashes: {}", final_hashes);
    println!("Average rate: {} H/s", format_hash_rate(avg_rate));

    if total_workers > 1 {
        let per_worker = avg_rate / total_workers as f64;
        println!("Per-worker: {} H/s", format_hash_rate(per_worker));
    }

    println!("✅ Benchmark completed!");
}

fn format_hash_rate(rate: f64) -> String {
    if rate >= 1_000_000.0 {
        format!("{:.2}M", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.2}K", rate / 1_000.0)
    } else {
        format!("{:.0}", rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_parses_zk_aggregation_flags() {
        let args = Args::try_parse_from([
            "quantus-miner",
            "serve",
            "--enable-zk-aggregation",
            "--node-rpc",
            "ws://127.0.0.1:9944",
            "--aggregation-account",
            "alice",
            "--aggregation-key",
            "test-key",
            "--zk-bins-dir",
            "/tmp/zk-bins",
            "--zk-workers",
            "2",
            "--max-active-zk-jobs",
            "3",
            "--min-aggregation-reward",
            "42",
            "--zk-miner-bond",
            "50",
            "--claim-strategy",
            "reward-density",
            "--dry-run-zk-aggregation",
        ])
        .expect("serve args should parse");

        let Some(Command::Serve {
            enable_zk_aggregation,
            node_rpc,
            aggregation_account,
            aggregation_key,
            zk_bins_dir,
            zk_workers,
            max_active_zk_jobs,
            min_aggregation_reward,
            zk_miner_bond,
            claim_strategy,
            dry_run_zk_aggregation,
            ..
        }) = args.command
        else {
            panic!("expected serve command");
        };

        assert!(enable_zk_aggregation);
        assert_eq!(node_rpc, "ws://127.0.0.1:9944");
        assert_eq!(aggregation_account.as_deref(), Some("alice"));
        assert_eq!(aggregation_key.as_deref(), Some("test-key"));
        assert_eq!(zk_bins_dir, Some(PathBuf::from("/tmp/zk-bins")));
        assert_eq!(zk_workers, 2);
        assert_eq!(max_active_zk_jobs, 3);
        assert_eq!(min_aggregation_reward, 42);
        assert_eq!(zk_miner_bond, 50);
        assert_eq!(claim_strategy, CliClaimStrategy::RewardDensity);
        assert!(dry_run_zk_aggregation);
    }
}
