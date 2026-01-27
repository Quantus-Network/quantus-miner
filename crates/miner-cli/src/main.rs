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
enum Command {
    /// Run the mining service
    Serve {
        /// Address of the node to connect to (e.g., "127.0.0.1:9833")
        #[arg(long, env = "MINER_NODE_ADDR")]
        node_addr: std::net::SocketAddr,

        /// Number of CPU worker threads to use for mining (default: auto-detect)
        #[arg(long = "cpu-workers", env = "MINER_CPU_WORKERS")]
        cpu_workers: Option<usize>,

        /// Number of GPU devices to use for mining (default: auto-detect)
        #[arg(long = "gpu-devices", env = "MINER_GPU_DEVICES")]
        gpu_devices: Option<usize>,

        /// Target duration for GPU mining batches in milliseconds (default: 3000)
        #[arg(long = "gpu-batch-duration-ms", env = "MINER_GPU_BATCH_DURATION_MS")]
        gpu_batch_duration_ms: Option<u64>,

        /// Port for Prometheus metrics HTTP endpoint (default: 9900)
        #[arg(long = "metrics-port", env = "MINER_METRICS_PORT", default_value_t = 9900)]
        metrics_port: u16,

        /// Enable verbose logging
        #[arg(short, long, env = "MINER_VERBOSE")]
        verbose: bool,
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
        eprintln!("Error: No command provided. Use 'serve --node-addr <ADDRESS>' to start mining.");
        eprintln!("Example: quantus-miner serve --node-addr 127.0.0.1:9833");
        std::process::exit(1);
    };

    match command {
        Command::Serve {
            node_addr,
            cpu_workers,
            gpu_devices,
            gpu_batch_duration_ms,
            metrics_port,
            verbose,
        } => {
            init_logger(verbose);

            log::info!("Starting external miner service...");

            // Start metrics HTTP server
            if let Err(e) = metrics::start_http_exporter(metrics_port).await {
                log::error!("Failed to start metrics exporter: {e:?}");
                std::process::exit(1);
            }
            log::info!("Metrics available at http://0.0.0.0:{}/metrics", metrics_port);

            let config = ServiceConfig {
                node_addr,
                cpu_workers,
                gpu_devices,
                gpu_batch_duration_ms,
            };

            if let Err(e) = run(config).await {
                log::error!("Miner service terminated with error: {e:?}");
                std::process::exit(1);
            }
        }

        Command::Benchmark {
            cpu_workers,
            gpu_devices,
            duration,
            verbose,
        } => {
            init_logger(verbose);
            run_benchmark(cpu_workers, gpu_devices, duration).await;
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

async fn run_benchmark(cpu_workers: Option<usize>, gpu_devices: Option<usize>, duration: u64) {
    let effective_cpu_workers = cpu_workers.unwrap_or_else(num_cpus::get);

    // Initialize GPU engine
    let (gpu_engine, effective_gpu_devices) =
        match miner_service::resolve_gpu_configuration(gpu_devices, None) {
            Ok((engine, count)) => (engine, count),
            Err(e) => {
                eprintln!("❌ ERROR: {}", e);
                std::process::exit(1);
            }
        };

    let total_workers = effective_cpu_workers + effective_gpu_devices;

    println!("🚀 Quantus Miner Benchmark");
    println!("==========================");
    println!("CPU Workers: {} (Available: {})", effective_cpu_workers, num_cpus::get());
    println!("GPU Devices: {}", effective_gpu_devices);
    println!("Duration: {} seconds", duration);
    println!();

    if total_workers == 0 {
        eprintln!("Error: No workers specified");
        std::process::exit(1);
    }

    // Create CPU engine
    let cpu_engine: Option<Arc<dyn MinerEngine>> = if effective_cpu_workers > 0 {
        Some(Arc::new(engine_cpu::FastCpuEngine::new()))
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
                end: worker_start.saturating_add(U512::from(nonces_per_batch)).saturating_sub(U512::from(1u64)),
            };

            loop {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }

                let result = engine.search_range(&ctx, worker_range.clone(), &cancel);

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
