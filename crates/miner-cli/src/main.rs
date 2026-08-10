use clap::{Parser, Subcommand};
use engine_cpu::{AtomicBoolCancelCheck, EngineRange, JobIdCancelCheck, MinerEngine};
use miner_service::{run, ServiceConfig};
use pow_core::JobContext;
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

// CLI defaults
const DEFAULT_GPU_BATCH_SIZE: u32 = 1_000_000;
const DEFAULT_CPU_BATCH_SIZE: u64 = 10_000;

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
        gpu_batch_size: u32,

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

        /// GPU throttle delay in milliseconds between batches (0 = no throttle)
        #[arg(
            long = "gpu-throttle-ms",
            env = "MINER_GPU_THROTTLE_MS",
            default_value_t = 0
        )]
        gpu_throttle_ms: u64,

        /// Allow integrated GPUs (APUs) even when discrete GPUs are available.
        /// By default, integrated GPUs are skipped when a discrete GPU is present
        /// to avoid resource contention and driver instability.
        #[arg(long = "allow-integrated", env = "MINER_ALLOW_INTEGRATED")]
        allow_integrated: bool,

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

        /// GPU batch size in nonces - controls how often GPU checks for cancellation
        #[arg(long = "gpu-batch-size", env = "MINER_GPU_BATCH_SIZE", default_value_t = DEFAULT_GPU_BATCH_SIZE)]
        gpu_batch_size: u32,

        /// CPU batch size in hashes - controls how often CPU checks for cancellation
        #[arg(long = "cpu-batch-size", env = "MINER_CPU_BATCH_SIZE", default_value_t = DEFAULT_CPU_BATCH_SIZE)]
        cpu_batch_size: u64,

        /// Benchmark duration in seconds (default: 10)
        #[arg(short, long, default_value_t = 10)]
        duration: u64,

        /// Simulate node job churn: every N seconds push a new random header
        /// (like NewJob). 0 = sustained single job (default).
        #[arg(long = "job-interval", default_value_t = 0.0)]
        job_interval: f64,

        /// PoW difficulty for job simulation (decimal). Default when
        /// --job-interval > 0: "max" (cancel-only; matches mainnet preemption).
        /// Pass a decimal to also exercise the Found→idle path.
        #[arg(long = "difficulty")]
        difficulty: Option<String>,

        /// Allow integrated GPUs (APUs) even when discrete GPUs are available
        #[arg(long = "allow-integrated", env = "MINER_ALLOW_INTEGRATED")]
        allow_integrated: bool,

        /// Enable verbose logging
        #[arg(short, long, env = "MINER_VERBOSE")]
        verbose: bool,
    },
}

/// Quantus External Miner CLI
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
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
            gpu_throttle_ms,
            metrics_port,
            allow_integrated,
            verbose,
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
                gpu_throttle_ms,
                allow_integrated,
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
            job_interval,
            difficulty,
            allow_integrated,
            verbose,
        } => {
            init_logger(verbose);
            run_benchmark(
                cpu_workers,
                gpu_devices,
                gpu_batch_size,
                cpu_batch_size,
                duration,
                job_interval,
                difficulty,
                allow_integrated,
            )
            .await;
        }
    }
}

fn init_logger(verbose: bool) {
    if std::env::var("RUST_LOG").is_err() {
        // Filter out noisy wgpu/naga shader compilation logs
        let log_level = if verbose {
            "debug,miner=debug,gpu_engine=debug,engine_cpu=debug,wgpu=warn,wgpu_core=warn,wgpu_hal=warn,naga=warn"
        } else {
            "info,miner=info,gpu_engine=info,wgpu=error,wgpu_core=error,wgpu_hal=error,naga=error"
        };
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();
}

fn parse_difficulty(raw: Option<&str>, job_interval: f64) -> U512 {
    match raw {
        // Job sim defaults to unreachable difficulty: mainnet almost always
        // preempts via NewJob rather than a local find.
        None if job_interval > 0.0 => U512::MAX,
        None => U512::MAX,
        Some(s) if s.eq_ignore_ascii_case("max") => U512::MAX,
        Some(s) => match U512::from_dec_str(s) {
            Ok(d) if !d.is_zero() => d,
            Ok(_) => {
                eprintln!("❌ ERROR: --difficulty must be non-zero (or \"max\")");
                std::process::exit(1);
            }
            Err(_) => {
                eprintln!("❌ ERROR: invalid --difficulty '{s}' (decimal or \"max\")");
                std::process::exit(1);
            }
        },
    }
}

#[derive(Default)]
struct PhaseAccum {
    device_index: usize,
    wind_up: Vec<f64>,
    busy: Vec<f64>,
    wind_down: Vec<f64>,
    samples: u64,
}

fn mean_ms(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn p50_ms(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}


fn random_header() -> [u8; 32] {
    let mut header = [0u8; 32];
    rand::rng().fill_bytes(&mut header);
    header
}

async fn run_benchmark(
    cpu_workers: Option<usize>,
    gpu_devices: Option<usize>,
    gpu_batch_size: u32,
    cpu_batch_size: u64,
    duration: u64,
    job_interval: f64,
    difficulty_arg: Option<String>,
    allow_integrated: bool,
) {
    // When --gpu-devices is set and --cpu-workers is omitted, default to GPU-only
    // so hardware A/B numbers aren't polluted by host CPU hashrate.
    let effective_cpu_workers = match (cpu_workers, gpu_devices) {
        (Some(n), _) => n,
        (None, Some(_)) => 0,
        (None, None) => num_cpus::get(),
    };

    // Initialize GPU engine (no throttle for benchmark)
    let (gpu_engine, effective_gpu_devices) = match miner_service::resolve_gpu_configuration(
        gpu_devices,
        gpu_batch_size,
        0,
        allow_integrated,
    ) {
        Ok((engine, count)) => (engine, count),
        Err(e) => {
            eprintln!("❌ ERROR: {}", e);
            std::process::exit(1);
        }
    };

    let total_workers = effective_cpu_workers + effective_gpu_devices;
    let difficulty = parse_difficulty(difficulty_arg.as_deref(), job_interval);

    println!("🚀 Quantus Miner Benchmark");
    println!("==========================");
    println!(
        "CPU Workers: {} (Available: {})",
        effective_cpu_workers,
        num_cpus::get()
    );
    println!("GPU Devices: {}", effective_gpu_devices);
    println!("GPU batch size: {} nonces", gpu_batch_size);
    println!("CPU batch size: {} hashes", cpu_batch_size);
    println!("Duration: {} seconds", duration);
    if job_interval > 0.0 {
        println!("Job interval: {:.2}s (simulated NewJob)", job_interval);
        if difficulty == U512::MAX {
            println!("Difficulty: max (cancel-only churn, no finds)");
        } else {
            println!("Difficulty: {difficulty}");
        }
    } else {
        println!("Job interval: off (sustained single job)");
    }
    println!();

    if total_workers == 0 {
        eprintln!("Error: No workers specified");
        std::process::exit(1);
    }

    if job_interval < 0.0 {
        eprintln!("❌ ERROR: --job-interval must be >= 0");
        std::process::exit(1);
    }

    // Create CPU engine
    let cpu_engine: Option<Arc<dyn MinerEngine>> = if effective_cpu_workers > 0 {
        Some(Arc::new(engine_cpu::FastCpuEngine::new(cpu_batch_size)))
    } else {
        None
    };

    if job_interval > 0.0 {
        run_benchmark_with_jobs(
            cpu_engine,
            gpu_engine,
            effective_cpu_workers,
            effective_gpu_devices,
            duration,
            job_interval,
            difficulty,
        )
        .await;
    } else {
        run_benchmark_sustained(
            cpu_engine,
            gpu_engine,
            effective_cpu_workers,
            effective_gpu_devices,
            gpu_batch_size,
            cpu_batch_size,
            duration,
        )
        .await;
    }
}

/// Continuous hashing on one header (difficulty MAX). Measures peak sustained H/s.
async fn run_benchmark_sustained(
    cpu_engine: Option<Arc<dyn MinerEngine>>,
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    effective_cpu_workers: usize,
    effective_gpu_devices: usize,
    gpu_batch_size: u32,
    cpu_batch_size: u64,
    duration: u64,
) {
    let total_workers = effective_cpu_workers + effective_gpu_devices;
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let benchmark_start = Instant::now();

    let header = random_header();
    let difficulty = U512::MAX;
    let ref_engine = cpu_engine.as_ref().or(gpu_engine.as_ref()).unwrap();
    let ctx = ref_engine.prepare_context(header, difficulty);

    println!("⛏️  Starting sustained benchmark...");

    let mut handles = Vec::new();
    let total_hashes = Arc::new(Mutex::new(0u64));

    let cpu_chunk = cpu_batch_size.max(10_000);
    let gpu_chunk = gpu_batch_size as u64;

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
            let mut nonce = U512::from(worker_id as u64).saturating_mul(stride);

            loop {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }

                let worker_range = EngineRange {
                    start: nonce,
                    end: nonce
                        .saturating_add(U512::from(nonces_per_batch))
                        .saturating_sub(U512::from(1u64)),
                };

                let cancel_check = AtomicBoolCancelCheck(&cancel);
                let result = engine.search_range(&ctx, worker_range, &cancel_check);

                match result {
                    engine_cpu::EngineStatus::Found { hash_count, .. }
                    | engine_cpu::EngineStatus::Exhausted { hash_count }
                    | engine_cpu::EngineStatus::Cancelled { hash_count }
                    | engine_cpu::EngineStatus::DeviceLost { hash_count } => {
                        *hashes.lock().unwrap() += hash_count;
                    }
                    engine_cpu::EngineStatus::Running { .. } => {}
                }

                if matches!(result, engine_cpu::EngineStatus::DeviceLost { .. }) {
                    break;
                }

                nonce = nonce.saturating_add(U512::from(nonces_per_batch));

                if start.elapsed() >= Duration::from_secs(duration) {
                    break;
                }
            }

            engine_gpu::GpuEngine::clear_worker_resources();
        });

        handles.push(handle);
    }

    progress_and_join(
        handles,
        cancel_flag,
        total_hashes,
        benchmark_start,
        duration,
        None,
        None,
    )
    .await;
}

/// Serve-like path: open-ended search, JobId cancel, periodic NewJob, idle after Found.
async fn run_benchmark_with_jobs(
    cpu_engine: Option<Arc<dyn MinerEngine>>,
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    effective_cpu_workers: usize,
    effective_gpu_devices: usize,
    duration: u64,
    job_interval: f64,
    difficulty: U512,
) {
    let total_workers = effective_cpu_workers + effective_gpu_devices;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let current_job_id = Arc::new(AtomicU64::new(0));
    let job_ctx: Arc<RwLock<JobContext>> = Arc::new(RwLock::new(JobContext::new(
        random_header(),
        difficulty,
    )));
    let total_hashes = Arc::new(Mutex::new(0u64));
    let finds = Arc::new(AtomicU64::new(0));
    let jobs_started = Arc::new(AtomicU64::new(0));
    let phase_stats: Arc<Mutex<Vec<PhaseAccum>>> = Arc::new(Mutex::new(
        (0..total_workers)
            .map(|i| PhaseAccum {
                device_index: i.saturating_sub(effective_cpu_workers),
                ..Default::default()
            })
            .collect(),
    ));

    // Publish job 1 (ctx before id bump so workers never see a stale header).
    {
        *job_ctx.write().unwrap() = JobContext::new(random_header(), difficulty);
        let id = current_job_id.fetch_add(1, Ordering::SeqCst) + 1;
        jobs_started.store(id, Ordering::Relaxed);
    }

    println!("⛏️  Starting job-simulation benchmark...");

    let mut handles = Vec::new();
    let benchmark_start = Instant::now();

    for worker_id in 0..total_workers {
        let is_gpu = worker_id >= effective_cpu_workers;
        let engine = if !is_gpu {
            cpu_engine.as_ref().unwrap().clone()
        } else {
            gpu_engine.as_ref().unwrap().clone()
        };

        let stop = stop_flag.clone();
        let job_id_counter = current_job_id.clone();
        let job_ctx = job_ctx.clone();
        let hashes = total_hashes.clone();
        let finds = finds.clone();
        let phase_stats = phase_stats.clone();

        let handle = thread::spawn(move || {
            let stride = U512::from(1_000_000_000_000u64);
            let worker_start = U512::from(worker_id as u64).saturating_mul(stride);

            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let my_job_id = job_id_counter.load(Ordering::SeqCst);
                if my_job_id == 0 {
                    thread::sleep(Duration::from_millis(1));
                    continue;
                }

                let ctx = job_ctx.read().unwrap().clone();
                let cancel_check = JobIdCancelCheck {
                    current_job_id: &job_id_counter,
                    my_job_id,
                };

                // Match serve: open-ended range; engine batches internally.
                let range = EngineRange {
                    start: worker_start,
                    end: U512::MAX,
                };

                let result = engine.search_range(&ctx, range, &cancel_check);

                if is_gpu {
                    if let Some(t) = engine_gpu::GpuEngine::take_last_search_timing() {
                        let mut stats = phase_stats.lock().unwrap();
                        let row = &mut stats[worker_id];
                        row.device_index = t.device_index;
                        row.wind_up.push(t.wind_up.as_secs_f64() * 1000.0);
                        row.busy.push(t.busy.as_secs_f64() * 1000.0);
                        row.wind_down.push(t.wind_down.as_secs_f64() * 1000.0);
                        row.samples += 1;
                    }
                }

                match result {
                    engine_cpu::EngineStatus::Found { hash_count, .. } => {
                        *hashes.lock().unwrap() += hash_count;
                        finds.fetch_add(1, Ordering::Relaxed);
                        // Serve parks the GPU until the next NewJob after a find.
                        while !stop.load(Ordering::Relaxed)
                            && job_id_counter.load(Ordering::SeqCst) == my_job_id
                        {
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                    engine_cpu::EngineStatus::Cancelled { hash_count }
                    | engine_cpu::EngineStatus::Exhausted { hash_count } => {
                        *hashes.lock().unwrap() += hash_count;
                    }
                    engine_cpu::EngineStatus::DeviceLost { hash_count } => {
                        *hashes.lock().unwrap() += hash_count;
                        break;
                    }
                    engine_cpu::EngineStatus::Running { .. } => {}
                }
            }

            engine_gpu::GpuEngine::clear_worker_resources();
        });

        handles.push(handle);
    }

    // Job feeder (simulated node)
    let feeder_stop = stop_flag.clone();
    let feeder_job_id = current_job_id.clone();
    let feeder_ctx = job_ctx.clone();
    let feeder_jobs = jobs_started.clone();
    let feeder = thread::spawn(move || {
        let interval = Duration::from_secs_f64(job_interval);
        while !feeder_stop.load(Ordering::Relaxed) {
            thread::sleep(interval);
            if feeder_stop.load(Ordering::Relaxed) {
                break;
            }
            *feeder_ctx.write().unwrap() = JobContext::new(random_header(), difficulty);
            let id = feeder_job_id.fetch_add(1, Ordering::SeqCst) + 1;
            feeder_jobs.store(id, Ordering::Relaxed);
            log::info!("simulated NewJob id={id}");
        }
    });

    let stats = Some((finds.clone(), jobs_started.clone()));
    progress_and_join(
        handles,
        stop_flag.clone(),
        total_hashes,
        benchmark_start,
        duration,
        stats,
        Some(current_job_id.clone()),
    )
    .await;

    stop_flag.store(true, Ordering::Relaxed);
    current_job_id.fetch_add(1, Ordering::SeqCst);
    let _ = feeder.join();

    print_phase_report(&phase_stats.lock().unwrap(), job_interval);
}

fn print_phase_report(stats: &[PhaseAccum], job_interval: f64) {
    let gpu_rows: Vec<&PhaseAccum> = stats.iter().filter(|s| s.samples > 0).collect();
    if gpu_rows.is_empty() {
        return;
    }

    println!();
    println!("📐 Job-phase timings (per GPU worker, ms)");
    println!("=========================================");
    println!(
        "{:<8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7} {:>8}",
        "device", "samples", "wind_up", "busy", "wind_down", "wu_p50", "busy_p50", "busy%", "vs_int%"
    );

    let interval_ms = job_interval * 1000.0;
    for row in gpu_rows {
        let mut wu = row.wind_up.clone();
        let mut bu = row.busy.clone();
        let mut wd = row.wind_down.clone();
        let wu_mean = mean_ms(&wu);
        let bu_mean = mean_ms(&bu);
        let wd_mean = mean_ms(&wd);
        let wu_p50 = p50_ms(&mut wu);
        let bu_p50 = p50_ms(&mut bu);
        let _wd_p50 = p50_ms(&mut wd);
        let phase_sum = wu_mean + bu_mean + wd_mean;
        let busy_share = if phase_sum > 0.0 {
            (bu_mean / phase_sum) * 100.0
        } else {
            0.0
        };
        let vs_interval = if interval_ms > 0.0 {
            (bu_mean / interval_ms) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<8} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>6.1}% {:>7.0}%",
            row.device_index,
            row.samples,
            wu_mean,
            bu_mean,
            wd_mean,
            wu_p50,
            bu_p50,
            busy_share,
            vs_interval
        );
    }
    println!(
        "(wind_up = setup→first batch; busy = GPU batch wall time; wind_down = cancel seen→return)"
    );
    println!(
        "busy% = share of search phases; vs_int% = mean busy / job-interval ({job_interval:.2}s) — can exceed 100% when a batch overruns the interval"
    );
}

async fn progress_and_join(
    handles: Vec<thread::JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
    total_hashes: Arc<Mutex<u64>>,
    benchmark_start: Instant,
    duration: u64,
    job_stats: Option<(Arc<AtomicU64>, Arc<AtomicU64>)>,
    job_id_to_bump: Option<Arc<AtomicU64>>,
) {
    let mut last_update = Instant::now();

    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;

        if benchmark_start.elapsed() >= Duration::from_secs(duration) {
            stop_flag.store(true, Ordering::Relaxed);
            // Cancel in-flight GPU batches (JobIdCancelCheck) and wake Found-waiters.
            if let Some(ref job_id) = job_id_to_bump {
                job_id.fetch_add(1, Ordering::SeqCst);
            }
            break;
        }

        if last_update.elapsed() >= Duration::from_secs(1) {
            let current = *total_hashes.lock().unwrap();
            let elapsed = benchmark_start.elapsed().as_secs_f64();
            if current > 0 {
                let rate = current as f64 / elapsed;
                if let Some((finds, jobs)) = &job_stats {
                    println!(
                        "⏱️  {:.1}s - {} H/s (jobs={}, finds={})",
                        elapsed,
                        format_hash_rate(rate),
                        jobs.load(Ordering::Relaxed),
                        finds.load(Ordering::Relaxed)
                    );
                } else {
                    println!("⏱️  {:.1}s - {} H/s", elapsed, format_hash_rate(rate));
                }
            }
            last_update = Instant::now();
        }
    }

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
    if let Some((finds, jobs)) = job_stats {
        println!("Jobs started: {}", jobs.load(Ordering::Relaxed));
        println!("Solutions found: {}", finds.load(Ordering::Relaxed));
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
