use clap::{Parser, ValueEnum};
use miner_service::{run, EngineSelection, ServiceConfig};

/// Quantus External Miner CLI
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port number to listen on for the miner HTTP API
    #[arg(short, long, env = "MINER_PORT", default_value_t = 9833)]
    port: u16,

    /// Number of worker threads (logical CPUs) to use for mining (defaults to all available)
    #[arg(long = "workers", env = "MINER_WORKERS")]
    workers: Option<usize>,

    /// Optional Prometheus metrics exporter port; if omitted, metrics are disabled
    #[arg(long, env = "MINER_METRICS_PORT")]
    metrics_port: Option<u16>,

    /// Target milliseconds for per-thread progress updates (chunking).
    /// Smaller values increase metrics freshness but add a bit of overhead.
    #[arg(long = "progress-chunk-ms", env = "MINER_PROGRESS_CHUNK_MS")]
    progress_chunk_ms: Option<u64>,

    /// Mining engine to use (default: cpu-fast). Options: cpu-baseline, cpu-fast
    #[arg(long, env = "MINER_ENGINE", value_enum, default_value_t = EngineCli::CpuFast)]
    engine: EngineCli,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum EngineCli {
    /// Baseline CPU engine (reference implementation)
    CpuBaseline,
    /// Optimized CPU engine (incremental precompute + step_mul)
    CpuFast,
    // Cuda,      // planned: CUDA GPU engine
    // Opencl,    // planned: OpenCL GPU engine
}

impl From<EngineCli> for EngineSelection {
    fn from(value: EngineCli) -> Self {
        match value {
            EngineCli::CpuBaseline => EngineSelection::CpuBaseline,
            EngineCli::CpuFast => EngineSelection::CpuFast,
            // EngineCli::Cuda => EngineSelection::Cuda,
            // EngineCli::Opencl => EngineSelection::OpenCl,
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize logger early to capture startup messages.
    // If RUST_LOG is not set, default to info level for our app.
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // Log effective configuration
    log::info!("Starting external miner service...");
    log::info!("API listening port: {}", args.port);
    match args.workers {
        Some(n) if n > 0 => log::info!("Using specified number of workers: {}", n),
        Some(_) => {
            log::warn!("Workers must be positive. Defaulting to all available logical CPUs.")
        }
        None => log::info!("Using all available logical CPUs (auto-detected)."),
    }
    match args.metrics_port {
        Some(p) => log::info!("Metrics enabled on port {}", p),
        None => log::info!("Metrics disabled (no --metrics-port provided)"),
    }
    log::info!("Selected engine: {:?}", args.engine);

    let config = ServiceConfig {
        port: args.port,
        workers: args.workers,
        metrics_port: args.metrics_port,
        progress_chunk_ms: args.progress_chunk_ms,
        engine: args.engine.into(),
    };

    if let Err(e) = run(config).await {
        log::error!("Miner service terminated with error: {e:?}");
        std::process::exit(1);
    }
}
