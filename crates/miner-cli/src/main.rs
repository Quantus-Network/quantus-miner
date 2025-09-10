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

    /// Mining engine to use (default: cpu-fast).
    /// Options: cpu-baseline, cpu-fast, cpu-chain-manipulator, cpu-montgomery, gpu-cuda, gpu-opencl
    /// Note: GPU engines are currently unimplemented and will return a clear error at runtime.
    #[arg(long, env = "MINER_ENGINE", value_enum, default_value_t = EngineCli::CpuFast)]
    engine: EngineCli,
}

#[allow(clippy::enum_variant_names)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum EngineCli {
    /// Baseline CPU engine (reference implementation)
    CpuBaseline,
    /// Optimized CPU engine (incremental precompute + step_mul)
    CpuFast,
    /// Montgomery-optimized CPU engine (fixed-width 512-bit ops)
    CpuMontgomery,
    /// Throttling CPU engine that slows per block to help reduce difficulty
    CpuChainManipulator,
    /// CUDA GPU engine (unimplemented; selecting will return an error)
    GpuCuda,
    /// OpenCL GPU engine (unimplemented; selecting will return an error)
    GpuOpencl,
}

impl From<EngineCli> for EngineSelection {
    fn from(value: EngineCli) -> Self {
        match value {
            EngineCli::CpuBaseline => EngineSelection::CpuBaseline,
            EngineCli::CpuFast => EngineSelection::CpuFast,
            EngineCli::CpuMontgomery => EngineSelection::CpuMontgomery,
            EngineCli::CpuChainManipulator => EngineSelection::CpuChainManipulator,
            EngineCli::GpuCuda => EngineSelection::GpuCuda,
            EngineCli::GpuOpencl => EngineSelection::GpuOpenCl,
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

    // Log effective configuration (concise; see ServiceConfig Display)
    log::info!("Starting external miner service...");

    let config = ServiceConfig {
        port: args.port,
        workers: args.workers,
        metrics_port: args.metrics_port,
        progress_chunk_ms: args.progress_chunk_ms,
        manip_solved_blocks: args.manip_solved_blocks,
        manip_base_delay_ns: args.manip_base_delay_ns,
        manip_step_batch: args.manip_step_batch,
        manip_throttle_cap: args.manip_throttle_cap,
        engine: args.engine.into(),
    };
    log::info!("Effective config: {}", config);

    if let Err(e) = run(config).await {
        log::error!("Miner service terminated with error: {e:?}");
        std::process::exit(1);
    }
}
