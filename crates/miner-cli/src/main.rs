use clap::{Parser, ValueEnum};
use miner_service::{run, EngineSelection, ServiceConfig};
use std::net::SocketAddr;

/// Quantus External Miner CLI
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port number to listen on for the miner HTTP API
    #[arg(short, long, env = "MINER_PORT", default_value_t = 9833)]
    port: u16,

    /// Number of CPU cores to use for mining (defaults to all logical CPUs)
    #[arg(long, env = "MINER_CORES")]
    num_cores: Option<usize>,

    /// Optional Prometheus metrics exporter port; if omitted, metrics are disabled
    #[arg(long, env = "MINER_METRICS_PORT")]
    metrics_port: Option<u16>,

    /// Mining engine to use
    #[arg(long, env = "MINER_ENGINE", value_enum, default_value_t = EngineCli::CpuBaseline)]
    engine: EngineCli,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum EngineCli {
    /// Baseline CPU engine (reference implementation)
    CpuBaseline,
    // CpuFast,   // planned: incremental + Montgomery
    // Cuda,      // planned: CUDA GPU engine
    // Opencl,    // planned: OpenCL GPU engine
}

impl From<EngineCli> for EngineSelection {
    fn from(value: EngineCli) -> Self {
        match value {
            EngineCli::CpuBaseline => EngineSelection::CpuBaseline,
            // EngineCli::CpuFast => EngineSelection::CpuFast,
            // EngineCli::Cuda => EngineSelection::Cuda,
            // EngineCli::Opencl => EngineSelection::OpenCl,
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize logger early to capture startup messages
    env_logger::init();

    // Log effective configuration
    log::info!("Starting external miner service...");
    log::info!("API listening port: {}", args.port);
    match args.num_cores {
        Some(n) if n > 0 => log::info!("Using specified number of cores: {}", n),
        Some(_) => log::warn!("Number of cores must be positive. Defaulting to all available cores."),
        None => log::info!("Using all available cores (auto-detected)."),
    }
    match args.metrics_port {
        Some(p) => log::info!("Metrics enabled on port {}", p),
        None => log::info!("Metrics disabled (no --metrics-port provided)"),
    }
    log::info!("Selected engine: {:?}", args.engine);

    let config = ServiceConfig {
        port: args.port,
        num_cores: args.num_cores,
        metrics_port: args.metrics_port,
        engine: args.engine.into(),
    };

    if let Err(e) = run(config).await {
        log::error!("Miner service terminated with error: {e:?}");
        std::process::exit(1);
    }
}
