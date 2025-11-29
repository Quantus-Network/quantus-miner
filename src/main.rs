use clap::Parser;
use quantus_miner::*;
// Import everything from lib.rs
use std::net::SocketAddr;
use warp::Filter;
use env_logger::Env;

/// Quantus External Miner Service
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port number to listen on
    #[arg(short, long, env = "MINER_PORT", default_value_t = 9833)]
    port: u16,

    /// Port number to listen on for Prometheus metrics
    #[arg(long, env = "METRICS_PORT", default_value_t = 9900)]
    metrics_port: u16,

    /// Number of CPU cores to use for mining
    #[arg(long, env = "MINER_CORES")]
    num_cores: Option<usize>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse(); // Parse args - this handles --help and --version
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    log::info!("Starting external miner service...");

    let mut state = MiningState::new();

    if let Some(num_cores) = args.num_cores {
        if num_cores > 0 {
            log::info!("Using specified number of cores: {}", num_cores);
            state.num_cores = num_cores;
        } else {
            log::warn!("Number of cores must be positive. Defaulting to all available cores.");
        }
    } else {
        log::info!("Using all available cores: {}", state.num_cores);
    }

    if state.num_cores == 0 {
        log::error!("Invalid core count detected. This should not happen.");
        std::process::exit(1);
    }

    // --- Start the mining loop ---
    state.start_mining_loop().await;

    // --- Start Metrics Server ---
    let metrics_state = state.clone();
    let metrics_port = args.metrics_port;
    tokio::spawn(async move {
        log::info!("Starting metrics server on port {}", metrics_port);
        let metrics_route = warp::get()
            .and(warp::path("metrics"))
            .and(warp::any().map(move || metrics_state.clone()))
            .map(|state: MiningState| {
                // We need to read from async lock
                // But map is sync. 
                // Use and_then for async handler
                state
            })
            .and_then(|state: MiningState| async move {
                let metrics = state.metrics.read().await;
                Ok::<_, std::convert::Infallible>(warp::reply::with_header(
                    metrics.format_prometheus(),
                    "Content-Type",
                    "text/plain; version=0.0.4; charset=utf-8",
                ))
            });

        warp::serve(metrics_route)
            .run(([0, 0, 0, 0], metrics_port))
            .await;
    });

    // --- Set up Warp filters ---
    let state_clone = state.clone(); // Clone state for the filter closure
    let state_filter = warp::any().map(move || state_clone.clone());

    // Use handle_mine_request from lib.rs
    let mine_route = warp::post()
        .and(warp::path("mine"))
        .and(warp::body::json()) // Expect MiningRequest from lib.rs
        .and(state_filter.clone())
        .and_then(handle_mine_request);

    // Use handle_result_request from lib.rs
    let result_route = warp::get()
        .and(warp::path("result"))
        .and(warp::path::param())
        .and(state_filter.clone())
        .and_then(handle_result_request);

    // Use handle_cancel_request from lib.rs
    let cancel_route = warp::post()
        .and(warp::path("cancel"))
        .and(warp::path::param())
        .and(state_filter.clone())
        .and_then(handle_cancel_request);

    let routes = mine_route.or(result_route).or(cancel_route);

    // Use the port from parsed arguments
    let addr: SocketAddr = ([0, 0, 0, 0], args.port).into();
    log::debug!("Server starting on {}", addr);
    warp::serve(routes).run(addr).await;
}
