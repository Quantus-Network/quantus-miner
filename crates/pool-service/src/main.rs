//! Quantus captcha share pool.
//!
//! Sits between a quantus-node (external miner protocol) and browser captcha
//! solvers: hands out low-difficulty share challenges over the real block
//! header, verifies solves, mints single-use tokens for site backends, and
//! submits any share that happens to meet full network difficulty as a block.

mod http;
mod state;
mod upstream;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use primitive_types::U512;

#[derive(Parser, Debug)]
#[command(name = "pool-service", about = "Quantus captcha share pool")]
struct Args {
    /// HTTP listen address for the captcha/siteverify API.
    #[arg(long, default_value = "127.0.0.1:8787", env = "POOL_HTTP_ADDR")]
    http_addr: SocketAddr,

    /// Address of the quantus-node external-miner QUIC endpoint.
    /// If omitted, runs in standalone mode with synthetic jobs (demo only).
    #[arg(long, env = "POOL_NODE_ADDR")]
    node_addr: Option<SocketAddr>,

    /// Share difficulty: expected number of hashes per captcha solve.
    /// Measured browser WASM rate ≈ 120 kH/s on an M-series laptop, so
    /// 50000 ≈ 0.4 s desktop / ~2 s phone. Raise for stronger rate limiting.
    #[arg(long, default_value = "50000", env = "POOL_SHARE_DIFFICULTY")]
    share_difficulty: u64,

    /// Secret that protected-site backends must present to /siteverify.
    #[arg(long, default_value = "dev-secret", env = "POOL_SITE_SECRET")]
    site_secret: String,

    /// Optional directory of static files to serve (demo page / widget).
    #[arg(long, env = "POOL_SERVE_DIR")]
    serve_dir: Option<PathBuf>,

    /// Job rotation interval in standalone mode, seconds.
    #[arg(long, default_value = "20")]
    standalone_job_secs: u64,

    /// Maximum live captcha sessions; /api/session returns 503 when full.
    #[arg(long, default_value = "100000", env = "POOL_MAX_SESSIONS")]
    max_sessions: usize,

    /// Maximum live (unredeemed) share tokens; shares are refused when full.
    #[arg(long, default_value = "100000", env = "POOL_MAX_TOKENS")]
    max_tokens: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    anyhow::ensure!(args.share_difficulty > 0, "--share-difficulty must be > 0");
    if args.site_secret == "dev-secret" {
        log::warn!("Using default site secret; set --site-secret in production");
    }

    let (solution_tx, solution_rx) = tokio::sync::mpsc::channel(16);
    let state = state::PoolState::new(
        U512::from(args.share_difficulty),
        args.site_secret.clone(),
        solution_tx,
        state::Limits {
            max_sessions: args.max_sessions,
            max_tokens: args.max_tokens,
        },
    );

    // Periodic cleanup of expired sessions/tokens.
    {
        let state = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                state.gc();
            }
        });
    }

    // Job source.
    {
        let state = state.clone();
        match args.node_addr {
            Some(addr) => {
                log::info!("Upstream: quantus-node at {}", addr);
                tokio::spawn(upstream::run_node_client(state, addr, solution_rx));
            }
            None => {
                log::warn!("No --node-addr given: running STANDALONE with synthetic jobs");
                tokio::spawn(upstream::run_standalone(
                    state,
                    Duration::from_secs(args.standalone_job_secs),
                    solution_rx,
                ));
            }
        }
    }

    http::serve(state, args.http_addr, args.serve_dir).await;
    Ok(())
}
