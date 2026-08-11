//! Quantus captcha share pool.
//!
//! Sits between a quantus-node (external miner protocol) and browser captcha
//! solvers: hands out low-difficulty share challenges over the real block
//! header, verifies solves, mints single-use tokens for site backends, and
//! submits any share that happens to meet full network difficulty as a block.

mod http;
mod rate_limit;
mod state;
mod upstream;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use primitive_types::U512;

use crate::rate_limit::SessionRateLimit;

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

    /// Shared auth token from the node's logs / miner-auth-token file.
    /// Required when `--node-addr` is set.
    #[arg(long, env = "POOL_AUTH_TOKEN", conflicts_with = "auth_token_file")]
    auth_token: Option<String>,

    /// Path to a file containing the shared auth token (trimmed).
    #[arg(long, env = "POOL_AUTH_TOKEN_FILE", conflicts_with = "auth_token")]
    auth_token_file: Option<PathBuf>,

    /// SHA-256 fingerprint of the node's miner TLS certificate (hex).
    /// Required when `--node-addr` is set.
    #[arg(
        long,
        env = "POOL_TLS_CERT_SHA256",
        conflicts_with = "tls_cert_sha256_file"
    )]
    tls_cert_sha256: Option<String>,

    /// Path to a file containing the node's miner TLS cert SHA-256 fingerprint.
    #[arg(
        long,
        env = "POOL_TLS_CERT_SHA256_FILE",
        conflicts_with = "tls_cert_sha256"
    )]
    tls_cert_sha256_file: Option<PathBuf>,

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

    /// Max /api/session issuances per client IP per minute.
    #[arg(long, default_value = "60", env = "POOL_SESSIONS_PER_IP_PER_MIN")]
    sessions_per_ip_per_min: u32,
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

    let limiter = Arc::new(rate_limit::SessionIssuerLimiter::new(SessionRateLimit {
        max_per_ip: args.sessions_per_ip_per_min,
        window: Duration::from_secs(60),
    }));

    // Periodic cleanup of expired sessions/tokens and stale rate-limit windows.
    {
        let state = state.clone();
        let limiter = limiter.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                state.gc();
                limiter.gc();
            }
        });
    }

    // Job source.
    {
        let state = state.clone();
        match args.node_addr {
            Some(addr) => {
                let auth_token = resolve_auth_token(args.auth_token, args.auth_token_file)?;
                let tls_cert_sha256 =
                    resolve_tls_cert_sha256(args.tls_cert_sha256, args.tls_cert_sha256_file)?;
                log::info!("Upstream: quantus-node at {}", addr);
                tokio::spawn(upstream::run_node_client(
                    state,
                    addr,
                    auth_token,
                    tls_cert_sha256,
                    solution_rx,
                ));
            }
            None => {
                if args.auth_token.is_some()
                    || args.auth_token_file.is_some()
                    || args.tls_cert_sha256.is_some()
                    || args.tls_cert_sha256_file.is_some()
                {
                    log::warn!("Ignoring auth/TLS pin flags in standalone mode (no --node-addr)");
                }
                log::warn!("No --node-addr given: running STANDALONE with synthetic jobs");
                tokio::spawn(upstream::run_standalone(
                    state,
                    Duration::from_secs(args.standalone_job_secs),
                    solution_rx,
                ));
            }
        }
    }

    http::serve(state, limiter, args.http_addr, args.serve_dir).await;
    Ok(())
}

fn resolve_auth_token(
    auth_token: Option<String>,
    auth_token_file: Option<PathBuf>,
) -> anyhow::Result<String> {
    resolve_required_secret(
        auth_token,
        auth_token_file,
        "--auth-token",
        "--auth-token-file",
        "when --node-addr is set, pass --auth-token <TOKEN> or --auth-token-file <PATH> \
         (copy from the node's logs or its miner-auth-token file)",
    )
}

fn resolve_tls_cert_sha256(value: Option<String>, file: Option<PathBuf>) -> anyhow::Result<String> {
    resolve_required_secret(
        value,
        file,
        "--tls-cert-sha256",
        "--tls-cert-sha256-file",
        "when --node-addr is set, pass --tls-cert-sha256 <HEX> or \
         --tls-cert-sha256-file <PATH> (copy from the node's logs or its \
         miner-tls-cert-sha256 file)",
    )
}

fn resolve_required_secret(
    value: Option<String>,
    file: Option<PathBuf>,
    value_flag: &str,
    file_flag: &str,
    missing_msg: &str,
) -> anyhow::Result<String> {
    if let Some(value) = value {
        let value = value.trim().to_string();
        anyhow::ensure!(!value.is_empty(), "{value_flag} is empty");
        return Ok(value);
    }
    if let Some(path) = file {
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("failed to read {file_flag} {}: {}", path.display(), e))?;
        let value = contents.trim().to_string();
        anyhow::ensure!(!value.is_empty(), "{file_flag} {} is empty", path.display());
        return Ok(value);
    }
    anyhow::bail!("{missing_msg}");
}
