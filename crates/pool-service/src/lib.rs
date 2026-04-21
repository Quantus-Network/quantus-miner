//! Mining pool service for browser-based Quantus miners.
//!
//! This module provides a pool that:
//! - Connects to a Quantus node via QUIC to receive mining jobs
//! - Accepts browser miners via WebSocket (WSS)
//! - Distributes jobs to connected browser miners
//! - Tracks block wins per miner address
//! - Batches payouts (16 wins per payout transaction)
//!
//! # Architecture
//!
//! ```text
//! Browser Miners ──WSS──► Pool ──QUIC──► Quantus Node
//!      │                   │
//!      │                   ├── Tracks wins per address
//!      │                   └── Batches payouts (16 wins)
//!      │
//!      └── Each miner registers with their rewards address
//! ```

mod coordinator;
mod node_client;
mod websocket;
mod wins;

pub use coordinator::PoolCoordinator;
pub use wins::{BlockWin, WinTracker};

use std::net::SocketAddr;

/// Configuration for the pool service.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Address of the Quantus node to connect to (QUIC).
    pub node_addr: SocketAddr,
    /// Port to listen for WebSocket connections from browser miners.
    pub ws_port: u16,
    /// The pool operator's inner hash for receiving block rewards.
    /// This is used when submitting solutions to the node.
    pub pool_inner_hash: [u8; 32],
    /// Number of block wins to accumulate before triggering a batch payout.
    pub payout_batch_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            node_addr: "127.0.0.1:9833".parse().unwrap(),
            ws_port: 9834,
            pool_inner_hash: [0u8; 32],
            payout_batch_size: 16,
        }
    }
}

/// Start the pool service.
pub async fn run(config: PoolConfig) -> anyhow::Result<()> {
    log::info!("Starting Quantus mining pool...");
    log::info!("  Node address: {}", config.node_addr);
    log::info!("  WebSocket port: {}", config.ws_port);
    log::info!("  Payout batch size: {} blocks", config.payout_batch_size);

    let coordinator = PoolCoordinator::new(config);
    coordinator.run().await
}
