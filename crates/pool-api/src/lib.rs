//! Protocol types for Quantus mining pool communication.
//!
//! This crate defines the WebSocket protocol between browser miners and the pool.
//! The pool acts as an intermediary between browser miners and the Quantus node.
//!
//! # Protocol Flow
//!
//! ```text
//! Browser                          Pool                           Node
//!    │                               │                               │
//!    │── Register { address } ──────►│                               │
//!    │◄─ Registered { miner_id } ────│                               │
//!    │                               │                               │
//!    │── Ready ─────────────────────►│                               │
//!    │                               │◄── NewJob ────────────────────│
//!    │◄─ NewJob ─────────────────────│                               │
//!    │                               │                               │
//!    │   (browser mines locally)     │                               │
//!    │                               │                               │
//!    │── JobResult { nonce, ... } ──►│                               │
//!    │                               │── JobResult ─────────────────►│
//!    │                               │                               │
//!    │◄─ BlockWon { block } ─────────│   (if solution valid)         │
//!    │                               │                               │
//! ```

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Messages sent from browser miners to the pool.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MinerToPool {
    /// Register with the pool, providing the miner's rewards address.
    /// This must be sent before Ready.
    Register {
        /// The miner's wormhole address for receiving payouts.
        /// This should be a valid SS58-encoded address.
        address: String,
    },

    /// Signal readiness to receive mining jobs.
    /// Must be sent after Register.
    Ready,

    /// Submit a mining result (potential solution).
    JobResult {
        /// The job ID this result is for.
        job_id: String,
        /// The nonce that was found (hex-encoded U512, no 0x prefix).
        nonce: String,
        /// The work bytes (hex-encoded [u8; 64], 128 chars, no 0x prefix).
        work: String,
        /// Number of hashes computed.
        hash_count: u64,
        /// Time spent mining this job (seconds).
        elapsed_time: f64,
    },
}

/// Messages sent from the pool to browser miners.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PoolToMiner {
    /// Acknowledgment of successful registration.
    Registered {
        /// Unique ID assigned to this miner for the session.
        miner_id: u64,
    },

    /// A new mining job to work on.
    /// Receiving a new job implicitly cancels any previous job.
    NewJob {
        /// Unique job identifier.
        job_id: String,
        /// Header hash to mine (hex-encoded, 64 chars, no 0x prefix).
        mining_hash: String,
        /// Difficulty threshold (U512 as decimal string).
        difficulty: String,
    },

    /// Notification that this miner won a block.
    BlockWon {
        /// The block number that was won.
        block_number: u64,
        /// The job ID that was solved.
        job_id: String,
    },

    /// Notification that another miner found the solution first.
    JobCompleted {
        /// The job ID that was completed.
        job_id: String,
        /// Whether this miner was the winner.
        you_won: bool,
    },

    /// Error message from the pool.
    Error {
        /// Error description.
        message: String,
    },
}

/// A record of a block win for payout tracking.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BlockWin {
    /// The miner's rewards address.
    pub miner_address: String,
    /// The block number that was won.
    pub block_number: u64,
    /// The job ID that was solved.
    pub job_id: String,
    /// Timestamp when the block was won (Unix epoch seconds).
    pub timestamp: u64,
    /// Whether this win has been paid out.
    pub paid: bool,
}

/// Configuration for batch payouts.
#[derive(Debug, Clone)]
pub struct PayoutConfig {
    /// Number of block wins to accumulate before triggering a payout.
    pub batch_size: usize,
    /// The pool's inner hash for sending wormhole transactions.
    pub pool_inner_hash: [u8; 32],
}

impl Default for PayoutConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            pool_inner_hash: [0u8; 32],
        }
    }
}

// ============================================================================
// WebSocket Message Helpers
// ============================================================================

/// Write a pool message to a WebSocket sink.
pub async fn write_pool_message<S>(sink: &mut S, msg: &PoolToMiner) -> std::io::Result<()>
where
    S: futures_util::Sink<WsMessage> + Unpin,
    S::Error: std::fmt::Display,
{
    let json = serde_json::to_string(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    sink.send(WsMessage::Text(json))
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(())
}

/// Read a miner message from a WebSocket stream.
pub async fn read_miner_message<S>(stream: &mut S) -> std::io::Result<MinerToPool>
where
    S: futures_util::Stream<Item = Result<WsMessage, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        match stream.next().await {
            Some(Ok(WsMessage::Text(text))) => {
                return serde_json::from_str(&text)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e));
            }
            Some(Ok(WsMessage::Binary(data))) => {
                return serde_json::from_slice(&data)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e));
            }
            Some(Ok(WsMessage::Ping(_) | WsMessage::Pong(_))) => continue,
            Some(Ok(WsMessage::Close(_))) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "WebSocket closed",
                ));
            }
            Some(Ok(WsMessage::Frame(_))) => continue,
            Some(Err(e)) => {
                return Err(std::io::Error::other(e.to_string()));
            }
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "WebSocket stream ended",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miner_to_pool_serialization() {
        let msg = MinerToPool::Register {
            address: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"register\""));
        assert!(json.contains("\"address\":"));

        let msg = MinerToPool::Ready;
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"ready\""));

        let msg = MinerToPool::JobResult {
            job_id: "123".to_string(),
            nonce: "abc123".to_string(),
            work: "deadbeef".to_string(),
            hash_count: 1000,
            elapsed_time: 1.5,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"job_result\""));
    }

    #[test]
    fn test_pool_to_miner_serialization() {
        let msg = PoolToMiner::Registered { miner_id: 42 };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"registered\""));
        assert!(json.contains("\"miner_id\":42"));

        let msg = PoolToMiner::NewJob {
            job_id: "456".to_string(),
            mining_hash: "abcd".to_string(),
            difficulty: "1000".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"new_job\""));

        let msg = PoolToMiner::BlockWon {
            block_number: 12345,
            job_id: "789".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"block_won\""));
    }

    #[test]
    fn test_miner_to_pool_deserialization() {
        let json = r#"{"type":"register","address":"5GrwvaEF..."}"#;
        let msg: MinerToPool = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, MinerToPool::Register { .. }));

        let json = r#"{"type":"ready"}"#;
        let msg: MinerToPool = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, MinerToPool::Ready));
    }
}
