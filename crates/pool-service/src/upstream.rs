//! Job sources: either a QUIC connection to a real quantus-node (speaking the
//! external miner protocol), or a standalone generator for demos/tests.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use primitive_types::U512;
use quantus_miner_api::{
    read_message, write_message, ApiResponseStatus, MinerMessage, MiningResult,
};
use tokio::sync::mpsc::Receiver;

use crate::state::{FoundBlock, Job, PoolState};

/// Connect to a node as an external miner and keep the pool's job current.
/// Solutions arriving on `solutions` are pushed upstream as JobResults.
pub async fn run_node_client(
    state: Arc<PoolState>,
    node_addr: SocketAddr,
    auth_token: String,
    tls_cert_sha256: String,
    mut solutions: Receiver<FoundBlock>,
) {
    let mut reconnect_delay = Duration::from_secs(1);
    const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);

    // A block taken off the channel but not yet acknowledged by a successful
    // upstream write. Kept across reconnects so a failed write doesn't drop
    // the block (the node may still accept it after reconnection if the job
    // hasn't moved on).
    let mut pending: Option<FoundBlock> = None;

    loop {
        log::info!("Connecting to node at {}...", node_addr);
        match quic_transport::connect(node_addr, &auth_token, &tls_cert_sha256).await {
            Ok((connection, mut send, mut recv)) => {
                log::info!("Connected to node at {}", node_addr);
                let mut authenticated = false;

                loop {
                    // Retry any unsubmitted block before doing anything else.
                    if let Some(block) = pending.take() {
                        match submit_block(&mut send, &block).await {
                            Ok(()) => {
                                log::info!(
                                    "Submitted block solution to node (job {})",
                                    block.job_id
                                );
                            }
                            Err(e) => {
                                log::error!("Failed to submit block solution, will retry: {}", e);
                                pending = Some(block);
                                break; // reconnect and retry
                            }
                        }
                    }

                    tokio::select! {
                        biased;

                        reason = connection.closed() => {
                            let msg = reason.to_string();
                            if !authenticated && msg.to_ascii_lowercase().contains("auth") {
                                log::error!(
                                    "Permanent auth rejection from node ({msg}); not retrying. \
                                     Check --auth-token / miner-auth-token"
                                );
                                return;
                            }
                            log::warn!("Node connection closed: {}", msg);
                            break;
                        }

                        // A captcha share met network difficulty: submit it.
                        block = solutions.recv() => {
                            if let Some(block) = block {
                                // Park it in `pending`; the top of the loop
                                // submits it and keeps it on write failure.
                                pending = Some(block);
                            }
                        }

                        msg = read_message(&mut recv) => {
                            match msg {
                                Ok(MinerMessage::NewJob(request)) => {
                                    authenticated = true;
                                    match parse_job(&request.job_id, &request.mining_hash, &request.difficulty) {
                                        Ok(job) => state.set_job(job),
                                        Err(e) => log::warn!("Ignoring malformed job from node: {}", e),
                                    }
                                }
                                Ok(other) => {
                                    log::warn!("Unexpected message from node: {:?}", other);
                                }
                                Err(e) => {
                                    log::warn!("Read error from node: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                }

                if authenticated {
                    reconnect_delay = Duration::from_secs(1);
                }
            }
            Err(e) => {
                if e.downcast_ref::<quic_transport::PermanentConnectError>()
                    .is_some()
                {
                    log::error!("Permanent connection error (not retrying): {e}");
                    return;
                }
                log::warn!("Failed to connect to node: {}", e);
            }
        }

        log::info!("Reconnecting in {:?}...", reconnect_delay);
        tokio::time::sleep(reconnect_delay).await;
        reconnect_delay = (reconnect_delay * 2).min(MAX_RECONNECT_DELAY);
    }
}

async fn submit_block(send: &mut quinn::SendStream, block: &FoundBlock) -> std::io::Result<()> {
    let result = MiningResult {
        status: ApiResponseStatus::Completed,
        job_id: block.job_id.clone(),
        nonce: Some(format!("{:x}", block.nonce)),
        work: Some(hex::encode(block.nonce.to_big_endian())),
        hash_count: 0,
        elapsed_time: 0.0,
        miner_id: None,
    };
    write_message(send, &MinerMessage::JobResult(result)).await
}

fn parse_job(job_id: &str, mining_hash: &str, difficulty: &str) -> anyhow::Result<Job> {
    let bytes = hex::decode(mining_hash)?;
    let header: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("mining_hash must be 32 bytes"))?;
    let network_difficulty =
        U512::from_dec_str(difficulty).map_err(|e| anyhow::anyhow!("bad difficulty: {:?}", e))?;
    if network_difficulty.is_zero() {
        anyhow::bail!("zero difficulty");
    }
    Ok(Job {
        job_id: job_id.to_string(),
        header,
        network_difficulty,
    })
}

/// Standalone mode: synthesize a fresh random job every `interval` so the
/// captcha works without a running node. Solutions are logged and dropped.
pub async fn run_standalone(
    state: Arc<PoolState>,
    interval: Duration,
    mut solutions: Receiver<FoundBlock>,
) {
    use rand::RngCore;
    let mut counter: u64 = 0;

    loop {
        let mut header = [0u8; 32];
        rand::rng().fill_bytes(&mut header);
        counter += 1;
        state.set_job(Job {
            job_id: format!("standalone-{}", counter),
            header,
            // Unreachable "network" difficulty: standalone mode never finds blocks.
            network_difficulty: U512::MAX,
        });

        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            block = solutions.recv() => {
                if let Some(block) = block {
                    log::info!("(standalone) would submit block for job {}", block.job_id);
                }
            }
        }
    }
}
