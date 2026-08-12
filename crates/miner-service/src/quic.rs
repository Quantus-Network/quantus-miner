//! QUIC client for connecting to blockchain nodes.
//!
//! This module provides a QUIC client that connects to a blockchain node
//! and handles bidirectional streaming for receiving mining jobs and
//! sending results.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use engine_cpu::MinerEngine;
use primitive_types::U512;

use quantus_miner_api::{
    read_message, write_message, ApiResponseStatus, MinerMessage, MiningResult,
};

use crate::{EngineType, WorkerPool};
use pow_core::format_hashrate;

/// Connect to a node and start mining.
///
/// This function connects to the node, receives mining jobs, and sends results.
/// It automatically reconnects if the connection is lost.
///
/// Uses a persistent worker pool to avoid thread creation overhead between jobs.
pub async fn connect_and_mine(
    node_addr: SocketAddr,
    auth_token: &str,
    tls_cert_sha256: &str,
    cpu_engine: Option<Arc<dyn MinerEngine>>,
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    cpu_workers: usize,
    gpu_devices: usize,
) -> anyhow::Result<()> {
    // Create persistent worker pool once - it lives for the entire miner lifetime
    let worker_pool = WorkerPool::new(cpu_engine, gpu_engine, cpu_workers, gpu_devices);

    let mut reconnect_delay = Duration::from_secs(1);
    const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);

    loop {
        log::info!("⛏️ Connecting to node at {}...", node_addr);

        match establish_connection(node_addr, auth_token, tls_cert_sha256).await {
            Ok((connection, send, recv)) => {
                log::info!("⛏️ Connected to node at {}", node_addr);

                let mut authenticated = false;
                if let Err(e) =
                    handle_connection(connection, send, recv, &worker_pool, &mut authenticated)
                        .await
                {
                    // Cancel any running job when connection drops
                    worker_pool.cancel();
                    if e.downcast_ref::<quic_transport::PermanentConnectError>().is_some() {
                        log::error!("⛏️ Permanent connection error (not retrying): {e}");
                        return Err(e);
                    }
                    log::info!("⛏️ Connection lost: {}", e);
                }
                // Only clear backoff after the node accepted auth (first NewJob).
                // connect() already treats explicit "auth failed" as permanent;
                // this covers any other post-Ready close that looked like success.
                if authenticated {
                    reconnect_delay = Duration::from_secs(1);
                }
            }
            Err(e) => {
                if e.downcast_ref::<quic_transport::PermanentConnectError>().is_some() {
                    log::error!("⛏️ Permanent connection error (not retrying): {e}");
                    return Err(e);
                }
                log::warn!("⛏️ Failed to connect to node: {}", e);
            }
        }

        log::info!("⛏️ Reconnecting in {:?}...", reconnect_delay);
        tokio::time::sleep(reconnect_delay).await;
        reconnect_delay = (reconnect_delay * 2).min(MAX_RECONNECT_DELAY);
    }
}

/// Establish a QUIC connection to the node (shared transport crate).
async fn establish_connection(
    addr: SocketAddr,
    auth_token: &str,
    tls_cert_sha256: &str,
) -> anyhow::Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream)> {
    let result = quic_transport::connect(addr, auth_token, tls_cert_sha256).await?;
    log::info!(
        "⛏️ QUIC connection and bidirectional stream established to {}",
        addr
    );
    Ok(result)
}

/// Helper to send a message while monitoring connection health.
async fn send_message_checked(
    connection: &quinn::Connection,
    send: &mut quinn::SendStream,
    msg: &MinerMessage,
) -> anyhow::Result<()> {
    tokio::select! {
        biased;
        reason = connection.closed() => {
            Err(anyhow::anyhow!("Connection closed: {}", reason))
        }
        result = write_message(send, msg) => {
            result.map_err(|e| anyhow::anyhow!("Failed to send message: {}", e))
        }
    }
}

/// Handle an established connection, receiving jobs and sending results.
///
/// Sets `authenticated` when the first `NewJob` arrives (proof the node
/// accepted our Ready token).
async fn handle_connection(
    connection: quinn::Connection,
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    worker_pool: &WorkerPool,
    authenticated: &mut bool,
) -> anyhow::Result<()> {
    use crossbeam_channel::RecvTimeoutError;

    // Set static metrics once per connection
    metrics::set_effective_cpus(num_cpus::get() as i64);
    metrics::set_workers(worker_pool.worker_count() as i64);
    metrics::set_cpu_workers(worker_pool.cpu_worker_count() as i64);
    metrics::set_gpu_devices(worker_pool.gpu_worker_count() as i64);
    metrics::reset_hash_tracker();

    // Current job state
    // - node_job_id: The string ID from the node (e.g., "27") - used in protocol messages
    // - internal_job_id: Our internal numeric ID from WorkerPool - used to detect stale results
    let mut node_job_id: Option<String> = None;
    let mut internal_job_id: u64 = 0;
    let mut job_start_time: Option<Instant> = None;
    let mut cpu_hashes: u64 = 0;
    let mut gpu_hashes: u64 = 0;
    let mut result_sent_for_current_job = false;

    log::info!("⛏️ Waiting for mining jobs from node...");

    loop {
        // Poll for worker results (non-blocking via spawn_blocking)
        let poll_result = if node_job_id.is_some() && !result_sent_for_current_job {
            let rx = worker_pool.result_receiver().clone();
            tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_millis(10)))
                .await
                .unwrap_or(Err(RecvTimeoutError::Disconnected))
        } else {
            Err(RecvTimeoutError::Timeout)
        };

        // Handle worker result if any
        if let Ok(worker_result) = poll_result {
            // Track hashes by engine type (always, even for stale results)
            match worker_result.engine_type {
                EngineType::Cpu => {
                    cpu_hashes += worker_result.hash_count;
                    metrics::record_cpu_hashes(worker_result.hash_count);
                }
                EngineType::Gpu => {
                    gpu_hashes += worker_result.hash_count;
                    metrics::record_gpu_hashes(worker_result.hash_count);
                }
            }

            // Check if this result is for the current job (not stale)
            if worker_result.job_id != internal_job_id {
                log::debug!(
                    "⏰ Discarding stale result from worker {} (result job_id {} != current {})",
                    worker_result.thread_id,
                    worker_result.job_id,
                    internal_job_id
                );
                continue;
            }

            // Only send result for the FIRST solution found for THIS job
            if let Some(candidate) = worker_result.candidate {
                if !result_sent_for_current_job {
                    if let Some(ref job_id) = node_job_id {
                        let total_hashes = cpu_hashes + gpu_hashes;
                        let elapsed = job_start_time
                            .map(|t| t.elapsed().as_secs_f64())
                            .unwrap_or(0.0);

                        log::info!(
                            "⛏️ Job {job_id} completed: {total_hashes} hashes in {elapsed:.2}s ({})",
                            format_hashrate(total_hashes as f64 / elapsed.max(0.001))
                        );

                        // Mark as sent BEFORE sending to prevent duplicates
                        result_sent_for_current_job = true;
                        worker_pool.cancel();
                        metrics::set_active_jobs(0);

                        let result = MiningResult {
                            status: ApiResponseStatus::Completed,
                            job_id: job_id.clone(),
                            nonce: Some(format!("{:x}", candidate.nonce)),
                            work: Some(hex::encode(candidate.work)),
                            hash_count: total_hashes,
                            elapsed_time: elapsed,
                            miner_id: None,
                        };

                        let msg = MinerMessage::JobResult(result);
                        send_message_checked(&connection, &mut send, &msg).await?;
                    }
                }
            }
        }

        // Check for incoming messages and connection health
        tokio::select! {
            biased;

            reason = connection.closed() => {
                let msg = reason.to_string();
                if !*authenticated && msg.to_ascii_lowercase().contains("auth") {
                    return Err(quic_transport::PermanentConnectError(format!(
                        "node rejected miner auth ({msg}); check --auth-token / miner-auth-token"
                    ))
                    .into());
                }
                return Err(anyhow::anyhow!("Connection closed: {}", msg));
            }

            msg_result = read_message(&mut recv) => {
                match msg_result {
                    Ok(MinerMessage::NewJob(request)) => {
                        *authenticated = true;
                        log::info!(
                            "⛏️ Received job: id={}, hash=0x{}",
                            request.job_id,
                            request.mining_hash
                        );

                        // Parse header hash
                        let header_hash: [u8; 32] = match hex::decode(&request.mining_hash) {
                            Ok(bytes) if bytes.len() == 32 => bytes.try_into().unwrap(),
                            _ => {
                                log::warn!("Invalid mining_hash in request");
                                let result = MiningResult {
                                    status: ApiResponseStatus::Failed,
                                    job_id: request.job_id,
                                    nonce: None,
                                    work: None,
                                    hash_count: 0,
                                    elapsed_time: 0.0,
                                    miner_id: None,
                                };
                                let msg = MinerMessage::JobResult(result);
                                send_message_checked(&connection, &mut send, &msg).await?;
                                continue;
                            }
                        };

                        // Parse difficulty
                        let difficulty = match U512::from_dec_str(&request.difficulty) {
                            Ok(d) => d,
                            Err(_) => {
                                log::warn!("Invalid difficulty in request: parse error");
                                let result = MiningResult {
                                    status: ApiResponseStatus::Failed,
                                    job_id: request.job_id,
                                    nonce: None,
                                    work: None,
                                    hash_count: 0,
                                    elapsed_time: 0.0,
                                    miner_id: None,
                                };
                                let msg = MinerMessage::JobResult(result);
                                send_message_checked(&connection, &mut send, &msg).await?;
                                continue;
                            }
                        };

                        // Reject zero difficulty to prevent division-by-zero panic in JobContext::new
                        if difficulty.is_zero() {
                            log::warn!("Invalid difficulty in request: zero is not allowed");
                            let result = MiningResult {
                                status: ApiResponseStatus::Failed,
                                job_id: request.job_id,
                                nonce: None,
                                work: None,
                                hash_count: 0,
                                elapsed_time: 0.0,
                                miner_id: None,
                            };
                            let msg = MinerMessage::JobResult(result);
                            send_message_checked(&connection, &mut send, &msg).await?;
                            continue;
                        }

                        // Reset state for new job
                        cpu_hashes = 0;
                        gpu_hashes = 0;
                        job_start_time = Some(Instant::now());
                        node_job_id = Some(request.job_id.clone());
                        result_sent_for_current_job = false;

                        log::debug!("Starting job {}", request.job_id);
                        metrics::set_active_jobs(1);

                        // start_job returns the internal job ID used to detect stale results
                        internal_job_id = worker_pool.start_job(header_hash, difficulty);
                    }
                    Ok(MinerMessage::JobResult(_)) => {
                        log::warn!("Received unexpected JobResult from node");
                    }
                    Ok(MinerMessage::Ready { .. }) => {
                        log::warn!("Received unexpected Ready from node");
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        return Err(anyhow::anyhow!("Node disconnected"));
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!("Read error: {}", e));
                    }
                }
            }

            // Short sleep to yield when no messages
            _ = tokio::time::sleep(Duration::from_millis(1)) => {}
        }
    }
}
