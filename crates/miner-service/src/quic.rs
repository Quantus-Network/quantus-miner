//! QUIC client for connecting to blockchain nodes.
//!
//! This module provides a QUIC client that connects to a blockchain node
//! and handles bidirectional streaming for receiving mining jobs and
//! sending results.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::RecvTimeoutError;
use engine_cpu::MinerEngine;
use primitive_types::U512;
use quinn::{ClientConfig, Endpoint};
use rustls::client::ServerCertVerified;
use tokio::sync::mpsc;

use quantus_miner_api::{read_message, write_message, ApiResponseStatus, MinerMessage, MiningResult};

use crate::{spawn_mining_workers, MiningCandidate};

/// Connect to a node and start mining.
///
/// This function connects to the node, receives mining jobs, and sends results.
/// It automatically reconnects if the connection is lost.
pub async fn connect_and_mine(
    node_addr: SocketAddr,
    cpu_engine: Option<Arc<dyn MinerEngine>>,
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    cpu_workers: usize,
    gpu_devices: usize,
) -> anyhow::Result<()> {
    let mut reconnect_delay = Duration::from_secs(1);
    const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);

    loop {
        log::info!("⛏️ Connecting to node at {}...", node_addr);

        match establish_connection(node_addr).await {
            Ok((connection, send, recv)) => {
                log::info!("⛏️ Connected to node at {}", node_addr);
                reconnect_delay = Duration::from_secs(1);

                if let Err(e) = handle_connection(
                    connection,
                    send,
                    recv,
                    cpu_engine.clone(),
                    gpu_engine.clone(),
                    cpu_workers,
                    gpu_devices,
                )
                .await
                {
                    log::info!("⛏️ Connection lost: {}", e);
                }
            }
            Err(e) => {
                log::warn!("⛏️ Failed to connect to node: {}", e);
            }
        }

        log::info!("⛏️ Reconnecting in {:?}...", reconnect_delay);
        tokio::time::sleep(reconnect_delay).await;
        reconnect_delay = (reconnect_delay * 2).min(MAX_RECONNECT_DELAY);
    }
}

/// Establish a QUIC connection to the node.
async fn establish_connection(
    addr: SocketAddr,
) -> anyhow::Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream)> {
    let mut crypto = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_custom_certificate_verifier(Arc::new(InsecureCertVerifier))
        .with_no_client_auth();

    crypto.alpn_protocols = vec![b"quantus-miner".to_vec()];

    let mut client_config = ClientConfig::new(Arc::new(crypto));

    let mut transport_config = quinn::TransportConfig::default();
    transport_config.keep_alive_interval(Some(Duration::from_secs(10)));
    transport_config.max_idle_timeout(Some(Duration::from_secs(60).try_into().unwrap()));
    client_config.transport_config(Arc::new(transport_config));

    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
    endpoint.set_default_client_config(client_config);

    let connection = endpoint.connect(addr, "localhost")?.await?;
    log::info!("⛏️ QUIC connection established to {}", addr);

    log::info!("⛏️ Opening bidirectional stream to node...");
    let (mut send, recv) = connection.open_bi().await?;

    // Send Ready message to establish the stream
    write_message(&mut send, &MinerMessage::Ready).await?;
    log::info!("⛏️ Bidirectional stream established");

    Ok((connection, send, recv))
}

/// Handle an established connection, receiving jobs and sending results.
async fn handle_connection(
    _connection: quinn::Connection,
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    cpu_engine: Option<Arc<dyn MinerEngine>>,
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    cpu_workers: usize,
    gpu_devices: usize,
) -> anyhow::Result<()> {
    // Channel for receiving mining results
    let (result_tx, mut result_rx) = mpsc::channel::<MiningResult>(16);

    // Current job's cancel flag
    let mut current_cancel: Option<Arc<AtomicBool>> = None;
    let mut current_job_id: Option<String> = None;

    log::info!("⛏️ Waiting for mining jobs from node...");

    loop {
        tokio::select! {
            biased;

            // Handle incoming messages from the node
            msg_result = read_message(&mut recv) => {
                match msg_result {
                    Ok(MinerMessage::NewJob(request)) => {
                        log::info!(
                            "⛏️ Received job: id={}, hash={}...",
                            request.job_id,
                            &request.mining_hash[..8]
                        );

                        // Cancel any existing job
                        if let Some(cancel) = current_cancel.take() {
                            log::debug!("Cancelling previous job");
                            cancel.store(true, Ordering::Relaxed);
                        }

                        // Parse and validate request
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
                                };
                                let _ = result_tx.send(result).await;
                                continue;
                            }
                        };

                        let difficulty = match U512::from_dec_str(&request.distance_threshold) {
                            Ok(d) => d,
                            Err(_) => {
                                log::warn!("Invalid difficulty in request");
                                let result = MiningResult {
                                    status: ApiResponseStatus::Failed,
                                    job_id: request.job_id,
                                    nonce: None,
                                    work: None,
                                    hash_count: 0,
                                    elapsed_time: 0.0,
                                };
                                let _ = result_tx.send(result).await;
                                continue;
                            }
                        };

                        // Create cancel flag for this job
                        let cancel = Arc::new(AtomicBool::new(false));
                        current_cancel = Some(cancel.clone());
                        current_job_id = Some(request.job_id.clone());

                        // Generate random nonce start
                        let nonce_start = generate_random_nonce_start();
                        let nonce_end = U512::MAX;

                        log::debug!(
                            "Starting job {} with nonce start: {:x}...",
                            request.job_id,
                            nonce_start
                        );

                        // Spawn mining task
                        let job_id = request.job_id.clone();
                        let tx = result_tx.clone();
                        let cpu_eng = cpu_engine.clone();
                        let gpu_eng = gpu_engine.clone();

                        tokio::spawn(async move {
                            let result = run_mining_job(
                                job_id,
                                header_hash,
                                difficulty,
                                nonce_start,
                                nonce_end,
                                cpu_eng,
                                gpu_eng,
                                cpu_workers,
                                gpu_devices,
                                cancel,
                            )
                            .await;
                            let _ = tx.send(result).await;
                        });
                    }
                    Ok(MinerMessage::JobResult(_)) => {
                        log::warn!("Received unexpected JobResult from node");
                    }
                    Ok(MinerMessage::Ready) => {
                        log::warn!("Received unexpected Ready from node");
                    }
                    Err(e) => {
                        if e.kind() == std::io::ErrorKind::UnexpectedEof {
                            return Err(anyhow::anyhow!("Node disconnected"));
                        }
                        return Err(anyhow::anyhow!("Read error: {}", e));
                    }
                }
            }

            // Handle mining results
            Some(result) = result_rx.recv() => {
                // Only send if this is for the current job
                if current_job_id.as_ref() == Some(&result.job_id) {
                    log::info!(
                        "⛏️ Sending result: job_id={}, status={:?}",
                        result.job_id,
                        result.status
                    );

                    let msg = MinerMessage::JobResult(result);
                    if let Err(e) = write_message(&mut send, &msg).await {
                        return Err(anyhow::anyhow!("Failed to send result: {}", e));
                    }

                    current_job_id = None;
                    current_cancel = None;
                } else {
                    log::debug!(
                        "Discarding stale result for job_id={}",
                        result.job_id
                    );
                }
            }
        }
    }
}

/// Run a mining job and return the result.
async fn run_mining_job(
    job_id: String,
    header_hash: [u8; 32],
    difficulty: U512,
    nonce_start: U512,
    nonce_end: U512,
    cpu_engine: Option<Arc<dyn MinerEngine>>,
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    cpu_workers: usize,
    gpu_devices: usize,
    cancel_flag: Arc<AtomicBool>,
) -> MiningResult {
    let start_time = Instant::now();
    let total_workers = cpu_workers + gpu_devices;

    // Spawn worker threads
    let (receiver, _handles) = spawn_mining_workers(
        header_hash,
        difficulty,
        nonce_start,
        nonce_end,
        cpu_engine,
        gpu_engine,
        cpu_workers,
        gpu_devices,
        cancel_flag.clone(),
    );

    // Wait for results
    let mut total_hashes = 0u64;
    let best_candidate: Option<MiningCandidate> = None;
    let mut completed_workers = 0usize;

    loop {
        // Check if cancelled externally
        if cancel_flag.load(Ordering::Relaxed) && best_candidate.is_none() {
            log::debug!("Job {} cancelled", job_id);
            return MiningResult {
                status: ApiResponseStatus::Cancelled,
                job_id,
                nonce: None,
                work: None,
                hash_count: total_hashes,
                elapsed_time: start_time.elapsed().as_secs_f64(),
            };
        }

        // Use spawn_blocking to avoid blocking the async runtime
        let recv_result = {
            let rx = receiver.clone();
            tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_millis(100)))
                .await
                .unwrap_or(Err(RecvTimeoutError::Disconnected))
        };

        match recv_result {
            Ok(worker_result) => {
                total_hashes += worker_result.hash_count;

                if let Some(candidate) = worker_result.candidate {
                    // Found a solution!
                    log::info!(
                        "⛏️ Job {} completed: {} hashes in {:.2}s",
                        job_id,
                        total_hashes,
                        start_time.elapsed().as_secs_f64()
                    );

                    // Signal other workers to stop
                    cancel_flag.store(true, Ordering::Relaxed);

                    return MiningResult {
                        status: ApiResponseStatus::Completed,
                        job_id,
                        nonce: Some(format!("{:x}", candidate.nonce)),
                        work: Some(hex::encode(candidate.work)),
                        hash_count: total_hashes,
                        elapsed_time: start_time.elapsed().as_secs_f64(),
                    };
                }

                if worker_result.completed {
                    completed_workers += 1;
                    if completed_workers >= total_workers {
                        // All workers done, no solution found
                        log::warn!("Job {} failed: no solution found", job_id);
                        return MiningResult {
                            status: ApiResponseStatus::Failed,
                            job_id,
                            nonce: None,
                            work: None,
                            hash_count: total_hashes,
                            elapsed_time: start_time.elapsed().as_secs_f64(),
                        };
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                // Continue waiting
            }
            Err(RecvTimeoutError::Disconnected) => {
                // Channel closed, all workers done
                if best_candidate.is_some() {
                    break;
                }
                log::warn!("Job {} failed: workers disconnected", job_id);
                return MiningResult {
                    status: ApiResponseStatus::Failed,
                    job_id,
                    nonce: None,
                    work: None,
                    hash_count: total_hashes,
                    elapsed_time: start_time.elapsed().as_secs_f64(),
                };
            }
        }
    }

    // Should not reach here, but just in case
    MiningResult {
        status: ApiResponseStatus::Failed,
        job_id,
        nonce: None,
        work: None,
        hash_count: total_hashes,
        elapsed_time: start_time.elapsed().as_secs_f64(),
    }
}

/// Generate a random 512-bit nonce starting point.
fn generate_random_nonce_start() -> U512 {
    let mut bytes = [0u8; 64];
    getrandom::getrandom(&mut bytes).expect("Failed to generate random bytes");
    U512::from_big_endian(&bytes)
}

/// Certificate verifier that accepts any certificate (for self-signed certs).
struct InsecureCertVerifier;

impl rustls::client::ServerCertVerifier for InsecureCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }
}
