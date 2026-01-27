//! QUIC client for connecting to blockchain nodes.
//!
//! This module provides a QUIC client that connects to a blockchain node
//! and handles bidirectional streaming for receiving mining jobs and
//! sending results.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::RecvTimeoutError;
use engine_cpu::MinerEngine;
use primitive_types::U512;
use quinn::{ClientConfig, Endpoint};
use rustls::client::ServerCertVerified;

use quantus_miner_api::{
    read_message, write_message, ApiResponseStatus, MinerMessage, MiningResult,
};

use crate::{EngineType, WorkerPool};

/// Connect to a node and start mining.
///
/// This function connects to the node, receives mining jobs, and sends results.
/// It automatically reconnects if the connection is lost.
///
/// Uses a persistent worker pool to avoid thread creation overhead between jobs.
pub async fn connect_and_mine(
    node_addr: SocketAddr,
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

        match establish_connection(node_addr).await {
            Ok((connection, send, recv)) => {
                log::info!("⛏️ Connected to node at {}", node_addr);
                reconnect_delay = Duration::from_secs(1);

                if let Err(e) = handle_connection(connection, send, recv, &worker_pool).await {
                    log::info!("⛏️ Connection lost: {}", e);
                    // Cancel any running job when connection drops
                    worker_pool.cancel();
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
    worker_pool: &WorkerPool,
) -> anyhow::Result<()> {
    // Set static metrics once per connection
    metrics::set_effective_cpus(num_cpus::get() as i64);
    metrics::set_workers(worker_pool.worker_count() as i64);
    metrics::set_cpu_workers(worker_pool.cpu_worker_count() as i64);
    metrics::set_gpu_devices(worker_pool.gpu_worker_count() as i64);

    // Reset hash rate tracker for accurate rate calculation
    metrics::reset_hash_tracker();

    // Current job state
    let mut current_job_id: Option<String> = None;
    let mut job_start_time: Option<Instant> = None;
    let mut cpu_hashes: u64 = 0;
    let mut gpu_hashes: u64 = 0;

    log::info!("⛏️ Waiting for mining jobs from node...");

    loop {
        // Poll for worker results (non-blocking via spawn_blocking)
        let poll_result = if current_job_id.is_some() {
            let rx = worker_pool.result_receiver().clone();
            tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_millis(10)))
                .await
                .unwrap_or(Err(RecvTimeoutError::Disconnected))
        } else {
            Err(RecvTimeoutError::Timeout) // No active job, just skip
        };

        // Handle worker result if any
        if let Ok(worker_result) = poll_result {
            // Track hashes by engine type
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

            // Check if solution found
            if let Some(candidate) = worker_result.candidate {
                if let Some(job_id) = current_job_id.take() {
                    let total_hashes = cpu_hashes + gpu_hashes;
                    let elapsed = job_start_time
                        .map(|t| t.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    let hash_rate = if elapsed > 0.0 {
                        total_hashes as f64 / elapsed
                    } else {
                        0.0
                    };

                    log::info!(
                        "⛏️ Job {} completed: {} hashes in {:.2}s ({:.2} MH/s)",
                        job_id,
                        total_hashes,
                        elapsed,
                        hash_rate / 1_000_000.0
                    );

                    // Signal other workers to stop
                    worker_pool.cancel();
                    metrics::set_active_jobs(0);

                    let result = MiningResult {
                        status: ApiResponseStatus::Completed,
                        job_id,
                        nonce: Some(format!("{:x}", candidate.nonce)),
                        work: Some(hex::encode(candidate.work)),
                        hash_count: total_hashes,
                        elapsed_time: elapsed,
                    };

                    let msg = MinerMessage::JobResult(result);
                    if let Err(e) = write_message(&mut send, &msg).await {
                        return Err(anyhow::anyhow!("Failed to send result: {}", e));
                    }

                    // Reset for next job
                    cpu_hashes = 0;
                    gpu_hashes = 0;
                    job_start_time = None;
                }
            }
        }

        // Check for incoming messages (with short timeout to allow result polling)
        tokio::select! {
            biased;

            msg_result = read_message(&mut recv) => {
                match msg_result {
                    Ok(MinerMessage::NewJob(request)) => {
                        log::info!(
                            "⛏️ Received job: id={}, hash={}...",
                            request.job_id,
                            &request.mining_hash[..8]
                        );

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
                                let msg = MinerMessage::JobResult(result);
                                write_message(&mut send, &msg).await?;
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
                                let msg = MinerMessage::JobResult(result);
                                write_message(&mut send, &msg).await?;
                                continue;
                            }
                        };

                        // Reset hash counters for new job
                        cpu_hashes = 0;
                        gpu_hashes = 0;
                        job_start_time = Some(Instant::now());
                        current_job_id = Some(request.job_id.clone());

                        log::debug!("Starting job {}", request.job_id);
                        metrics::set_active_jobs(1);

                        // Start job via persistent worker pool
                        // This cancels any previous job and dispatches to all workers
                        worker_pool.start_job(header_hash, difficulty);
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

            // Short sleep to yield to other tasks when no messages
            _ = tokio::time::sleep(Duration::from_millis(1)) => {}
        }
    }
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
