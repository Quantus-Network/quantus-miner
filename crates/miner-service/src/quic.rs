//! QUIC transport layer for the miner service.
//!
//! This module provides a QUIC server that accepts connections from blockchain nodes
//! and handles bidirectional streaming for mining job submission and result delivery.
//!
//! # Protocol
//!
//! The protocol is simple:
//! - Node sends `MinerMessage::NewJob` to submit a mining job (implicitly cancels any previous)
//! - Miner sends `MinerMessage::JobResult` when mining completes
//!
//! # Wire Format
//!
//! Messages are length-prefixed JSON: 4-byte big-endian length followed by JSON payload.

use std::net::SocketAddr;
use std::sync::Arc;

use quinn::{Endpoint, ServerConfig};
use rustls::{Certificate, PrivateKey};
use tokio::sync::mpsc;

use quantus_miner_api::{read_message, write_message, MinerMessage, MiningResult};

use crate::MiningService;

/// QUIC server configuration and state.
pub struct QuicServer {
    endpoint: Endpoint,
    mining_service: MiningService,
}

impl QuicServer {
    /// Create a new QUIC server bound to the given address.
    ///
    /// Generates a self-signed certificate for TLS.
    pub fn new(addr: SocketAddr, mining_service: MiningService) -> anyhow::Result<Self> {
        let server_config = generate_self_signed_config()?;
        let endpoint = Endpoint::server(server_config, addr)?;

        log::info!("QUIC server listening on {}", addr);

        Ok(Self {
            endpoint,
            mining_service,
        })
    }

    /// Run the server, accepting connections and handling them.
    ///
    /// This function runs forever (or until the endpoint is closed).
    pub async fn run(self) -> anyhow::Result<()> {
        log::info!("QUIC server ready to accept connections");

        while let Some(incoming) = self.endpoint.accept().await {
            let mining_service = self.mining_service.clone();

            tokio::spawn(async move {
                match incoming.await {
                    Ok(connection) => {
                        let remote = connection.remote_address();
                        log::info!("New QUIC connection from {}", remote);

                        if let Err(e) = handle_connection(connection, mining_service).await {
                            log::warn!("Connection from {} closed with error: {}", remote, e);
                        } else {
                            log::info!("Connection from {} closed gracefully", remote);
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to accept connection: {}", e);
                    }
                }
            });
        }

        Ok(())
    }
}

/// Handle a single QUIC connection from a node.
///
/// Opens a bidirectional stream for the connection lifetime:
/// - Receives `NewJob` messages from the node
/// - Sends `JobResult` messages back when mining completes
async fn handle_connection(
    connection: quinn::Connection,
    mining_service: MiningService,
) -> anyhow::Result<()> {
    // Accept a bidirectional stream from the client
    let (send, recv) = connection.accept_bi().await?;

    handle_stream(send, recv, mining_service, connection.remote_address()).await
}

/// Handle a bidirectional stream for mining communication.
async fn handle_stream(
    send: quinn::SendStream,
    recv: quinn::RecvStream,
    mining_service: MiningService,
    remote_addr: SocketAddr,
) -> anyhow::Result<()> {
    // Wrap streams for async read/write
    let mut send = send;
    let mut recv = recv;

    // Channel for receiving mining results from the service
    let (result_tx, mut result_rx) = mpsc::channel::<MiningResult>(16);

    // Current job ID being worked on (for stale result detection)
    let mut current_job_id: Option<String> = None;

    loop {
        tokio::select! {
            // Handle incoming messages from the node
            msg_result = read_message(&mut recv) => {
                match msg_result {
                    Ok(MinerMessage::NewJob(request)) => {
                        log::info!(
                            "Received NewJob from {}: job_id={}, hash={}",
                            remote_addr,
                            request.job_id,
                            &request.mining_hash[..8]
                        );

                        // Cancel any existing job before starting the new one
                        if let Some(old_job_id) = current_job_id.take() {
                            log::debug!("Cancelling previous job: {}", old_job_id);
                            mining_service.cancel_job(&old_job_id).await;
                        }

                        // Start the new job
                        current_job_id = Some(request.job_id.clone());

                        let job_id = request.job_id.clone();
                        let result_tx = result_tx.clone();
                        let service = mining_service.clone();

                        // Spawn job processing
                        tokio::spawn(async move {
                            process_mining_job(request, service, result_tx).await;
                        });

                        log::debug!("Started mining job: {}", job_id);
                    }
                    Ok(MinerMessage::JobResult(_)) => {
                        // Node should not send JobResult to miner
                        log::warn!("Received unexpected JobResult from node, ignoring");
                    }
                    Err(e) => {
                        if e.kind() == std::io::ErrorKind::UnexpectedEof {
                            log::info!("Node disconnected");
                        } else {
                            log::warn!("Error reading message: {}", e);
                        }
                        // Cancel any running job
                        if let Some(job_id) = current_job_id.take() {
                            mining_service.cancel_job(&job_id).await;
                        }
                        return Ok(());
                    }
                }
            }

            // Handle mining results to send back to the node
            Some(result) = result_rx.recv() => {
                // Check if this result is for the current job
                if current_job_id.as_ref() == Some(&result.job_id) {
                    log::info!(
                        "Sending JobResult to {}: job_id={}, status={:?}",
                        remote_addr,
                        result.job_id,
                        result.status
                    );

                    let msg = MinerMessage::JobResult(result);
                    if let Err(e) = write_message(&mut send, &msg).await {
                        log::warn!("Failed to send result: {}", e);
                        return Err(e.into());
                    }

                    // Clear current job since it's complete
                    current_job_id = None;
                } else {
                    log::debug!(
                        "Discarding stale result for job_id={} (current={:?})",
                        result.job_id,
                        current_job_id
                    );
                }
            }
        }
    }
}

/// Process a mining job and send the result via the channel.
async fn process_mining_job(
    request: quantus_miner_api::MiningRequest,
    mining_service: MiningService,
    result_tx: mpsc::Sender<MiningResult>,
) {
    use primitive_types::U512;
    use quantus_miner_api::ApiResponseStatus;

    // Parse and validate the request
    let header_hash: [u8; 32] = match hex::decode(&request.mining_hash) {
        Ok(bytes) if bytes.len() == 32 => bytes.try_into().unwrap(),
        _ => {
            log::warn!("Invalid mining_hash in request: {}", request.job_id);
            let _ = result_tx
                .send(MiningResult {
                    status: ApiResponseStatus::Failed,
                    job_id: request.job_id,
                    nonce: None,
                    work: None,
                    hash_count: 0,
                    elapsed_time: 0.0,
                })
                .await;
            return;
        }
    };

    let difficulty = match U512::from_dec_str(&request.distance_threshold) {
        Ok(d) => d,
        Err(_) => {
            log::warn!("Invalid difficulty in request: {}", request.job_id);
            let _ = result_tx
                .send(MiningResult {
                    status: ApiResponseStatus::Failed,
                    job_id: request.job_id,
                    nonce: None,
                    work: None,
                    hash_count: 0,
                    elapsed_time: 0.0,
                })
                .await;
            return;
        }
    };

    let nonce_start = match U512::from_str_radix(&request.nonce_start, 16) {
        Ok(n) => n,
        Err(_) => {
            log::warn!("Invalid nonce_start in request: {}", request.job_id);
            let _ = result_tx
                .send(MiningResult {
                    status: ApiResponseStatus::Failed,
                    job_id: request.job_id,
                    nonce: None,
                    work: None,
                    hash_count: 0,
                    elapsed_time: 0.0,
                })
                .await;
            return;
        }
    };

    let nonce_end = match U512::from_str_radix(&request.nonce_end, 16) {
        Ok(n) => n,
        Err(_) => {
            log::warn!("Invalid nonce_end in request: {}", request.job_id);
            let _ = result_tx
                .send(MiningResult {
                    status: ApiResponseStatus::Failed,
                    job_id: request.job_id,
                    nonce: None,
                    work: None,
                    hash_count: 0,
                    elapsed_time: 0.0,
                })
                .await;
            return;
        }
    };

    // Create and add the mining job
    let job = crate::MiningJob::new(header_hash, difficulty, nonce_start, nonce_end);
    let job_id = request.job_id.clone();

    if let Err(e) = mining_service.add_job(job_id.clone(), job).await {
        log::warn!("Failed to add job {}: {}", job_id, e);
        let _ = result_tx
            .send(MiningResult {
                status: ApiResponseStatus::Failed,
                job_id,
                nonce: None,
                work: None,
                hash_count: 0,
                elapsed_time: 0.0,
            })
            .await;
        return;
    }

    // Poll for job completion
    loop {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let job = match mining_service.get_job(&job_id).await {
            Some(j) => j,
            None => {
                // Job was removed (cancelled)
                log::debug!("Job {} was cancelled/removed", job_id);
                let _ = result_tx
                    .send(MiningResult {
                        status: ApiResponseStatus::Cancelled,
                        job_id,
                        nonce: None,
                        work: None,
                        hash_count: 0,
                        elapsed_time: 0.0,
                    })
                    .await;
                return;
            }
        };

        match job.status {
            crate::JobStatus::Running => {
                // Still working, continue polling
                continue;
            }
            crate::JobStatus::Completed => {
                let elapsed_time = job.start_time.elapsed().as_secs_f64();
                let (nonce_hex, work_hex) = match &job.best_result {
                    Some(result) => (
                        Some(format!("{:x}", result.nonce)),
                        Some(hex::encode(result.work)),
                    ),
                    None => (None, None),
                };

                log::info!(
                    "Job {} completed: {} hashes in {:.2}s",
                    job_id,
                    job.total_hash_count,
                    elapsed_time
                );

                let _ = result_tx
                    .send(MiningResult {
                        status: ApiResponseStatus::Completed,
                        job_id,
                        nonce: nonce_hex,
                        work: work_hex,
                        hash_count: job.total_hash_count,
                        elapsed_time,
                    })
                    .await;
                return;
            }
            crate::JobStatus::Failed => {
                let elapsed_time = job.start_time.elapsed().as_secs_f64();
                log::warn!("Job {} failed", job_id);
                let _ = result_tx
                    .send(MiningResult {
                        status: ApiResponseStatus::Failed,
                        job_id,
                        nonce: None,
                        work: None,
                        hash_count: job.total_hash_count,
                        elapsed_time,
                    })
                    .await;
                return;
            }
            crate::JobStatus::Cancelled => {
                let elapsed_time = job.start_time.elapsed().as_secs_f64();
                log::debug!("Job {} was cancelled", job_id);
                let _ = result_tx
                    .send(MiningResult {
                        status: ApiResponseStatus::Cancelled,
                        job_id,
                        nonce: None,
                        work: None,
                        hash_count: job.total_hash_count,
                        elapsed_time,
                    })
                    .await;
                return;
            }
        }
    }
}

/// Generate a self-signed TLS certificate and create a QUIC server config.
fn generate_self_signed_config() -> anyhow::Result<ServerConfig> {
    // Generate a self-signed certificate
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])?;
    let cert_der = cert.serialize_der()?;
    let key_der = cert.serialize_private_key_der();

    let cert_chain = vec![Certificate(cert_der)];
    let key = PrivateKey(key_der);

    // Build rustls config
    let mut server_crypto = rustls::ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)?;

    server_crypto.alpn_protocols = vec![b"quantus-miner".to_vec()];

    let mut server_config = ServerConfig::with_crypto(Arc::new(server_crypto));

    // Configure transport to allow longer idle periods between mining jobs
    // The node may take some time to prepare the next block after one is mined
    let mut transport_config = quinn::TransportConfig::default();
    transport_config.max_idle_timeout(Some(
        std::time::Duration::from_secs(60).try_into().unwrap(),
    ));
    server_config.transport_config(Arc::new(transport_config));

    Ok(server_config)
}
