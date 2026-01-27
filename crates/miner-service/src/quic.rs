//! QUIC client for connecting to blockchain nodes.
//!
//! This module provides a QUIC client that connects to a blockchain node
//! and handles bidirectional streaming for receiving mining jobs and
//! sending results.
//!
//! # Architecture
//!
//! The miner connects to the node (not the other way around). This allows:
//! - Instant reconnection when the miner restarts
//! - Multiple miners connecting to a single node
//! - Each miner independently selects random nonce starting points
//!
//! # Protocol
//!
//! - Node sends `MinerMessage::NewJob` to submit a mining job
//! - Miner sends `MinerMessage::JobResult` when mining completes
//! - When a new job arrives, any previous job is implicitly cancelled
//!
//! # Wire Format
//!
//! Messages are length-prefixed JSON: 4-byte big-endian length followed by JSON payload.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use primitive_types::U512;
use quinn::{ClientConfig, Endpoint};
use rustls::client::ServerCertVerified;
use tokio::sync::mpsc;

use quantus_miner_api::{read_message, write_message, ApiResponseStatus, MinerMessage, MiningResult};

use crate::MiningService;

/// Connect to a node and start mining.
///
/// This function connects to the node, receives mining jobs, and sends results.
/// It automatically reconnects if the connection is lost.
///
/// This function runs forever (or until an unrecoverable error occurs).
pub async fn connect_and_mine(
    node_addr: SocketAddr,
    mining_service: MiningService,
) -> anyhow::Result<()> {
    let mut reconnect_delay = Duration::from_secs(1);
    const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);

    loop {
        log::info!("⛏️ Connecting to node at {}...", node_addr);

        match establish_connection(node_addr).await {
            Ok((connection, send, recv)) => {
                log::info!("⛏️ Connected to node at {}", node_addr);
                reconnect_delay = Duration::from_secs(1); // Reset delay on success

                // Handle the connection until it fails
                if let Err(e) = handle_connection(connection, send, recv, mining_service.clone()).await {
                    log::info!("⛏️ Connection lost: {}", e);
                }
            }
            Err(e) => {
                log::warn!("⛏️ Failed to connect to node: {}", e);
            }
        }

        log::info!("⛏️ Reconnecting in {:?}...", reconnect_delay);
        tokio::time::sleep(reconnect_delay).await;

        // Exponential backoff
        reconnect_delay = (reconnect_delay * 2).min(MAX_RECONNECT_DELAY);
    }
}

/// Establish a QUIC connection to the node.
async fn establish_connection(
    addr: SocketAddr,
) -> anyhow::Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream)> {
    // Create client config with insecure certificate verification
    // (node uses self-signed certificate)
    let mut crypto = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_custom_certificate_verifier(Arc::new(InsecureCertVerifier))
        .with_no_client_auth();

    // Set ALPN protocol to match the node server
    crypto.alpn_protocols = vec![b"quantus-miner".to_vec()];

    let mut client_config = ClientConfig::new(Arc::new(crypto));

    // Set transport config with keep-alive to detect connection loss
    let mut transport_config = quinn::TransportConfig::default();
    transport_config.keep_alive_interval(Some(Duration::from_secs(10)));
    transport_config.max_idle_timeout(Some(Duration::from_secs(60).try_into().unwrap()));
    client_config.transport_config(Arc::new(transport_config));

    // Create endpoint
    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
    endpoint.set_default_client_config(client_config);

    // Connect to the node
    let connection = endpoint.connect(addr, "localhost")?.await?;
    log::info!("⛏️ QUIC connection established to {}", addr);

    // Open a bidirectional stream
    log::info!("⛏️ Opening bidirectional stream to node...");
    let (mut send, recv) = connection.open_bi().await?;
    
    // QUIC streams are lazily established - we need to send data to actually
    // create the stream on the server side. Send a Ready message to trigger
    // the stream creation and let the node know we're connected.
    log::debug!("Sending Ready message to establish stream...");
    write_message(&mut send, &MinerMessage::Ready).await?;
    log::info!("⛏️ Bidirectional stream established");

    Ok((connection, send, recv))
}

/// Handle an established connection, receiving jobs and sending results.
async fn handle_connection(
    _connection: quinn::Connection,
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    mining_service: MiningService,
) -> anyhow::Result<()> {
    // Channel for receiving mining results from the service
    let (result_tx, mut result_rx) = mpsc::channel::<MiningResult>(16);

    // Current job ID being worked on (for stale result detection)
    let mut current_job_id: Option<String> = None;

    log::info!("⛏️ Waiting for mining jobs from node...");

    loop {
        tokio::select! {
            // Prioritize reading to detect disconnection faster
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
                    }
                    Ok(MinerMessage::JobResult(_)) => {
                        // Node should not send JobResult to miner
                        log::warn!("Received unexpected JobResult from node, ignoring");
                    }
                    Ok(MinerMessage::Ready) => {
                        // Node should not send Ready to miner
                        log::warn!("Received unexpected Ready from node, ignoring");
                    }
                    Err(e) => {
                        if e.kind() == std::io::ErrorKind::UnexpectedEof {
                            return Err(anyhow::anyhow!("Node disconnected"));
                        }
                        return Err(anyhow::anyhow!("Read error: {}", e));
                    }
                }
            }

            // Handle mining results to send back to the node
            Some(result) = result_rx.recv() => {
                // Check if this result is for the current job
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

/// Generate a random 512-bit nonce starting point.
fn generate_random_nonce_start() -> U512 {
    let mut bytes = [0u8; 64];
    getrandom::getrandom(&mut bytes).expect("Failed to generate random bytes");
    U512::from_big_endian(&bytes)
}

/// Process a mining job and send the result via the channel.
async fn process_mining_job(
    request: quantus_miner_api::MiningRequest,
    mining_service: MiningService,
    result_tx: mpsc::Sender<MiningResult>,
) {
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

    // Generate random nonce starting point
    let nonce_start = generate_random_nonce_start();
    let nonce_end = U512::MAX;

    log::debug!(
        "Starting job {} with random nonce start: {:x}...",
        request.job_id,
        nonce_start
    );

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
                    "⛏️ Job {} completed: {} hashes in {:.2}s",
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

/// A certificate verifier that accepts any certificate.
///
/// This is used because the node uses a self-signed certificate.
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
        // Accept any certificate
        Ok(ServerCertVerified::assertion())
    }
}
