//! QUIC client for connecting to blockchain nodes.
//!
//! This module provides a QUIC client that connects to a blockchain node
//! and handles bidirectional streaming for receiving mining jobs and
//! sending results.
//!
//! # Certificate Verification
//!
//! The client supports two certificate verification modes:
//! - **Pinned**: Verify against a specific certificate fingerprint (recommended for remote connections)
//! - **Insecure**: Skip verification (suitable for localhost or trusted local network)
//!
//! The node prints its certificate fingerprint on startup in the format `sha256:<hex>`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use engine_cpu::MinerEngine;
use primitive_types::U512;
use quinn::{ClientConfig, Endpoint};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use sha2::{Digest, Sha256};

use quantus_miner_api::{
    read_message, write_message, ApiResponseStatus, MinerMessage, MiningResult,
};

use crate::{CertVerification, EngineType, WorkerPool};
use pow_core::format_hashrate;

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
    cert_verification: CertVerification,
) -> anyhow::Result<()> {
    // Create persistent worker pool once - it lives for the entire miner lifetime
    let worker_pool = WorkerPool::new(cpu_engine, gpu_engine, cpu_workers, gpu_devices);

    let mut reconnect_delay = Duration::from_secs(1);
    const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);

    loop {
        log::info!("⛏️ Connecting to node at {}...", node_addr);

        match establish_connection(node_addr, &cert_verification).await {
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
    cert_verification: &CertVerification,
) -> anyhow::Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream)> {
    // Create certificate verifier based on configuration
    let verifier: Arc<dyn ServerCertVerifier> = match cert_verification {
        CertVerification::Pinned(fingerprint) => {
            Arc::new(PinnedCertVerifier::new(fingerprint.clone())?)
        }
        CertVerification::Insecure => Arc::new(InsecureCertVerifier),
    };

    // Use post-quantum crypto provider
    let crypto = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls_post_quantum::provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| anyhow::anyhow!("Failed to set protocol versions: {e}"))?
    .dangerous()
    .with_custom_certificate_verifier(verifier)
    .with_no_client_auth();

    let mut client_config = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .map_err(|e| anyhow::anyhow!("Failed to create QUIC client config: {e}"))?,
    ));

    let mut transport_config = quinn::TransportConfig::default();
    transport_config.keep_alive_interval(Some(Duration::from_secs(5)));
    transport_config.max_idle_timeout(Some(Duration::from_secs(15).try_into().unwrap()));
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
async fn handle_connection(
    connection: quinn::Connection,
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    worker_pool: &WorkerPool,
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
                return Err(anyhow::anyhow!("Connection closed: {}", reason));
            }

            msg_result = read_message(&mut recv) => {
                match msg_result {
                    Ok(MinerMessage::NewJob(request)) => {
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
                    Ok(MinerMessage::Ready) => {
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

// ---------------------------------------------------------------------------
// Certificate Verifiers
// ---------------------------------------------------------------------------

/// Certificate verifier that pins to a specific certificate fingerprint.
///
/// The fingerprint is SHA-256 of the certificate DER, formatted as `sha256:<hex>`.
#[derive(Debug)]
struct PinnedCertVerifier {
    /// Expected fingerprint in lowercase hex (without the `sha256:` prefix)
    expected_fingerprint: String,
}

impl PinnedCertVerifier {
    fn new(fingerprint: String) -> anyhow::Result<Self> {
        // Parse and validate fingerprint format
        let fp = fingerprint
            .strip_prefix("sha256:")
            .ok_or_else(|| anyhow::anyhow!("Fingerprint must start with 'sha256:'"))?;

        if fp.len() != 64 {
            anyhow::bail!("Fingerprint must be 64 hex characters (got {})", fp.len());
        }

        // Validate hex
        if !fp.chars().all(|c| c.is_ascii_hexdigit()) {
            anyhow::bail!("Fingerprint contains invalid hex characters");
        }

        Ok(Self {
            expected_fingerprint: fp.to_lowercase(),
        })
    }

    fn compute_fingerprint(cert_der: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(cert_der);
        let hash = hasher.finalize();
        hex::encode(hash)
    }
}

impl ServerCertVerifier for PinnedCertVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let actual_fingerprint = Self::compute_fingerprint(end_entity.as_ref());

        if actual_fingerprint == self.expected_fingerprint {
            log::debug!("Certificate fingerprint verified: sha256:{actual_fingerprint}");
            Ok(ServerCertVerified::assertion())
        } else {
            log::error!(
                "Certificate fingerprint mismatch!\n  Expected: sha256:{}\n  Got:      sha256:{}",
                self.expected_fingerprint,
                actual_fingerprint
            );
            Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::BadSignature,
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        // We trust the certificate based on fingerprint, so accept the signature
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        // We trust the certificate based on fingerprint, so accept the signature
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        // Only support ML-DSA (post-quantum) - IANA TLS SignatureScheme registry
        // https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml#tls-signaturescheme
        vec![
            SignatureScheme::Unknown(0x0904), // ML-DSA-44
            SignatureScheme::Unknown(0x0905), // ML-DSA-65
            SignatureScheme::Unknown(0x0906), // ML-DSA-87
        ]
    }
}


/// Certificate verifier that accepts any certificate.
///
/// Suitable for localhost connections or trusted local networks where MITM is not a concern.
/// For remote connections over untrusted networks, use `PinnedCertVerifier` instead.
#[derive(Debug)]
struct InsecureCertVerifier;

impl ServerCertVerifier for InsecureCertVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        // Log the fingerprint for convenience (user can copy it for pinning)
        let fingerprint = PinnedCertVerifier::compute_fingerprint(end_entity.as_ref());
        log::debug!("Server certificate fingerprint: sha256:{fingerprint}");
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        // Only support ML-DSA (post-quantum)
        vec![
            SignatureScheme::Unknown(0x0904), // ML-DSA-44
            SignatureScheme::Unknown(0x0905), // ML-DSA-65
            SignatureScheme::Unknown(0x0906), // ML-DSA-87
        ]
    }
}
