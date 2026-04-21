//! QUIC client for connecting to the Quantus node.
//!
//! This module handles the connection to the node, receiving mining jobs,
//! and submitting solutions.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quantus_miner_api::{
    read_message, write_message, ApiResponseStatus, MinerMessage, MiningRequest, MiningResult,
};
use quinn::{ClientConfig, Endpoint};
use rustls::client::ServerCertVerified;
use tokio::sync::mpsc;

/// Events from the node.
#[derive(Debug)]
pub enum NodeEvent {
    /// A new mining job from the node.
    NewJob(MiningRequest),
    /// Connection was lost.
    Disconnected,
}

/// Client for communicating with the Quantus node.
pub struct NodeClient {
    node_addr: SocketAddr,
}

impl NodeClient {
    /// Create a new node client.
    pub fn new(node_addr: SocketAddr) -> Self {
        Self { node_addr }
    }

    /// Connect to the node and run the message loop.
    ///
    /// - `job_tx`: Channel to send new jobs to the coordinator.
    /// - `result_rx`: Channel to receive results to submit to the node.
    pub async fn run(
        &self,
        job_tx: mpsc::Sender<NodeEvent>,
        mut result_rx: mpsc::Receiver<MiningResult>,
    ) -> anyhow::Result<()> {
        let mut reconnect_delay = Duration::from_secs(1);
        const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);

        loop {
            log::info!("Connecting to node at {}...", self.node_addr);

            match self.establish_connection().await {
                Ok((connection, mut send, mut recv)) => {
                    log::info!("Connected to node at {}", self.node_addr);
                    reconnect_delay = Duration::from_secs(1);

                    // Run the message loop
                    loop {
                        tokio::select! {
                            biased;

                            // Check for connection close
                            reason = connection.closed() => {
                                log::warn!("Connection closed: {reason}");
                                let _ = job_tx.send(NodeEvent::Disconnected).await;
                                break;
                            }

                            // Receive messages from node
                            msg_result = read_message(&mut recv) => {
                                match msg_result {
                                    Ok(MinerMessage::NewJob(request)) => {
                                        log::info!(
                                            "Received job from node: id={}, hash={}...",
                                            request.job_id,
                                            &request.mining_hash[..8.min(request.mining_hash.len())]
                                        );
                                        if job_tx.send(NodeEvent::NewJob(request)).await.is_err() {
                                            log::error!("Job channel closed");
                                            return Ok(());
                                        }
                                    }
                                    Ok(other) => {
                                        log::debug!("Unexpected message from node: {other:?}");
                                    }
                                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                                        log::warn!("Node disconnected");
                                        let _ = job_tx.send(NodeEvent::Disconnected).await;
                                        break;
                                    }
                                    Err(e) => {
                                        log::error!("Error reading from node: {e}");
                                        break;
                                    }
                                }
                            }

                            // Send results to node
                            result = result_rx.recv() => {
                                match result {
                                    Some(result) => {
                                        log::info!(
                                            "Submitting result to node: job={}, status={:?}",
                                            result.job_id,
                                            result.status
                                        );
                                        let msg = MinerMessage::JobResult(result);
                                        if let Err(e) = write_message(&mut send, &msg).await {
                                            log::error!("Failed to send result to node: {e}");
                                            break;
                                        }
                                    }
                                    None => {
                                        log::info!("Result channel closed, shutting down");
                                        return Ok(());
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to connect to node: {e}");
                }
            }

            log::info!("Reconnecting in {reconnect_delay:?}...");
            tokio::time::sleep(reconnect_delay).await;
            reconnect_delay = (reconnect_delay * 2).min(MAX_RECONNECT_DELAY);
        }
    }

    /// Establish a QUIC connection to the node.
    async fn establish_connection(
        &self,
    ) -> anyhow::Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream)> {
        let mut crypto = rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_custom_certificate_verifier(Arc::new(InsecureCertVerifier))
            .with_no_client_auth();

        crypto.alpn_protocols = vec![b"quantus-miner".to_vec()];

        let mut client_config = ClientConfig::new(Arc::new(crypto));

        let mut transport_config = quinn::TransportConfig::default();
        transport_config.keep_alive_interval(Some(Duration::from_secs(5)));
        transport_config.max_idle_timeout(Some(Duration::from_secs(15).try_into().unwrap()));
        client_config.transport_config(Arc::new(transport_config));

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
        endpoint.set_default_client_config(client_config);

        let connection = endpoint.connect(self.node_addr, "localhost")?.await?;
        log::debug!("QUIC connection established to {}", self.node_addr);

        let (mut send, recv) = connection.open_bi().await?;

        // Send Ready message to establish the stream
        write_message(&mut send, &MinerMessage::Ready).await?;
        log::debug!("Sent Ready message to node");

        Ok((connection, send, recv))
    }
}

/// Create a MiningResult for submitting to the node.
pub fn create_mining_result(
    job_id: String,
    nonce: String,
    work: String,
    hash_count: u64,
    elapsed_time: f64,
) -> MiningResult {
    MiningResult {
        status: ApiResponseStatus::Completed,
        job_id,
        nonce: Some(nonce),
        work: Some(work),
        hash_count,
        elapsed_time,
        miner_id: None,
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
