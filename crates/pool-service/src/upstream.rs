//! Job sources: either a QUIC connection to a real quantus-node (speaking the
//! external miner protocol), or a standalone generator for demos/tests.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use primitive_types::U512;
use quantus_miner_api::{
    read_message, write_message, ApiResponseStatus, MinerMessage, MiningResult,
};
use quinn::{ClientConfig, Endpoint};
use rustls::client::ServerCertVerified;
use tokio::sync::mpsc::Receiver;

use crate::state::{FoundBlock, Job, PoolState};

/// Connect to a node as an external miner and keep the pool's job current.
/// Solutions arriving on `solutions` are pushed upstream as JobResults.
pub async fn run_node_client(
    state: Arc<PoolState>,
    node_addr: SocketAddr,
    mut solutions: Receiver<FoundBlock>,
) {
    let mut reconnect_delay = Duration::from_secs(1);
    const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);

    loop {
        log::info!("Connecting to node at {}...", node_addr);
        match establish_connection(node_addr).await {
            Ok((connection, mut send, mut recv)) => {
                log::info!("Connected to node at {}", node_addr);
                reconnect_delay = Duration::from_secs(1);

                loop {
                    tokio::select! {
                        biased;

                        reason = connection.closed() => {
                            log::warn!("Node connection closed: {}", reason);
                            break;
                        }

                        // A captcha share met network difficulty: submit it.
                        block = solutions.recv() => {
                            if let Some(block) = block {
                                let result = MiningResult {
                                    status: ApiResponseStatus::Completed,
                                    job_id: block.job_id,
                                    nonce: Some(format!("{:x}", block.nonce)),
                                    work: Some(hex::encode(block.nonce.to_big_endian())),
                                    hash_count: 0,
                                    elapsed_time: 0.0,
                                    miner_id: None,
                                };
                                if let Err(e) = write_message(&mut send, &MinerMessage::JobResult(result)).await {
                                    log::error!("Failed to submit block solution: {}", e);
                                    break;
                                }
                                log::info!("Submitted block solution to node");
                            }
                        }

                        msg = read_message(&mut recv) => {
                            match msg {
                                Ok(MinerMessage::NewJob(request)) => {
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
            }
            Err(e) => {
                log::warn!("Failed to connect to node: {}", e);
            }
        }

        log::info!("Reconnecting in {:?}...", reconnect_delay);
        tokio::time::sleep(reconnect_delay).await;
        reconnect_delay = (reconnect_delay * 2).min(MAX_RECONNECT_DELAY);
    }
}

fn parse_job(job_id: &str, mining_hash: &str, difficulty: &str) -> anyhow::Result<Job> {
    let bytes = hex::decode(mining_hash)?;
    let header: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("mining_hash must be 32 bytes"))?;
    let network_difficulty = U512::from_dec_str(difficulty)
        .map_err(|e| anyhow::anyhow!("bad difficulty: {:?}", e))?;
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
    transport_config.keep_alive_interval(Some(Duration::from_secs(5)));
    transport_config.max_idle_timeout(Some(Duration::from_secs(15).try_into().unwrap()));
    client_config.transport_config(Arc::new(transport_config));

    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
    endpoint.set_default_client_config(client_config);

    let connection = endpoint.connect(addr, "localhost")?.await?;
    let (mut send, recv) = connection.open_bi().await?;
    write_message(&mut send, &MinerMessage::Ready).await?;
    Ok((connection, send, recv))
}

/// Accepts any certificate (nodes use self-signed certs, same as miner-service).
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
