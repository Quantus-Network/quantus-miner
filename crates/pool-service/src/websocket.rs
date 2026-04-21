//! WebSocket server for browser-based miners.
//!
//! Accepts connections from browser miners over WS or WSS (secure WebSocket).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use pool_api::{MinerToPool, PoolToMiner};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, RwLock};
use tokio_rustls::TlsAcceptor;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Information about a connected browser miner.
#[derive(Debug, Clone)]
pub struct BrowserMiner {
    /// Unique ID for this miner session.
    pub id: u64,
    /// The miner's rewards address.
    pub address: String,
    /// Channel to send messages to this miner.
    pub tx: mpsc::Sender<PoolToMiner>,
}

/// Events from browser miners.
#[derive(Debug)]
pub enum BrowserEvent {
    /// A miner connected and registered.
    MinerConnected(BrowserMiner),
    /// A miner disconnected.
    MinerDisconnected(u64),
    /// A miner submitted a job result.
    JobResult {
        miner_id: u64,
        job_id: String,
        nonce: String,
        work: String,
        hash_count: u64,
        elapsed_time: f64,
    },
}

/// WebSocket server for browser miners.
pub struct WebSocketServer {
    /// Connected miners by ID.
    miners: Arc<RwLock<HashMap<u64, BrowserMiner>>>,
    /// Counter for assigning unique miner IDs.
    next_miner_id: AtomicU64,
    /// Channel to send events to the coordinator.
    event_tx: mpsc::Sender<BrowserEvent>,
}

impl WebSocketServer {
    /// Create a new WebSocket server.
    pub fn new(event_tx: mpsc::Sender<BrowserEvent>) -> Self {
        Self {
            miners: Arc::new(RwLock::new(HashMap::new())),
            next_miner_id: AtomicU64::new(1),
            event_tx,
        }
    }

    /// Start listening for connections.
    pub async fn run(&self, port: u16, no_tls: bool) -> anyhow::Result<()> {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
        let tls_acceptor = if no_tls {
            None
        } else {
            Some(create_tls_acceptor()?)
        };

        let protocol = if no_tls { "ws" } else { "wss" };
        log::info!("WebSocket server listening on {protocol}://0.0.0.0:{port}");

        loop {
            let (stream, addr) = listener.accept().await?;
            let tls_acceptor = tls_acceptor.clone();
            let miners = self.miners.clone();
            let event_tx = self.event_tx.clone();
            let miner_id = self.next_miner_id.fetch_add(1, Ordering::Relaxed);

            tokio::spawn(async move {
                let result = if let Some(tls_acceptor) = tls_acceptor {
                    handle_tls_connection(stream, addr, tls_acceptor, miners, event_tx, miner_id)
                        .await
                } else {
                    handle_plain_connection(stream, addr, miners, event_tx, miner_id).await
                };

                if let Err(e) = result {
                    log::debug!("Connection from {addr} ended: {e}");
                }
            });
        }
    }

    /// Broadcast a message to all connected miners.
    pub async fn broadcast(&self, msg: PoolToMiner) {
        let miners = self.miners.read().await;
        for (id, miner) in miners.iter() {
            if let Err(e) = miner.tx.try_send(msg.clone()) {
                log::debug!("Failed to send to miner {id}: {e}");
            }
        }
    }

    /// Send a message to a specific miner.
    pub async fn send_to(&self, miner_id: u64, msg: PoolToMiner) -> bool {
        let miners = self.miners.read().await;
        if let Some(miner) = miners.get(&miner_id) {
            miner.tx.try_send(msg).is_ok()
        } else {
            false
        }
    }

    /// Get the number of connected miners.
    pub async fn miner_count(&self) -> usize {
        self.miners.read().await.len()
    }

    /// Get information about a specific miner.
    pub async fn get_miner(&self, miner_id: u64) -> Option<BrowserMiner> {
        self.miners.read().await.get(&miner_id).cloned()
    }
}

/// Handle a plain (non-TLS) WebSocket connection.
async fn handle_plain_connection(
    stream: tokio::net::TcpStream,
    addr: SocketAddr,
    miners: Arc<RwLock<HashMap<u64, BrowserMiner>>>,
    event_tx: mpsc::Sender<BrowserEvent>,
    miner_id: u64,
) -> anyhow::Result<()> {
    log::debug!("New plain connection from {addr}");

    // WebSocket handshake
    let ws_stream = tokio_tungstenite::accept_async(stream).await?;
    let (mut write, mut read) = ws_stream.split();

    log::debug!("WebSocket established with {addr}");

    // Run the connection handler
    run_miner_session(&mut write, &mut read, addr, miners, event_tx, miner_id).await
}

/// Handle a TLS WebSocket connection.
async fn handle_tls_connection(
    stream: tokio::net::TcpStream,
    addr: SocketAddr,
    tls_acceptor: TlsAcceptor,
    miners: Arc<RwLock<HashMap<u64, BrowserMiner>>>,
    event_tx: mpsc::Sender<BrowserEvent>,
    miner_id: u64,
) -> anyhow::Result<()> {
    log::debug!("New TLS connection from {addr}");

    // TLS handshake
    let tls_stream = tls_acceptor.accept(stream).await?;

    // WebSocket handshake over TLS
    let ws_stream = tokio_tungstenite::accept_async(tls_stream).await?;
    let (mut write, mut read) = ws_stream.split();

    log::debug!("WebSocket (TLS) established with {addr}");

    // Run the connection handler
    run_miner_session(&mut write, &mut read, addr, miners, event_tx, miner_id).await
}

/// Run the miner session protocol.
async fn run_miner_session<S, R>(
    write: &mut S,
    read: &mut R,
    addr: SocketAddr,
    miners: Arc<RwLock<HashMap<u64, BrowserMiner>>>,
    event_tx: mpsc::Sender<BrowserEvent>,
    miner_id: u64,
) -> anyhow::Result<()>
where
    S: futures_util::Sink<WsMessage, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    R: futures_util::Stream<Item = Result<WsMessage, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    // Wait for Register message
    let miner_address = match read_message(read).await {
        Ok(MinerToPool::Register { address }) => {
            log::info!("Miner {miner_id} registered with address: {address}");
            address
        }
        Ok(other) => {
            log::warn!("Expected Register from {addr}, got {other:?}");
            let _ = write_message(
                write,
                &PoolToMiner::Error {
                    message: "Expected Register message first".to_string(),
                },
            )
            .await;
            return Ok(());
        }
        Err(e) => {
            log::debug!("Failed to read Register from {addr}: {e}");
            return Ok(());
        }
    };

    // Send Registered acknowledgment
    write_message(write, &PoolToMiner::Registered { miner_id }).await?;

    // Create channel for sending messages to this miner
    let (tx, mut rx) = mpsc::channel::<PoolToMiner>(16);

    // Register the miner
    let miner = BrowserMiner {
        id: miner_id,
        address: miner_address,
        tx,
    };
    miners.write().await.insert(miner_id, miner.clone());
    let _ = event_tx.send(BrowserEvent::MinerConnected(miner)).await;

    // Wait for Ready message
    match read_message(read).await {
        Ok(MinerToPool::Ready) => {
            log::debug!("Miner {miner_id} is ready");
        }
        Ok(other) => {
            log::warn!("Expected Ready from miner {miner_id}, got {other:?}");
        }
        Err(e) => {
            log::debug!("Failed to read Ready from miner {miner_id}: {e}");
            miners.write().await.remove(&miner_id);
            let _ = event_tx.send(BrowserEvent::MinerDisconnected(miner_id)).await;
            return Ok(());
        }
    }

    // Main message loop
    loop {
        tokio::select! {
            biased;

            // Receive messages from browser miner
            msg_result = read_message(read) => {
                match msg_result {
                    Ok(MinerToPool::JobResult { job_id, nonce, work, hash_count, elapsed_time }) => {
                        log::info!("Miner {miner_id} submitted result for job {job_id}");
                        let _ = event_tx.send(BrowserEvent::JobResult {
                            miner_id,
                            job_id,
                            nonce,
                            work,
                            hash_count,
                            elapsed_time,
                        }).await;
                    }
                    Ok(MinerToPool::Register { .. }) => {
                        log::warn!("Miner {miner_id} sent duplicate Register");
                    }
                    Ok(MinerToPool::Ready) => {
                        log::debug!("Miner {miner_id} sent duplicate Ready");
                    }
                    Err(e) => {
                        log::debug!("Miner {miner_id} disconnected: {e}");
                        break;
                    }
                }
            }

            // Send messages to browser miner
            msg = rx.recv() => {
                match msg {
                    Some(msg) => {
                        if let Err(e) = write_message(write, &msg).await {
                            log::debug!("Failed to send to miner {miner_id}: {e}");
                            break;
                        }
                    }
                    None => {
                        // Channel closed, shut down
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
    miners.write().await.remove(&miner_id);
    let _ = event_tx.send(BrowserEvent::MinerDisconnected(miner_id)).await;
    log::info!("Miner {miner_id} disconnected");

    Ok(())
}

/// Read a miner message from the WebSocket stream.
async fn read_message<R>(read: &mut R) -> std::io::Result<MinerToPool>
where
    R: futures_util::Stream<Item = Result<WsMessage, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        match read.next().await {
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

/// Write a pool message to the WebSocket sink.
async fn write_message<S>(write: &mut S, msg: &PoolToMiner) -> std::io::Result<()>
where
    S: futures_util::Sink<WsMessage, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
{
    let json = serde_json::to_string(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    write
        .send(WsMessage::Text(json))
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(())
}

/// Create a TLS acceptor with a self-signed certificate.
fn create_tls_acceptor() -> anyhow::Result<TlsAcceptor> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .map_err(|e| anyhow::anyhow!("Failed to generate certificate: {e}"))?;

    let cert_der = cert
        .serialize_der()
        .map_err(|e| anyhow::anyhow!("Failed to serialize certificate: {e}"))?;
    let key_der = cert.serialize_private_key_der();

    let cert_chain = vec![rustls::Certificate(cert_der)];
    let key = rustls::PrivateKey(key_der);

    let mut server_config = rustls::ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .map_err(|e| anyhow::anyhow!("Failed to create TLS config: {e}"))?;

    // Allow HTTP/1.1 for WebSocket upgrade
    server_config.alpn_protocols = vec![b"http/1.1".to_vec()];

    Ok(TlsAcceptor::from(Arc::new(server_config)))
}
