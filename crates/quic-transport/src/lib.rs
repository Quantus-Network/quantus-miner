//! Shared QUIC client used by miner-service and pool-service to connect to a
//! quantus-node's external-miner endpoint. Nodes use self-signed certificates,
//! so this transport intentionally skips certificate verification; keeping
//! that pattern in exactly one place is the point of this crate.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quantus_miner_api::{write_message, MinerMessage};
use quinn::{ClientConfig, Endpoint};
use rustls::client::ServerCertVerified;

/// Establish a QUIC connection to the node's external-miner endpoint, open
/// the bidirectional stream, and send the initial `Ready { token }` message.
pub async fn connect(
    addr: SocketAddr,
    auth_token: &str,
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
    write_message(
        &mut send,
        &MinerMessage::Ready {
            token: auth_token.to_string(),
        },
    )
    .await?;
    Ok((connection, send, recv))
}

/// Accepts any certificate (nodes use self-signed certs).
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
