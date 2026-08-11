//! Shared QUIC client used by miner-service and pool-service to connect to a
//! quantus-node's external-miner endpoint.
//!
//! The node uses a persisted self-signed certificate. Callers must supply the
//! expected SHA-256 fingerprint of that certificate (lowercase hex); the
//! transport rejects any other server cert.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quantus_miner_api::{write_message, MinerMessage};
use quinn::{ClientConfig, Endpoint};
use rustls::client::ServerCertVerified;
use sha2::{Digest, Sha256};

/// Establish a QUIC connection to the node's external-miner endpoint, open
/// the bidirectional stream, and send the initial `Ready { token }` message.
///
/// `tls_cert_sha256_hex` is the lowercase hex SHA-256 of the node's miner TLS
/// certificate DER (contents of the node's `miner-tls-cert-sha256` file).
pub async fn connect(
    addr: SocketAddr,
    auth_token: &str,
    tls_cert_sha256_hex: &str,
) -> anyhow::Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream)> {
    let expected = normalize_fingerprint(tls_cert_sha256_hex)
        .ok_or_else(|| anyhow::anyhow!("invalid TLS cert SHA-256 fingerprint (expected 64 hex chars)"))?;

    let mut crypto = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_custom_certificate_verifier(Arc::new(PinnedCertVerifier { expected }))
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

fn normalize_fingerprint(input: &str) -> Option<[u8; 32]> {
    let hex: String = input
        .chars()
        .filter(|c| !c.is_whitespace() && *c != ':')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

/// Accepts only a server certificate whose SHA-256 fingerprint matches.
struct PinnedCertVerifier {
    expected: [u8; 32],
}

impl rustls::client::ServerCertVerifier for PinnedCertVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let actual = Sha256::digest(&end_entity.0);
        if actual.as_slice() != self.expected.as_slice() {
            return Err(rustls::Error::General(
                "server certificate SHA-256 fingerprint mismatch".into(),
            ));
        }
        Ok(ServerCertVerified::assertion())
    }
}
