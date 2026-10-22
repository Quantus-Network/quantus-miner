//! Shared QUIC client used by miner-service and pool-service to connect to a
//! quantus-node's external-miner endpoint.
//!
//! The node uses a persisted self-signed certificate. Callers must supply the
//! expected SHA-256 fingerprint of that certificate (lowercase hex); the
//! transport rejects any other server cert.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quantus_miner_api::{write_message, MinerMessage, MAX_MESSAGE_SIZE};
use quinn::crypto::rustls::QuicClientConfig;
use quinn::{ClientConfig, Endpoint};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{ring, CryptoProvider};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use sha2::{Digest, Sha256};

/// Permanent misconfiguration / auth failure: callers must not reconnect-loop.
#[derive(Debug)]
pub struct PermanentConnectError(pub String);

impl std::fmt::Display for PermanentConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PermanentConnectError {}

/// Parse and validate miner auth inputs that never become valid by retrying.
///
/// Call this at process startup (before workers / HTTP) so a bad fingerprint
/// or oversized token exits immediately instead of looping forever.
pub fn validate_auth_config(auth_token: &str, tls_cert_sha256_hex: &str) -> anyhow::Result<()> {
    parse_fingerprint(tls_cert_sha256_hex)?;
    validate_ready_frame(auth_token)?;
    Ok(())
}

/// Parse a 64-hex (optional whitespace/`:` separators) SHA-256 fingerprint.
pub fn parse_fingerprint(tls_cert_sha256_hex: &str) -> anyhow::Result<[u8; 32]> {
    normalize_fingerprint(tls_cert_sha256_hex).ok_or_else(|| {
        PermanentConnectError("invalid TLS cert SHA-256 fingerprint (expected 64 hex chars)".into())
            .into()
    })
}

fn validate_ready_frame(auth_token: &str) -> anyhow::Result<()> {
    let frame = serde_json::to_vec(&MinerMessage::Ready {
        token: auth_token.to_string(),
    })
    .map_err(|e| PermanentConnectError(format!("failed to serialize Ready frame: {e}")))?;
    if frame.len() > MAX_MESSAGE_SIZE as usize {
        return Err(PermanentConnectError(format!(
            "auth token serializes to a {}-byte Ready frame; max is {} \
             (shorten the token or avoid characters that require JSON escaping)",
            frame.len(),
            MAX_MESSAGE_SIZE
        ))
        .into());
    }
    Ok(())
}

/// Establish a QUIC connection to the node's external-miner endpoint, open
/// the bidirectional stream, and send the initial `Ready { token }` message.
///
/// `tls_cert_sha256_hex` is the lowercase hex SHA-256 of the node's miner TLS
/// certificate DER (contents of the node's `miner-tls-cert-sha256` file).
///
/// Wrong tokens are rejected by the node closing with "auth failed"; that is
/// returned as [`PermanentConnectError`] so callers do not reconnect forever.
pub async fn connect(
    addr: SocketAddr,
    auth_token: &str,
    tls_cert_sha256_hex: &str,
) -> anyhow::Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream)> {
    // Re-check here so library callers that skip startup validation still fail
    // closed instead of looping on a deterministic parse error.
    let expected = parse_fingerprint(tls_cert_sha256_hex)?;
    validate_ready_frame(auth_token)?;

    let provider = Arc::new(ring::default_provider());
    let mut crypto = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(PinnedCertVerifier { expected, provider }))
        .with_no_client_auth();
    // Versioned with the wire protocol (node side: MINER_ALPN). `/2` = the
    // authenticated `Ready { token }` protocol; a mismatched node/miner pair
    // fails at the TLS handshake with "no application protocol" instead of an
    // opaque auth/deserialize error.
    crypto.alpn_protocols = vec![b"quantus-miner/2".to_vec()];

    let mut client_config = ClientConfig::new(Arc::new(QuicClientConfig::try_from(crypto)?));
    let mut transport_config = quinn::TransportConfig::default();
    transport_config.keep_alive_interval(Some(Duration::from_secs(5)));
    transport_config.max_idle_timeout(Some(Duration::from_secs(15).try_into().unwrap()));
    client_config.transport_config(Arc::new(transport_config));

    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
    endpoint.set_default_client_config(client_config);

    let connection = match endpoint.connect(addr, "localhost")?.await {
        Ok(c) => c,
        Err(e) => {
            let msg = e.to_string();
            // Wrong pin fails the TLS handshake deterministically.
            if msg.to_ascii_lowercase().contains("fingerprint") {
                return Err(PermanentConnectError(format!(
                    "TLS certificate pin rejected ({msg}); check --tls-cert-sha256 / \
                     miner-tls-cert-sha256"
                ))
                .into());
            }
            return Err(e.into());
        }
    };
    let (mut send, recv) = connection.open_bi().await?;
    write_message(
        &mut send,
        &MinerMessage::Ready {
            token: auth_token.to_string(),
        },
    )
    .await?;

    // Ready is fire-and-forget on the wire; the node closes immediately on a
    // bad token. Wait briefly so that surfaces as a permanent error here
    // instead of "connected" + 1s reconnect churn.
    const AUTH_REJECT_GRACE: Duration = Duration::from_secs(2);
    tokio::select! {
        reason = connection.closed() => {
            let msg = reason.to_string();
            if msg.to_ascii_lowercase().contains("auth") {
                return Err(PermanentConnectError(format!(
                    "node rejected miner auth ({msg}); check --auth-token / miner-auth-token"
                ))
                .into());
            }
            return Err(anyhow::anyhow!("connection closed during auth handshake: {msg}"));
        }
        _ = tokio::time::sleep(AUTH_REJECT_GRACE) => {}
    }

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
#[derive(Debug)]
struct PinnedCertVerifier {
    expected: [u8; 32],
    provider: Arc<CryptoProvider>,
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
        let actual = Sha256::digest(end_entity.as_ref());
        if actual.as_slice() != self.expected.as_slice() {
            return Err(rustls::Error::General(
                "server certificate SHA-256 fingerprint mismatch".into(),
            ));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_fingerprint() {
        let err = validate_auth_config("token", "nope").unwrap_err();
        assert!(
            err.downcast_ref::<PermanentConnectError>().is_some(),
            "{err}"
        );
    }

    #[test]
    fn accepts_valid_fingerprint_and_token() {
        let fp = "a".repeat(64);
        validate_auth_config("deadbeef", &fp).unwrap();
    }

    #[test]
    fn rejects_token_whose_ready_frame_exceeds_limit() {
        let fp = "a".repeat(64);
        // Escape-heavy token: raw length under a naive cap can still blow the frame.
        let token = "\"".repeat(600);
        let err = validate_auth_config(&token, &fp).unwrap_err();
        assert!(
            err.downcast_ref::<PermanentConnectError>().is_some(),
            "{err}"
        );
        assert!(err.to_string().contains("Ready frame"), "{err}");
    }

    /// Minimal stand-in for the node's miner endpoint: self-signed cert, same
    /// ALPN, accepts the bi-stream and holds the connection open.
    async fn spawn_node_stub() -> (SocketAddr, String) {
        let certified = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert_der = certified.cert.der().clone();
        let key =
            rustls::pki_types::PrivatePkcs8KeyDer::from(certified.signing_key.serialize_der());
        let mut server_crypto =
            rustls::ServerConfig::builder_with_provider(Arc::new(ring::default_provider()))
                .with_protocol_versions(&[&rustls::version::TLS13])
                .unwrap()
                .with_no_client_auth()
                .with_single_cert(vec![cert_der.clone()], key.into())
                .unwrap();
        server_crypto.alpn_protocols = vec![b"quantus-miner/2".to_vec()];
        let server_config = quinn::ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto).unwrap(),
        ));
        let endpoint = Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = endpoint.local_addr().unwrap();
        tokio::spawn(async move {
            while let Some(incoming) = endpoint.accept().await {
                tokio::spawn(async move {
                    if let Ok(connection) = incoming.await {
                        let _streams = connection.accept_bi().await;
                        tokio::time::sleep(Duration::from_secs(10)).await;
                    }
                });
            }
        });
        (addr, format!("{:x}", Sha256::digest(&cert_der)))
    }

    #[tokio::test]
    async fn wrong_pin_is_rejected_as_permanent_error() {
        let (addr, _) = spawn_node_stub().await;
        let err = connect(addr, "token", &"b".repeat(64)).await.unwrap_err();
        assert!(
            err.downcast_ref::<PermanentConnectError>().is_some(),
            "{err}"
        );
        assert!(err.to_string().contains("pin rejected"), "{err}");
    }

    #[tokio::test]
    async fn pinned_cert_is_accepted() {
        let (addr, fingerprint) = spawn_node_stub().await;
        connect(addr, "token", &fingerprint).await.unwrap();
    }
}
