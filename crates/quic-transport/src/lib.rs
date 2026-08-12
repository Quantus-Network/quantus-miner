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
use quinn::{ClientConfig, Endpoint};
use rustls::client::ServerCertVerified;
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
        PermanentConnectError(
            "invalid TLS cert SHA-256 fingerprint (expected 64 hex chars)".into(),
        )
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

    let mut crypto = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_custom_certificate_verifier(Arc::new(PinnedCertVerifier { expected }))
        .with_no_client_auth();
    // Versioned with the wire protocol (node side: MINER_ALPN). `/2` = the
    // authenticated `Ready { token }` protocol; a mismatched node/miner pair
    // fails at the TLS handshake with "no application protocol" instead of an
    // opaque auth/deserialize error.
    crypto.alpn_protocols = vec![b"quantus-miner/2".to_vec()];

    let mut client_config = ClientConfig::new(Arc::new(crypto));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_fingerprint() {
        let err = validate_auth_config("token", "nope").unwrap_err();
        assert!(err.downcast_ref::<PermanentConnectError>().is_some(), "{err}");
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
        assert!(err.downcast_ref::<PermanentConnectError>().is_some(), "{err}");
        assert!(err.to_string().contains("Ready frame"), "{err}");
    }
}
