#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

//! Minimal telemetry client for the Quantus External Miner.
//!
//! Capabilities:
//! - Connects to one or more telemetry endpoints (wss://...) asynchronously.
//! - Sends upstream-compatible envelopes with a stable per-process session id.
//! - Provides helpers to emit `system.connected` and `system.interval` events.
//! - Non-blocking: messages are buffered via bounded channels; on overflow, they are dropped.
//! - Auto-reconnects on send error; subsequent messages trigger reconnect attempts.
//!
//! Notes:
//! - This is intentionally small and self-contained. It does not embed Substrate's sc-telemetry.
//! - Endpoints must currently be WebSocket URLs (e.g., wss://telemetry.example/submit/).
//! - If no endpoints are configured or telemetry is disabled, the handle becomes a no-op.

use anyhow::Result;
use futures::{SinkExt, StreamExt};
use log::{debug, info, warn};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::MaybeTlsStream;
use tokio_tungstenite::WebSocketStream;
use uuid::Uuid;

/// Bounded channel capacity per endpoint writer task.
const TELEMETRY_CHANNEL_CAPACITY: usize = 1024;

/// Telemetry configuration.
#[derive(Clone, Debug, Default)]
pub struct TelemetryConfig {
    /// Enable/disable telemetry entirely.
    pub enabled: bool,
    /// WebSocket endpoints to send telemetry to (e.g., "wss://telemetry.example/submit/").
    pub endpoints: Vec<String>,
    /// Default verbosity (u8), applied if not overridden per message.
    pub verbosity: u8,
    /// Optional static identity to put in system.connected.
    pub name: Option<String>,
    pub implementation: Option<String>,
    pub version: Option<String>,
    pub chain: Option<String>,
    pub genesis_hash: Option<String>,
    /// Interval (seconds) recommendation for system.interval from the service (not enforced here).
    pub interval_secs: Option<u64>,
    /// Optional default association with a node (used when job-specific context doesn't provide one).
    pub default_link: Option<TelemetryNodeLink>,
}

/// Optional association with a node that the miner is working for.
#[derive(Clone, Debug, Default, Serialize)]
pub struct TelemetryNodeLink {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_telemetry_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_peer_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub genesis_hash: Option<String>,
}

/// Snapshot used to populate `system.interval` payloads.
#[derive(Clone, Debug, Default)]
pub struct SystemInterval {
    pub uptime_ms: u64,
    pub engine: Option<String>,
    pub workers: Option<u32>,
    pub hash_rate: Option<f64>,
    pub active_jobs: Option<i64>,
    /// If you want to hint which node is primarily linked when multiple jobs exist.
    pub linked_node_hint: Option<String>,
}

/// Cloneable handle to send telemetry events.
#[derive(Clone)]
pub struct TelemetryHandle {
    session_id: String,
    verbosity: u8,
    // One sender per endpoint task.
    senders: Arc<Vec<mpsc::Sender<EnvelopeMsg>>>,
    // Keep config for convenience payload builders.
    config: Arc<TelemetryConfig>,
}

impl TelemetryHandle {
    /// Returns true if this handle will actually send to any endpoint.
    pub fn is_enabled(&self) -> bool {
        !self.senders.is_empty()
    }

    /// Returns the stable session id used in the envelope `id`.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Emit a fully custom payload (must include `"msg"`).
    pub async fn emit_payload(&self, payload: Value, verbosity: Option<u8>) {
        if self.senders.is_empty() {
            return;
        }
        let msg = EnvelopeMsg { payload, verbosity };
        for tx in self.senders.iter() {
            // Drop on full; do not block mining paths. This is intentionally lossy under backpressure.
            if tx.try_send(msg.clone()).is_err() {
                // Best-effort: if try_send failed due to full buffer, attempt bounded send with small timeout.
                // Avoid blocking indefinitely.
                let tx = tx.clone();
                let msg = msg.clone();
                tokio::spawn(async move {
                    let _ = tx.send(msg).await;
                });
            }
        }
    }

    /// Emit a `system.connected` message using configuration defaults and an optional link.
    pub async fn emit_system_connected(&self, link: Option<&TelemetryNodeLink>) {
        let payload = build_system_connected_payload(&self.config, link);
        self.emit_payload(payload, Some(self.verbosity)).await;
    }

    /// Emit a `system.interval` message with lightweight health stats and an optional link hint.
    pub async fn emit_system_interval(&self, interval: &SystemInterval, link_hint: Option<&str>) {
        let mut payload = json!({
            "msg": "system.interval",
            "uptime_ms": interval.uptime_ms,
        });

        if let Some(engine) = &interval.engine {
            payload
                .as_object_mut()
                .unwrap()
                .insert("engine".to_string(), json!(engine));
        }
        if let Some(workers) = interval.workers {
            payload
                .as_object_mut()
                .unwrap()
                .insert("workers".to_string(), json!(workers));
        }
        if let Some(hash_rate) = interval.hash_rate {
            payload
                .as_object_mut()
                .unwrap()
                .insert("hash_rate".to_string(), json!(hash_rate));
        }
        if let Some(active_jobs) = interval.active_jobs {
            payload
                .as_object_mut()
                .unwrap()
                .insert("active_jobs".to_string(), json!(active_jobs));
        }
        if let Some(hint) = link_hint {
            payload
                .as_object_mut()
                .unwrap()
                .insert("linked_node_hint".to_string(), json!(hint));
        }

        self.emit_payload(payload, Some(self.verbosity)).await;
    }
}

/// Build a `system.connected` payload with upstream-compatible keys and optional `linked_node`.
fn build_system_connected_payload(
    config: &TelemetryConfig,
    link: Option<&TelemetryNodeLink>,
) -> Value {
    let name = config
        .name
        .as_deref()
        .unwrap_or("quantus-miner")
        .to_string();
    let implementation = config
        .implementation
        .as_deref()
        .unwrap_or("quantus-miner")
        .to_string();
    // Prefer build-time env var MINER_VERSION or fallback to crate version if the caller sets it.
    let version = config.version.clone().unwrap_or_else(|| {
        option_env!("MINER_VERSION")
            .map(|s| s.to_string())
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
    });

    let mut payload = json!({
        "msg": "system.connected",
        "name": name,
        "implementation": implementation,
        "version": version,
        "authority": false,
        "platform": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH
        }
    });

    if let Some(chain) = &config.chain {
        payload
            .as_object_mut()
            .unwrap()
            .insert("chain".to_string(), json!(chain));
    }
    if let Some(gh) = &config.genesis_hash {
        payload
            .as_object_mut()
            .unwrap()
            .insert("genesis_hash".to_string(), json!(gh));
    }

    let linked = link.or(config.default_link.as_ref());
    if let Some(linked_node) = linked {
        let v = serde_json::to_value(linked_node).unwrap_or(json!({}));
        if let Some(obj) = v.as_object() {
            // Only include non-empty association
            if !obj.is_empty() {
                payload
                    .as_object_mut()
                    .unwrap()
                    .insert("linked_node".to_string(), Value::Object(obj.clone()));
            }
        }
    }

    payload
}

/// Telemetry envelope to send to the transport.
#[derive(Clone, Debug)]
struct EnvelopeMsg {
    payload: Value,
    verbosity: Option<u8>,
}

/// Start the telemetry subsystem with the given configuration.
/// Returns a `TelemetryHandle` for emitting events. If `config.enabled` is false
/// or no endpoints are specified, the handle is a no-op.
pub fn start(config: TelemetryConfig) -> TelemetryHandle {
    let enabled = config.enabled && !config.endpoints.is_empty();
    let session_id = Uuid::new_v4().to_string();
    let verbosity = config.verbosity;
    let cfg = Arc::new(config);

    let mut senders: Vec<mpsc::Sender<EnvelopeMsg>> = Vec::new();

    if enabled {
        for ep in &cfg.endpoints {
            let (tx, rx) = mpsc::channel::<EnvelopeMsg>(TELEMETRY_CHANNEL_CAPACITY);
            senders.push(tx);
            spawn_endpoint_writer_task(ep.clone(), rx, session_id.clone(), verbosity);
        }
        info!(
            "miner-telemetry: started with {} endpoint(s), session_id={}",
            cfg.endpoints.len(),
            session_id
        );
    } else {
        info!("miner-telemetry: disabled (no endpoints or disabled in config)");
    }

    TelemetryHandle {
        session_id,
        verbosity,
        senders: Arc::new(senders),
        config: cfg,
    }
}

/// Spawn a writer task for a single endpoint. It owns a receiver and attempts to keep
/// a write connection alive, reconnecting on failure. Messages received while disconnected
/// are dropped (best-effort model).
fn spawn_endpoint_writer_task(
    endpoint: String,
    mut rx: mpsc::Receiver<EnvelopeMsg>,
    session_id: String,
    default_verbosity: u8,
) {
    tokio::spawn(async move {
        // Writer half of the websocket.
        type WsWrite =
            futures::stream::SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;

        let mut writer: Option<WsWrite> = None;
        let mut backoff_secs: u64 = 1;

        loop {
            // Receive next message to send. If all senders are gone, exit task gracefully.
            let Some(msg) = rx.recv().await else {
                debug!(
                    "miner-telemetry: endpoint task for {} shutting down (channel closed)",
                    endpoint
                );
                break;
            };

            // Ensure connectivity prior to sending.
            if writer.is_none() {
                match connect_ws(&endpoint).await {
                    Ok(w) => {
                        writer = Some(w);
                        backoff_secs = 1; // reset backoff after success
                        debug!("miner-telemetry: connected to {}", endpoint);
                    }
                    Err(e) => {
                        warn!(
                            "miner-telemetry: connect failed to {}: {e}, backing off {}s",
                            endpoint, backoff_secs
                        );
                        sleep(Duration::from_secs(backoff_secs)).await;
                        backoff_secs = (backoff_secs.saturating_mul(2)).min(30);
                        // Drop this message and continue; newer messages will trigger subsequent attempts.
                        continue;
                    }
                }
            }

            // Build envelope with timestamp and (optional) verbosity.
            let ts = now_ms();
            let verbosity = msg.verbosity.unwrap_or(default_verbosity);
            let envelope = json!({
                "id": session_id,
                "ts": ts,
                "verbosity": verbosity,
                "payload": msg.payload
            });
            let line = match serde_json::to_string(&envelope) {
                Ok(s) => s,
                Err(e) => {
                    warn!("miner-telemetry: failed to serialize envelope: {e}");
                    continue;
                }
            };

            // Send over WebSocket
            let mut send_ok = false;
            if let Some(w) = writer.as_mut() {
                match w.send(Message::Text(line)).await {
                    Ok(()) => {
                        send_ok = true;
                    }
                    Err(e) => {
                        warn!(
                            "miner-telemetry: send error to {}: {e}. Will drop connection and reconnect.",
                            endpoint
                        );
                    }
                }
            }

            if !send_ok {
                writer = None; // drop and reconnect on next message
                               // Small backoff to avoid hot-loop if immediate failures keep happening.
                sleep(Duration::from_millis(250)).await;
            }
        }
    });
}

/// Establish a websocket connection and return the writer half.
/// A reader task is also spawned to continuously drain incoming frames (pings, acks, etc.).
async fn connect_ws(
    endpoint: &str,
) -> Result<futures::stream::SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>> {
    let url = endpoint.to_string();
    let (ws, _resp) = tokio_tungstenite::connect_async(url).await?;
    let (write, mut read) = ws.split();

    // Spawn a task to drain incoming frames to keep the connection healthy.
    tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Ping(data)) => {
                    debug!("miner-telemetry: received ping ({} bytes)", data.len());
                }
                Ok(Message::Pong(_)) => {
                    // no-op
                }
                Ok(Message::Text(_)) | Ok(Message::Binary(_)) => {
                    // Most telemetry servers don't push data; ignore.
                }
                Ok(Message::Frame(_)) => {
                    // Ignore low-level frames; keep connection alive
                }
                Ok(Message::Close(frame)) => {
                    debug!("miner-telemetry: server closed connection: {:?}", frame);
                    break;
                }
                Err(e) => {
                    debug!("miner-telemetry: read error: {e}");
                    break;
                }
            }
        }
        debug!("miner-telemetry: reader task ended");
    });

    Ok(write)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}

// -------------------------------------------------------------------------------------
// Tests (basic serialization and handle behavior)
// -------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_connected_payload() {
        let cfg = TelemetryConfig {
            enabled: true,
            endpoints: vec![],
            verbosity: 0,
            name: Some("qm".into()),
            implementation: Some("quantus-miner".into()),
            version: Some("0.0.1".into()),
            chain: Some("Resonance".into()),
            genesis_hash: Some("0xabc".into()),
            interval_secs: Some(15),
            default_link: Some(TelemetryNodeLink {
                node_telemetry_id: Some("node-uuid".into()),
                node_peer_id: Some("12D3KooW..".into()),
                node_name: Some("node01".into()),
                node_version: Some("1.2.3".into()),
                chain: Some("Resonance".into()),
                genesis_hash: Some("0xabc".into()),
            }),
        };
        let payload = build_system_connected_payload(&cfg, None);
        assert_eq!(
            payload.get("msg").and_then(|v| v.as_str()),
            Some("system.connected")
        );
        assert_eq!(payload.get("name").and_then(|v| v.as_str()), Some("qm"));
        assert_eq!(
            payload.get("chain").and_then(|v| v.as_str()),
            Some("Resonance")
        );
        assert!(payload.get("linked_node").is_some());
    }

    #[tokio::test]
    async fn test_handle_noop_when_disabled() {
        let cfg = TelemetryConfig::default();
        let handle = start(cfg);
        assert!(!handle.is_enabled());
        handle
            .emit_system_interval(&SystemInterval::default(), None)
            .await;
        handle.emit_system_connected(None).await;
    }
}
