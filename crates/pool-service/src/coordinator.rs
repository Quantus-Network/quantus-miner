//! Pool coordinator that manages job distribution and result collection.
//!
//! The coordinator:
//! - Receives jobs from the node via QUIC
//! - Distributes jobs to connected browser miners via WebSocket
//! - Collects results and submits winning solutions to the node
//! - Tracks block wins for batch payouts

use std::sync::Arc;

use pool_api::PoolToMiner;
use quantus_miner_api::MiningResult;
use tokio::sync::{mpsc, RwLock};

use crate::node_client::{create_mining_result, NodeClient, NodeEvent};
use crate::websocket::{BrowserEvent, WebSocketServer};
use crate::wins::WinTracker;
use crate::PoolConfig;

/// Current job being mined.
#[derive(Debug, Clone)]
struct CurrentJob {
    job_id: String,
    mining_hash: String,
    difficulty: String,
}

/// Pool coordinator.
pub struct PoolCoordinator {
    config: PoolConfig,
}

impl PoolCoordinator {
    /// Create a new pool coordinator.
    pub fn new(config: PoolConfig) -> Self {
        Self { config }
    }

    /// Run the pool coordinator.
    pub async fn run(self) -> anyhow::Result<()> {
        // Create channels for communication
        let (node_event_tx, mut node_event_rx) = mpsc::channel::<NodeEvent>(32);
        let (result_tx, result_rx) = mpsc::channel::<MiningResult>(32);
        let (browser_event_tx, mut browser_event_rx) = mpsc::channel::<BrowserEvent>(64);

        // Create win tracker
        let storage_path = std::env::current_dir().ok().map(|p| p.join("pool_wins.jsonl"));
        let win_tracker = Arc::new(RwLock::new(WinTracker::new(
            self.config.payout_batch_size,
            storage_path,
        )));

        // Create WebSocket server
        let ws_server = Arc::new(WebSocketServer::new(browser_event_tx));

        // Current job state
        let current_job: Arc<RwLock<Option<CurrentJob>>> = Arc::new(RwLock::new(None));

        // Start node client task
        let node_client = NodeClient::new(self.config.node_addr);
        let node_task = tokio::spawn(async move {
            if let Err(e) = node_client.run(node_event_tx, result_rx).await {
                log::error!("Node client error: {e}");
            }
        });

        // Start WebSocket server task
        let ws_port = self.config.ws_port;
        let ws_server_clone = ws_server.clone();
        let ws_task = tokio::spawn(async move {
            if let Err(e) = ws_server_clone.run(ws_port).await {
                log::error!("WebSocket server error: {e}");
            }
        });

        log::info!("Pool coordinator started");

        // Main event loop
        loop {
            tokio::select! {
                biased;

                // Events from node
                Some(event) = node_event_rx.recv() => {
                    match event {
                        NodeEvent::NewJob(request) => {
                            log::info!(
                                "New job from node: id={}, difficulty={}",
                                request.job_id,
                                request.distance_threshold
                            );

                            // Store current job
                            {
                                let mut job = current_job.write().await;
                                *job = Some(CurrentJob {
                                    job_id: request.job_id.clone(),
                                    mining_hash: request.mining_hash.clone(),
                                    difficulty: request.distance_threshold.clone(),
                                });
                            }

                            // Broadcast to all browser miners
                            let msg = PoolToMiner::NewJob {
                                job_id: request.job_id,
                                mining_hash: request.mining_hash,
                                difficulty: request.distance_threshold,
                            };
                            ws_server.broadcast(msg).await;

                            let count = ws_server.miner_count().await;
                            log::info!("Broadcasted job to {count} browser miner(s)");
                        }
                        NodeEvent::Disconnected => {
                            log::warn!("Lost connection to node");
                            // Clear current job
                            *current_job.write().await = None;
                        }
                    }
                }

                // Events from browser miners
                Some(event) = browser_event_rx.recv() => {
                    match event {
                        BrowserEvent::MinerConnected(miner) => {
                            log::info!(
                                "Browser miner {} connected: address={}",
                                miner.id,
                                miner.address
                            );

                            // Send current job if there is one
                            if let Some(job) = current_job.read().await.as_ref() {
                                let msg = PoolToMiner::NewJob {
                                    job_id: job.job_id.clone(),
                                    mining_hash: job.mining_hash.clone(),
                                    difficulty: job.difficulty.clone(),
                                };
                                ws_server.send_to(miner.id, msg).await;
                            }
                        }
                        BrowserEvent::MinerDisconnected(miner_id) => {
                            log::info!("Browser miner {miner_id} disconnected");
                        }
                        BrowserEvent::JobResult {
                            miner_id,
                            job_id,
                            nonce,
                            work,
                            hash_count,
                            elapsed_time,
                        } => {
                            // Check if this is for the current job
                            let current = current_job.read().await;
                            if current.as_ref().map(|j| &j.job_id) != Some(&job_id) {
                                log::debug!(
                                    "Ignoring stale result from miner {miner_id} for job {job_id}"
                                );
                                continue;
                            }
                            drop(current);

                            log::info!(
                                "Miner {miner_id} found solution for job {job_id}!"
                            );

                            // Get miner's address for win tracking
                            let miner_address = ws_server
                                .get_miner(miner_id)
                                .await
                                .map(|m| m.address.clone())
                                .unwrap_or_else(|| "unknown".to_string());

                            // Submit to node
                            let result = create_mining_result(
                                job_id.clone(),
                                nonce,
                                work,
                                hash_count,
                                elapsed_time,
                            );

                            if result_tx.send(result).await.is_err() {
                                log::error!("Failed to send result to node client");
                                continue;
                            }

                            // Record the win
                            // Note: We record the win optimistically. In a production system,
                            // we'd wait for confirmation from the node that the block was accepted.
                            let payout_batch = {
                                let mut tracker = win_tracker.write().await;
                                // TODO: Get actual block number from node
                                tracker.record_win(miner_address.clone(), 0, job_id.clone())
                            };

                            // Notify the winning miner
                            ws_server
                                .send_to(
                                    miner_id,
                                    PoolToMiner::BlockWon {
                                        block_number: 0, // TODO: Get from node
                                        job_id: job_id.clone(),
                                    },
                                )
                                .await;

                            // Notify other miners that job is complete
                            ws_server
                                .broadcast(PoolToMiner::JobCompleted {
                                    job_id,
                                    you_won: false,
                                })
                                .await;

                            // Process payout batch if ready
                            if let Some(batch) = payout_batch {
                                log::info!(
                                    "Payout batch ready with {} wins!",
                                    batch.len()
                                );
                                // TODO: Actually send the wormhole transaction
                                // For now, just log the batch
                                for win in &batch {
                                    log::info!(
                                        "  - {} won block {} (job {})",
                                        win.miner_address,
                                        win.block_number,
                                        win.job_id
                                    );
                                }

                                // Mark as paid (even though we haven't actually paid yet)
                                // In production, this would happen after the tx is confirmed
                                win_tracker.write().await.mark_paid(&batch);
                            }
                        }
                    }
                }

                // Handle shutdown signals
                _ = tokio::signal::ctrl_c() => {
                    log::info!("Received shutdown signal");
                    break;
                }
            }
        }

        // Cleanup
        node_task.abort();
        ws_task.abort();

        log::info!("Pool coordinator stopped");
        Ok(())
    }
}
