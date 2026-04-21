//! Block win tracking and payout batching.
//!
//! Tracks which miners won blocks and manages batch payouts.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// A record of a block win.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BlockWin {
    /// The miner's rewards address.
    pub miner_address: String,
    /// The block number that was won.
    pub block_number: u64,
    /// The job ID that was solved.
    pub job_id: String,
    /// Timestamp when the block was won (Unix epoch seconds).
    pub timestamp: u64,
    /// Whether this win has been paid out.
    pub paid: bool,
}

/// Tracks block wins and manages payout batching.
pub struct WinTracker {
    /// Unpaid wins waiting for batch payout.
    unpaid_wins: VecDeque<BlockWin>,
    /// Number of wins to batch before payout.
    batch_size: usize,
    /// Path to persist wins (for crash recovery).
    storage_path: Option<PathBuf>,
}

impl WinTracker {
    /// Create a new win tracker.
    pub fn new(batch_size: usize, storage_path: Option<PathBuf>) -> Self {
        let mut tracker = Self {
            unpaid_wins: VecDeque::new(),
            batch_size,
            storage_path,
        };

        // Load any unpaid wins from storage
        if let Err(e) = tracker.load_from_storage_if_exists() {
            log::warn!("Failed to load wins from storage: {e}");
        }

        tracker
    }

    /// Load wins from storage if path is configured.
    fn load_from_storage_if_exists(&mut self) -> anyhow::Result<()> {
        let Some(ref path) = self.storage_path else {
            return Ok(());
        };
        let path = path.clone();
        self.load_from_storage(&path)
    }

    /// Record a new block win.
    ///
    /// Returns `Some(Vec<BlockWin>)` if we've accumulated enough wins for a batch payout.
    pub fn record_win(
        &mut self,
        miner_address: String,
        block_number: u64,
        job_id: String,
    ) -> Option<Vec<BlockWin>> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let win = BlockWin {
            miner_address,
            block_number,
            job_id,
            timestamp,
            paid: false,
        };

        log::info!(
            "Block win recorded: address={}, block={}",
            win.miner_address,
            win.block_number
        );

        self.unpaid_wins.push_back(win.clone());
        self.persist_win(&win);

        // Check if we have enough wins for a batch payout
        if self.unpaid_wins.len() >= self.batch_size {
            Some(self.take_batch())
        } else {
            log::info!(
                "Wins pending payout: {}/{}",
                self.unpaid_wins.len(),
                self.batch_size
            );
            None
        }
    }

    /// Take a batch of wins for payout.
    fn take_batch(&mut self) -> Vec<BlockWin> {
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if let Some(win) = self.unpaid_wins.pop_front() {
                batch.push(win);
            }
        }
        batch
    }

    /// Mark wins as paid.
    pub fn mark_paid(&mut self, wins: &[BlockWin]) {
        // Update storage to mark these as paid
        if let Some(ref path) = self.storage_path {
            if let Err(e) = self.update_storage_paid(path, wins) {
                log::error!("Failed to update storage after payout: {e}");
            }
        }

        log::info!("Marked {} wins as paid", wins.len());
    }

    /// Get the number of unpaid wins.
    pub fn unpaid_count(&self) -> usize {
        self.unpaid_wins.len()
    }

    /// Persist a win to storage.
    fn persist_win(&self, win: &BlockWin) {
        let Some(ref path) = self.storage_path else {
            return;
        };

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path);

        match file {
            Ok(mut f) => {
                if let Ok(json) = serde_json::to_string(win) {
                    if let Err(e) = writeln!(f, "{json}") {
                        log::error!("Failed to write win to storage: {e}");
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to open storage file: {e}");
            }
        }
    }

    /// Load wins from storage.
    fn load_from_storage(&mut self, path: &PathBuf) -> anyhow::Result<()> {
        if !path.exists() {
            return Ok(());
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<BlockWin>(&line) {
                Ok(win) if !win.paid => {
                    self.unpaid_wins.push_back(win);
                }
                Ok(_) => {} // Already paid, skip
                Err(e) => {
                    log::warn!("Failed to parse win record: {e}");
                }
            }
        }

        log::info!("Loaded {} unpaid wins from storage", self.unpaid_wins.len());
        Ok(())
    }

    /// Update storage to mark wins as paid.
    fn update_storage_paid(&self, path: &PathBuf, paid_wins: &[BlockWin]) -> anyhow::Result<()> {
        if !path.exists() {
            return Ok(());
        }

        // Read all wins
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut all_wins: Vec<BlockWin> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(win) = serde_json::from_str::<BlockWin>(&line) {
                all_wins.push(win);
            }
        }

        // Mark paid wins
        let paid_blocks: std::collections::HashSet<u64> =
            paid_wins.iter().map(|w| w.block_number).collect();

        for win in &mut all_wins {
            if paid_blocks.contains(&win.block_number) {
                win.paid = true;
            }
        }

        // Rewrite file
        let mut file = File::create(path)?;
        for win in all_wins {
            let json = serde_json::to_string(&win)?;
            writeln!(file, "{json}")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_win_tracking() {
        let mut tracker = WinTracker::new(3, None);

        // First two wins shouldn't trigger payout
        assert!(tracker
            .record_win("addr1".to_string(), 100, "job1".to_string())
            .is_none());
        assert!(tracker
            .record_win("addr2".to_string(), 101, "job2".to_string())
            .is_none());
        assert_eq!(tracker.unpaid_count(), 2);

        // Third win should trigger batch
        let batch = tracker
            .record_win("addr3".to_string(), 102, "job3".to_string())
            .expect("should have batch");
        assert_eq!(batch.len(), 3);
        assert_eq!(tracker.unpaid_count(), 0);
    }

    #[test]
    fn test_block_win_serialization() {
        let win = BlockWin {
            miner_address: "5GrwvaEF...".to_string(),
            block_number: 12345,
            job_id: "abc123".to_string(),
            timestamp: 1700000000,
            paid: false,
        };

        let json = serde_json::to_string(&win).unwrap();
        let parsed: BlockWin = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.miner_address, win.miner_address);
        assert_eq!(parsed.block_number, win.block_number);
        assert!(!parsed.paid);
    }
}
