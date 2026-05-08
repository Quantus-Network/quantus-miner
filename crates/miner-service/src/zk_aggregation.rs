use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimStrategy {
    Oldest,
    RewardDensity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkAggregationConfig {
    pub node_rpc: String,
    pub aggregation_account: Option<String>,
    pub aggregation_key: Option<String>,
    pub zk_bins_dir: Option<PathBuf>,
    pub workers: usize,
    pub max_active_jobs: usize,
    pub min_aggregation_reward: u128,
    pub claim_strategy: ClaimStrategy,
    pub dry_run: bool,
}

impl ZkAggregationConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.workers == 0 {
            anyhow::bail!("--zk-workers must be greater than zero when ZK aggregation is enabled");
        }
        if self.max_active_jobs == 0 {
            anyhow::bail!(
                "--max-active-zk-jobs must be greater than zero when ZK aggregation is enabled"
            );
        }
        let Some(zk_bins_dir) = &self.zk_bins_dir else {
            anyhow::bail!("--zk-bins-dir is required when ZK aggregation is enabled");
        };
        if !zk_bins_dir.exists() {
            anyhow::bail!(
                "ZK bins directory does not exist: {}",
                zk_bins_dir.display()
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregationJob {
    pub bundle_id: [u8; 32],
    pub aggregator_address: [u8; 32],
    pub candidate_proofs: Vec<Vec<u8>>,
    pub ordered_candidate_ids: Vec<[u8; 32]>,
    pub deadline_block: u32,
}

#[derive(Debug)]
pub struct AggregationWorkerPool {
    config: ZkAggregationConfig,
}

impl AggregationWorkerPool {
    pub fn new(config: ZkAggregationConfig) -> anyhow::Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn max_active_jobs(&self) -> usize {
        self.config.max_active_jobs
    }

    pub fn should_cancel_for_new_pow_job(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation_rejects_missing_zk_bins() {
        let config = ZkAggregationConfig {
            node_rpc: "ws://127.0.0.1:9944".to_string(),
            aggregation_account: None,
            aggregation_key: None,
            zk_bins_dir: None,
            workers: 1,
            max_active_jobs: 1,
            min_aggregation_reward: 0,
            claim_strategy: ClaimStrategy::Oldest,
            dry_run: true,
        };

        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("--zk-bins-dir is required"));
    }

    #[test]
    fn worker_pool_does_not_cancel_on_pow_job() {
        let dir = tempfile::tempdir().unwrap();
        let config = ZkAggregationConfig {
            node_rpc: "ws://127.0.0.1:9944".to_string(),
            aggregation_account: None,
            aggregation_key: None,
            zk_bins_dir: Some(dir.path().to_path_buf()),
            workers: 1,
            max_active_jobs: 1,
            min_aggregation_reward: 0,
            claim_strategy: ClaimStrategy::Oldest,
            dry_run: true,
        };

        let pool = AggregationWorkerPool::new(config).unwrap();
        assert!(!pool.should_cancel_for_new_pow_job());
    }
}
