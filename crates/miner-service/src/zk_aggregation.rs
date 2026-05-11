use anyhow::{anyhow, bail, Context};
use async_trait::async_trait;
use plonky2::plonk::proof::ProofWithPublicInputs as ProverProofWithPublicInputs;
use qp_wormhole_aggregator::aggregator::{AggregationBackend, CircuitType, Layer1Aggregator};
use qp_wormhole_verifier::{
    parse_aggregated_public_inputs, AggregatedPublicCircuitInputs, BytesDigest,
    ProofWithPublicInputs as VerifierProofWithPublicInputs, PublicInputsByAccount,
    WormholeVerifier, C, D, F,
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::{Path, PathBuf},
};

pub const REQUIRED_ZK_ARTIFACTS: &[&str] = &[
    "aggregated_common.bin",
    "aggregated_verifier.bin",
    "layer1_common.bin",
    "layer1_prover.bin",
    "layer1_verifier.bin",
    "config.json",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimStrategy {
    Oldest,
    RewardDensity,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ZkAggregationConfig {
    pub node_rpc: String,
    pub aggregation_account: Option<String>,
    pub aggregation_keystore: Option<PathBuf>,
    pub zk_bins_dir: Option<PathBuf>,
    pub workers: usize,
    pub max_active_jobs: usize,
    pub min_aggregation_reward: u128,
    pub miner_bond: u128,
    pub claim_strategy: ClaimStrategy,
    pub dry_run: bool,
}

impl fmt::Debug for ZkAggregationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZkAggregationConfig")
            .field("node_rpc", &redact_rpc_url(&self.node_rpc))
            .field("aggregation_account", &self.aggregation_account)
            .field("aggregation_keystore", &self.aggregation_keystore)
            .field("zk_bins_dir", &self.zk_bins_dir)
            .field("workers", &self.workers)
            .field("max_active_jobs", &self.max_active_jobs)
            .field("min_aggregation_reward", &self.min_aggregation_reward)
            .field("miner_bond", &self.miner_bond)
            .field("claim_strategy", &self.claim_strategy)
            .field("dry_run", &self.dry_run)
            .finish()
    }
}

impl ZkAggregationConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.node_rpc.trim().is_empty() {
            bail!("--node-rpc must not be empty when ZK aggregation is enabled");
        }
        if self.workers == 0 {
            bail!("--zk-workers must be greater than zero when ZK aggregation is enabled");
        }
        if self.max_active_jobs == 0 {
            bail!("--max-active-zk-jobs must be greater than zero when ZK aggregation is enabled");
        }
        if !self.dry_run && self.miner_bond == 0 {
            bail!(
                "--zk-miner-bond must be greater than zero unless --dry-run-zk-aggregation is set"
            );
        }
        if !self.dry_run {
            match self.aggregation_account.as_deref().map(str::trim) {
                Some(account) if !account.is_empty() => {}
                _ => bail!(
                    "--aggregation-account is required unless --dry-run-zk-aggregation is set"
                ),
            }
            let Some(keystore_path) = &self.aggregation_keystore else {
                bail!("--aggregation-keystore is required unless --dry-run-zk-aggregation is set");
            };
            validate_keystore_path(keystore_path)?;
        } else if let Some(keystore_path) = &self.aggregation_keystore {
            validate_keystore_path(keystore_path)?;
        }

        let Some(zk_bins_dir) = &self.zk_bins_dir else {
            bail!("--zk-bins-dir is required when ZK aggregation is enabled");
        };
        validate_zk_bins_dir(zk_bins_dir)
    }
}

pub fn redact_rpc_url(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return redact_authority(url);
    };
    let (scheme, rest_with_sep) = url.split_at(scheme_end);
    let rest = &rest_with_sep[3..];
    match rest.find('@') {
        Some(at) => {
            let after_auth = &rest[at + 1..];
            let redacted_auth = if rest[..at].contains(':') {
                "***:***"
            } else {
                "***"
            };
            format!("{scheme}://{redacted_auth}@{after_auth}")
        }
        None => url.to_string(),
    }
}

fn redact_authority(value: &str) -> String {
    match value.find('@') {
        Some(at) if value[..at].contains(':') => format!("***:***@{}", &value[at + 1..]),
        Some(at) => format!("***@{}", &value[at + 1..]),
        None => value.to_string(),
    }
}

pub fn validate_keystore_path(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        bail!(
            "aggregation keystore path does not exist: {}",
            path.display()
        );
    }
    let metadata = path.metadata().with_context(|| {
        format!(
            "failed to read aggregation keystore metadata: {}",
            path.display()
        )
    })?;
    if !metadata.is_file() && !metadata.is_dir() {
        bail!(
            "aggregation keystore path must be a file or directory: {}",
            path.display()
        );
    }
    validate_keystore_permissions(path, &metadata)
}

#[cfg(unix)]
fn validate_keystore_permissions(path: &Path, metadata: &std::fs::Metadata) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mode = metadata.permissions().mode();
    if mode & 0o077 != 0 {
        bail!(
            "aggregation keystore path permissions are too broad for {}: expected no group/other permissions",
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_keystore_permissions(
    _path: &Path,
    _metadata: &std::fs::Metadata,
) -> anyhow::Result<()> {
    Ok(())
}

pub fn validate_zk_bins_dir(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        bail!("ZK bins directory does not exist: {}", path.display());
    }
    if !path.is_dir() {
        bail!("ZK bins path is not a directory: {}", path.display());
    }

    let missing = REQUIRED_ZK_ARTIFACTS
        .iter()
        .filter(|file| !path.join(file).is_file())
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "ZK bins directory {} is missing required artifact(s): {}",
            path.display(),
            missing.join(", ")
        );
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BundleGroupKey {
    pub circuit_id: [u8; 32],
    pub public_input_layout_version: u32,
    pub num_leaf_proofs: u32,
    pub num_layer0_proofs: u32,
    pub asset_id: u32,
    pub volume_fee_bps: u32,
    pub block_hash: [u8; 32],
    pub block_number: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExitSlotSummary {
    pub summed_output_amount: u32,
    pub exit_account: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L0CandidateSummary {
    pub candidate_id: [u8; 32],
    pub group_key: BundleGroupKey,
    pub submitted_at: u32,
    pub expires_at: u32,
    pub aggregation_tip: u128,
    pub estimated_reward: u128,
    pub estimated_proving_cost: u64,
    pub nullifiers: Vec<[u8; 32]>,
    pub exit_summary: Vec<ExitSlotSummary>,
}

impl L0CandidateSummary {
    fn reward_for_selection(&self) -> u128 {
        self.estimated_reward.max(self.aggregation_tip)
    }

    fn proving_cost_for_selection(&self) -> u64 {
        self.estimated_proving_cost.max(1)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimedBundle {
    pub bundle_id: [u8; 32],
    pub group_key: BundleGroupKey,
    pub ordered_candidate_ids: Vec<[u8; 32]>,
    pub aggregator_address: [u8; 32],
    pub deadline_block: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateBatch {
    pub group: BundleGroupKey,
    pub candidates: Vec<L0CandidateSummary>,
    pub total_reward: u128,
    pub estimated_proving_cost: u64,
    pub oldest_submitted_at: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedL0Candidate {
    pub candidate_id: [u8; 32],
    pub nullifiers: Vec<[u8; 32]>,
    pub inputs: Option<AggregatedPublicCircuitInputs>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregationRunOutcome {
    NoOpportunity,
    DryRunValidated {
        group: BundleGroupKey,
        candidate_count: usize,
        total_reward: u128,
    },
    Submitted {
        bundle_id: [u8; 32],
        proof_len: usize,
    },
}

#[async_trait]
pub trait AggregationChainClient: Send + Sync {
    async fn current_block(&self) -> anyhow::Result<u32>;
    async fn pending_groups(&self) -> anyhow::Result<Vec<BundleGroupKey>>;
    async fn pending_candidates(
        &self,
        group: &BundleGroupKey,
    ) -> anyhow::Result<Vec<L0CandidateSummary>>;
    async fn fetch_candidate_proof(&self, candidate_id: [u8; 32]) -> anyhow::Result<Vec<u8>>;
    async fn register_aggregator_if_needed(
        &self,
        config: &ZkAggregationConfig,
    ) -> anyhow::Result<()>;
    async fn claim_bundle(
        &self,
        group: BundleGroupKey,
        miner_bond: u128,
    ) -> anyhow::Result<ClaimedBundle>;
    async fn fetch_bundle(&self, bundle_id: [u8; 32]) -> anyhow::Result<ClaimedBundle>;
    async fn submit_l1_aggregate(
        &self,
        bundle_id: [u8; 32],
        proof_bytes: Vec<u8>,
    ) -> anyhow::Result<()>;
}

pub trait L0ProofValidator: Send + Sync {
    fn validate_candidate(
        &self,
        group: &BundleGroupKey,
        candidate: &L0CandidateSummary,
        proof_bytes: &[u8],
    ) -> anyhow::Result<ValidatedL0Candidate>;
}

#[async_trait]
pub trait L1ProofGenerator: Send + Sync {
    async fn prove_l1(
        &self,
        bundle: &ClaimedBundle,
        candidate_proofs: Vec<Vec<u8>>,
    ) -> anyhow::Result<Vec<u8>>;

    fn verify_l1(&self, proof_bytes: &[u8]) -> anyhow::Result<()>;
}

#[derive(Debug)]
pub struct LocalL0ProofValidator {
    verifier: WormholeVerifier,
}

impl LocalL0ProofValidator {
    pub fn new_from_bins_dir(bins_dir: &Path) -> anyhow::Result<Self> {
        validate_zk_bins_dir(bins_dir)?;
        let verifier = WormholeVerifier::new_from_files(
            &bins_dir.join("aggregated_verifier.bin"),
            &bins_dir.join("aggregated_common.bin"),
        )
        .context("failed to load layer-0 aggregate verifier artifacts")?;

        Ok(Self { verifier })
    }
}

impl L0ProofValidator for LocalL0ProofValidator {
    fn validate_candidate(
        &self,
        group: &BundleGroupKey,
        candidate: &L0CandidateSummary,
        proof_bytes: &[u8],
    ) -> anyhow::Result<ValidatedL0Candidate> {
        if candidate.group_key != *group {
            bail!("candidate group does not match selected bundle group");
        }

        let proof = VerifierProofWithPublicInputs::<F, C, D>::from_bytes(
            proof_bytes.to_vec(),
            &self.verifier.circuit_data.common,
        )
        .map_err(|err| anyhow!("failed to deserialize L0 aggregate proof: {}", err))?;
        let inputs = parse_aggregated_public_inputs(&proof)
            .context("failed to parse L0 aggregate public inputs")?;
        validate_l0_public_inputs_against_group(group, &inputs)?;
        self.verifier
            .verify_ref(&proof)
            .context("local L0 aggregate proof verification failed")?;

        let nullifiers = digest_vec_to_arrays(&inputs.nullifiers)?;
        ensure_no_duplicate_nullifiers(&nullifiers)?;

        Ok(ValidatedL0Candidate {
            candidate_id: candidate.candidate_id,
            nullifiers,
            inputs: Some(inputs),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ZkBinsL1ProofGenerator {
    bins_dir: PathBuf,
}

impl ZkBinsL1ProofGenerator {
    pub fn new(bins_dir: PathBuf) -> anyhow::Result<Self> {
        validate_zk_bins_dir(&bins_dir)?;
        Ok(Self { bins_dir })
    }
}

#[async_trait]
impl L1ProofGenerator for ZkBinsL1ProofGenerator {
    async fn prove_l1(
        &self,
        bundle: &ClaimedBundle,
        candidate_proofs: Vec<Vec<u8>>,
    ) -> anyhow::Result<Vec<u8>> {
        let mut aggregator = Layer1Aggregator::new(
            &self.bins_dir,
            BytesDigest::new_unchecked(bundle.aggregator_address),
        )
        .context("failed to load layer-1 aggregation prover")?;
        if candidate_proofs.len() != aggregator.batch_size() {
            bail!(
                "claimed bundle has {} candidate proof(s), but L1 prover expects {}",
                candidate_proofs.len(),
                aggregator.batch_size()
            );
        }

        let layer0_common = aggregator
            .load_common_data(CircuitType::Leaf)
            .context("failed to load layer-0 common data")?;
        for proof_bytes in candidate_proofs {
            let proof =
                ProverProofWithPublicInputs::<F, C, D>::from_bytes(proof_bytes, &layer0_common)
                    .map_err(|err| anyhow!("failed to deserialize claimed L0 proof: {}", err))?;
            aggregator
                .push_proof(proof)
                .context("failed to enqueue L0 proof for layer-1 aggregation")?;
        }

        let proof = aggregator.aggregate().context("layer-1 proving failed")?;
        aggregator
            .verify(proof.clone())
            .context("local layer-1 proof verification failed")?;
        Ok(proof.to_bytes())
    }

    fn verify_l1(&self, proof_bytes: &[u8]) -> anyhow::Result<()> {
        let aggregator = Layer1Aggregator::new(&self.bins_dir, BytesDigest::new_unchecked([0; 32]))
            .context("failed to load layer-1 verifier")?;
        let layer1_common = aggregator
            .load_common_data(CircuitType::Root)
            .context("failed to load layer-1 common data")?;
        let proof = ProverProofWithPublicInputs::<F, C, D>::from_bytes(
            proof_bytes.to_vec(),
            &layer1_common,
        )
        .map_err(|err| anyhow!("failed to deserialize L1 aggregate proof: {}", err))?;
        aggregator
            .verify(proof)
            .context("local layer-1 proof verification failed")
    }
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

    pub async fn next_candidate_batch<C>(
        &self,
        client: &C,
    ) -> anyhow::Result<Option<CandidateBatch>>
    where
        C: AggregationChainClient,
    {
        let current_block = client.current_block().await?;
        let mut batches = Vec::new();

        for group in client.pending_groups().await? {
            let required = group.num_layer0_proofs as usize;
            if required == 0 {
                continue;
            }

            let mut candidates = client.pending_candidates(&group).await?;
            candidates.retain(|candidate| {
                candidate.group_key == group && candidate.expires_at > current_block
            });
            candidates.sort_by_key(|candidate| candidate.submitted_at);
            if candidates.len() < required {
                continue;
            }

            let selected = candidates.into_iter().take(required).collect::<Vec<_>>();
            let total_reward = selected
                .iter()
                .map(L0CandidateSummary::reward_for_selection)
                .sum::<u128>();
            if total_reward < self.config.min_aggregation_reward {
                continue;
            }
            let estimated_proving_cost = selected
                .iter()
                .map(L0CandidateSummary::proving_cost_for_selection)
                .sum::<u64>()
                .max(1);
            let oldest_submitted_at = selected
                .iter()
                .map(|candidate| candidate.submitted_at)
                .min()
                .unwrap_or(u32::MAX);

            batches.push(CandidateBatch {
                group,
                candidates: selected,
                total_reward,
                estimated_proving_cost,
                oldest_submitted_at,
            });
        }

        Ok(select_batch(&batches, self.config.claim_strategy).cloned())
    }

    pub async fn fetch_and_validate_candidate_batch<C, V>(
        &self,
        client: &C,
        validator: &V,
        batch: &CandidateBatch,
    ) -> anyhow::Result<Vec<Vec<u8>>>
    where
        C: AggregationChainClient,
        V: L0ProofValidator,
    {
        let mut proof_bytes = Vec::with_capacity(batch.candidates.len());
        let mut validated = Vec::with_capacity(batch.candidates.len());

        for candidate in &batch.candidates {
            let bytes = client.fetch_candidate_proof(candidate.candidate_id).await?;
            let validated_candidate =
                validator.validate_candidate(&batch.group, candidate, &bytes)?;
            proof_bytes.push(bytes);
            validated.push(validated_candidate);
        }

        let nullifiers = validated
            .iter()
            .flat_map(|candidate| candidate.nullifiers.iter().copied())
            .collect::<Vec<_>>();
        ensure_no_duplicate_nullifiers(&nullifiers)?;

        Ok(proof_bytes)
    }

    pub async fn run_once<C, V, P>(
        &self,
        client: &C,
        validator: &V,
        prover: &P,
    ) -> anyhow::Result<AggregationRunOutcome>
    where
        C: AggregationChainClient,
        V: L0ProofValidator,
        P: L1ProofGenerator,
    {
        let Some(batch) = self.next_candidate_batch(client).await? else {
            return Ok(AggregationRunOutcome::NoOpportunity);
        };

        let _validated_proofs = self
            .fetch_and_validate_candidate_batch(client, validator, &batch)
            .await?;

        if self.config.dry_run {
            return Ok(AggregationRunOutcome::DryRunValidated {
                group: batch.group,
                candidate_count: batch.candidates.len(),
                total_reward: batch.total_reward,
            });
        }

        client.register_aggregator_if_needed(&self.config).await?;
        let claimed = client
            .claim_bundle(batch.group.clone(), self.config.miner_bond)
            .await?;
        let claimed = client.fetch_bundle(claimed.bundle_id).await?;

        let selected_by_id = batch
            .candidates
            .iter()
            .map(|candidate| (candidate.candidate_id, candidate))
            .collect::<BTreeMap<_, _>>();
        let mut claimed_proofs = Vec::with_capacity(claimed.ordered_candidate_ids.len());
        let mut claimed_validated = Vec::with_capacity(claimed.ordered_candidate_ids.len());
        for candidate_id in &claimed.ordered_candidate_ids {
            let candidate = selected_by_id.get(candidate_id).ok_or_else(|| {
                anyhow!(
                    "claimed bundle contains candidate {:02x?} that was not selected pre-claim",
                    candidate_id
                )
            })?;
            let bytes = client.fetch_candidate_proof(*candidate_id).await?;
            let validated_candidate =
                validator.validate_candidate(&claimed.group_key, candidate, &bytes)?;
            claimed_proofs.push(bytes);
            claimed_validated.push(validated_candidate);
        }
        let claimed_nullifiers = claimed_validated
            .iter()
            .flat_map(|candidate| candidate.nullifiers.iter().copied())
            .collect::<Vec<_>>();
        ensure_no_duplicate_nullifiers(&claimed_nullifiers)?;

        let l1_proof = prover.prove_l1(&claimed, claimed_proofs).await?;
        prover
            .verify_l1(&l1_proof)
            .context("local L1 aggregate proof verification failed before submission")?;
        let proof_len = l1_proof.len();
        client
            .submit_l1_aggregate(claimed.bundle_id, l1_proof)
            .await?;
        let _observed = client.fetch_bundle(claimed.bundle_id).await?;

        Ok(AggregationRunOutcome::Submitted {
            bundle_id: claimed.bundle_id,
            proof_len,
        })
    }
}

fn select_batch<'a>(
    batches: &'a [CandidateBatch],
    strategy: ClaimStrategy,
) -> Option<&'a CandidateBatch> {
    match strategy {
        ClaimStrategy::Oldest => batches.iter().min_by_key(|batch| batch.oldest_submitted_at),
        ClaimStrategy::RewardDensity => batches.iter().max_by(|left, right| {
            compare_reward_density(left, right)
                .then_with(|| right.oldest_submitted_at.cmp(&left.oldest_submitted_at))
        }),
    }
}

fn compare_reward_density(left: &CandidateBatch, right: &CandidateBatch) -> Ordering {
    let left_score = left
        .total_reward
        .saturating_mul(right.estimated_proving_cost as u128);
    let right_score = right
        .total_reward
        .saturating_mul(left.estimated_proving_cost as u128);
    left_score.cmp(&right_score)
}

pub fn validate_l0_public_inputs_against_group(
    group: &BundleGroupKey,
    inputs: &AggregatedPublicCircuitInputs,
) -> anyhow::Result<()> {
    if inputs.asset_id != group.asset_id {
        bail!("candidate asset_id does not match bundle group");
    }
    if inputs.volume_fee_bps != group.volume_fee_bps {
        bail!("candidate volume_fee_bps does not match bundle group");
    }
    if digest_to_array(&inputs.block_data.block_hash)? != group.block_hash {
        bail!("candidate block_hash does not match bundle group");
    }
    if inputs.block_data.block_number != group.block_number {
        bail!("candidate block_number does not match bundle group");
    }
    Ok(())
}

pub fn ensure_no_duplicate_nullifiers(nullifiers: &[[u8; 32]]) -> anyhow::Result<()> {
    let mut seen = BTreeSet::new();
    for nullifier in nullifiers {
        if !seen.insert(*nullifier) {
            bail!("duplicate nullifier in candidate batch");
        }
    }
    Ok(())
}

fn digest_to_array(digest: &BytesDigest) -> anyhow::Result<[u8; 32]> {
    digest
        .as_ref()
        .try_into()
        .map_err(|_| anyhow!("digest has invalid length"))
}

fn digest_vec_to_arrays(digests: &[BytesDigest]) -> anyhow::Result<Vec<[u8; 32]>> {
    digests.iter().map(digest_to_array).collect()
}

#[allow(dead_code)]
fn exits_from_public_inputs(inputs: &[PublicInputsByAccount]) -> Vec<ExitSlotSummary> {
    inputs
        .iter()
        .filter_map(|exit| {
            Some(ExitSlotSummary {
                summed_output_amount: exit.summed_output_amount,
                exit_account: digest_to_array(&exit.exit_account).ok()?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::BTreeMap,
        fs,
        sync::{Arc, Mutex},
    };

    fn write_required_artifacts(dir: &Path) {
        for file in REQUIRED_ZK_ARTIFACTS {
            fs::write(dir.join(file), b"fixture").unwrap();
        }
    }

    fn write_secure_keystore(dir: &Path) -> PathBuf {
        let keystore = dir.join("aggregation-keystore");
        fs::create_dir_all(&keystore).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&keystore, fs::Permissions::from_mode(0o700)).unwrap();
        }
        keystore
    }

    fn test_config(dir: &Path, dry_run: bool, strategy: ClaimStrategy) -> ZkAggregationConfig {
        ZkAggregationConfig {
            node_rpc: "ws://127.0.0.1:9944".to_string(),
            aggregation_account: Some("aggregator".to_string()),
            aggregation_keystore: (!dry_run).then(|| write_secure_keystore(dir)),
            zk_bins_dir: Some(dir.to_path_buf()),
            workers: 1,
            max_active_jobs: 1,
            min_aggregation_reward: 0,
            miner_bond: if dry_run { 0 } else { 50 },
            claim_strategy: strategy,
            dry_run,
        }
    }

    fn group(id: u8, submitted_block: u32) -> BundleGroupKey {
        let mut circuit_id = [0u8; 32];
        circuit_id[0] = id;
        let mut block_hash = [0u8; 32];
        block_hash[0] = id;
        BundleGroupKey {
            circuit_id,
            public_input_layout_version: 1,
            num_leaf_proofs: 16,
            num_layer0_proofs: 1,
            asset_id: 0,
            volume_fee_bps: 10,
            block_hash,
            block_number: submitted_block,
        }
    }

    fn candidate(
        id: u8,
        group: BundleGroupKey,
        submitted_at: u32,
        reward: u128,
    ) -> L0CandidateSummary {
        let mut candidate_id = [0u8; 32];
        candidate_id[0] = id;
        let mut nullifier = [0u8; 32];
        nullifier[0] = id;
        L0CandidateSummary {
            candidate_id,
            group_key: group,
            submitted_at,
            expires_at: 100,
            aggregation_tip: reward,
            estimated_reward: reward,
            estimated_proving_cost: 1,
            nullifiers: vec![nullifier],
            exit_summary: Vec::new(),
        }
    }

    #[derive(Default)]
    struct MockState {
        current_block: u32,
        groups: Vec<BundleGroupKey>,
        candidates: BTreeMap<BundleGroupKey, Vec<L0CandidateSummary>>,
        proofs: BTreeMap<[u8; 32], Vec<u8>>,
        post_claim_proofs: BTreeMap<[u8; 32], Vec<u8>>,
        claimed_bundle: Option<ClaimedBundle>,
        registered_count: usize,
        claim_count: usize,
        submit_count: usize,
        submitted_proof: Option<Vec<u8>>,
    }

    #[derive(Clone, Default)]
    struct MockChainClient {
        state: Arc<Mutex<MockState>>,
    }

    impl MockChainClient {
        fn with_group(group: BundleGroupKey, candidates: Vec<L0CandidateSummary>) -> Self {
            let client = Self::default();
            {
                let mut state = client.state.lock().unwrap();
                state.current_block = 1;
                state.groups.push(group.clone());
                for candidate in &candidates {
                    state
                        .proofs
                        .insert(candidate.candidate_id, vec![candidate.candidate_id[0]]);
                }
                state.candidates.insert(group, candidates);
            }
            client
        }

        fn counters(&self) -> (usize, usize, usize) {
            let state = self.state.lock().unwrap();
            (
                state.registered_count,
                state.claim_count,
                state.submit_count,
            )
        }
    }

    #[async_trait]
    impl AggregationChainClient for MockChainClient {
        async fn current_block(&self) -> anyhow::Result<u32> {
            Ok(self.state.lock().unwrap().current_block)
        }

        async fn pending_groups(&self) -> anyhow::Result<Vec<BundleGroupKey>> {
            Ok(self.state.lock().unwrap().groups.clone())
        }

        async fn pending_candidates(
            &self,
            group: &BundleGroupKey,
        ) -> anyhow::Result<Vec<L0CandidateSummary>> {
            Ok(self
                .state
                .lock()
                .unwrap()
                .candidates
                .get(group)
                .cloned()
                .unwrap_or_default())
        }

        async fn fetch_candidate_proof(&self, candidate_id: [u8; 32]) -> anyhow::Result<Vec<u8>> {
            let state = self.state.lock().unwrap();
            if state.claim_count > 0 {
                if let Some(proof) = state.post_claim_proofs.get(&candidate_id) {
                    return Ok(proof.clone());
                }
            }
            state
                .proofs
                .get(&candidate_id)
                .cloned()
                .ok_or_else(|| anyhow!("missing candidate proof"))
        }

        async fn register_aggregator_if_needed(
            &self,
            _config: &ZkAggregationConfig,
        ) -> anyhow::Result<()> {
            self.state.lock().unwrap().registered_count += 1;
            Ok(())
        }

        async fn claim_bundle(
            &self,
            group: BundleGroupKey,
            _miner_bond: u128,
        ) -> anyhow::Result<ClaimedBundle> {
            let candidates = self
                .state
                .lock()
                .unwrap()
                .candidates
                .get(&group)
                .cloned()
                .unwrap_or_default();
            let mut bundle_id = [0u8; 32];
            bundle_id[0] = 9;
            let bundle = ClaimedBundle {
                bundle_id,
                group_key: group,
                ordered_candidate_ids: candidates
                    .iter()
                    .map(|candidate| candidate.candidate_id)
                    .collect(),
                aggregator_address: [2u8; 32],
                deadline_block: 50,
            };
            let mut state = self.state.lock().unwrap();
            state.claim_count += 1;
            state.claimed_bundle = Some(bundle.clone());
            Ok(bundle)
        }

        async fn fetch_bundle(&self, _bundle_id: [u8; 32]) -> anyhow::Result<ClaimedBundle> {
            self.state
                .lock()
                .unwrap()
                .claimed_bundle
                .clone()
                .ok_or_else(|| anyhow!("missing claimed bundle"))
        }

        async fn submit_l1_aggregate(
            &self,
            _bundle_id: [u8; 32],
            proof_bytes: Vec<u8>,
        ) -> anyhow::Result<()> {
            let mut state = self.state.lock().unwrap();
            state.submit_count += 1;
            state.submitted_proof = Some(proof_bytes);
            Ok(())
        }
    }

    struct SummaryProofValidator;

    impl L0ProofValidator for SummaryProofValidator {
        fn validate_candidate(
            &self,
            group: &BundleGroupKey,
            candidate: &L0CandidateSummary,
            proof_bytes: &[u8],
        ) -> anyhow::Result<ValidatedL0Candidate> {
            if candidate.group_key != *group {
                bail!("candidate group does not match selected bundle group");
            }
            if proof_bytes.first().copied() != Some(candidate.candidate_id[0]) {
                bail!("candidate proof bytes do not match candidate id");
            }
            ensure_no_duplicate_nullifiers(&candidate.nullifiers)?;
            Ok(ValidatedL0Candidate {
                candidate_id: candidate.candidate_id,
                nullifiers: candidate.nullifiers.clone(),
                inputs: None,
            })
        }
    }

    struct StaticProofGenerator;

    #[async_trait]
    impl L1ProofGenerator for StaticProofGenerator {
        async fn prove_l1(
            &self,
            _bundle: &ClaimedBundle,
            _candidate_proofs: Vec<Vec<u8>>,
        ) -> anyhow::Result<Vec<u8>> {
            Ok(vec![1, 2, 3])
        }

        fn verify_l1(&self, proof_bytes: &[u8]) -> anyhow::Result<()> {
            if proof_bytes == [1, 2, 3] {
                Ok(())
            } else {
                bail!("mock L1 proof failed local verification")
            }
        }
    }

    struct UnverifiedProofGenerator;

    #[async_trait]
    impl L1ProofGenerator for UnverifiedProofGenerator {
        async fn prove_l1(
            &self,
            _bundle: &ClaimedBundle,
            _candidate_proofs: Vec<Vec<u8>>,
        ) -> anyhow::Result<Vec<u8>> {
            Ok(vec![9, 9, 9])
        }

        fn verify_l1(&self, _proof_bytes: &[u8]) -> anyhow::Result<()> {
            bail!("mock L1 proof failed local verification")
        }
    }

    #[test]
    fn config_validation_rejects_missing_zk_bins() {
        let config = ZkAggregationConfig {
            node_rpc: "ws://127.0.0.1:9944".to_string(),
            aggregation_account: None,
            aggregation_keystore: None,
            zk_bins_dir: None,
            workers: 1,
            max_active_jobs: 1,
            min_aggregation_reward: 0,
            miner_bond: 0,
            claim_strategy: ClaimStrategy::Oldest,
            dry_run: true,
        };

        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("--zk-bins-dir is required"));
    }

    #[test]
    fn config_validation_requires_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let err = test_config(dir.path(), true, ClaimStrategy::Oldest)
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("missing required artifact"));
    }

    #[test]
    fn dry_run_config_allows_missing_signing_credentials() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let config = ZkAggregationConfig {
            node_rpc: "ws://127.0.0.1:9944".to_string(),
            aggregation_account: None,
            aggregation_keystore: None,
            zk_bins_dir: Some(dir.path().to_path_buf()),
            workers: 1,
            max_active_jobs: 1,
            min_aggregation_reward: 0,
            miner_bond: 0,
            claim_strategy: ClaimStrategy::Oldest,
            dry_run: true,
        };

        config.validate().unwrap();
    }

    #[test]
    fn non_dry_run_config_requires_account_and_keystore() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let mut config = test_config(dir.path(), false, ClaimStrategy::Oldest);
        config.aggregation_account = None;
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("--aggregation-account is required"));

        let mut config = test_config(dir.path(), false, ClaimStrategy::Oldest);
        config.aggregation_keystore = None;
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("--aggregation-keystore is required"));
    }

    #[test]
    fn rpc_url_redaction_hides_credentials() {
        assert_eq!(
            redact_rpc_url("wss://user:token@example.com/path"),
            "wss://***:***@example.com/path"
        );
        assert_eq!(
            redact_rpc_url("https://token@example.com"),
            "https://***@example.com"
        );
        assert_eq!(redact_rpc_url("ws://127.0.0.1:9944"), "ws://127.0.0.1:9944");
    }

    #[test]
    fn config_debug_redacts_rpc_credentials() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let mut config = test_config(dir.path(), false, ClaimStrategy::Oldest);
        config.node_rpc = "wss://user:token@example.com/path".to_string();

        let debug = format!("{config:?}");
        assert!(debug.contains("wss://***:***@example.com/path"));
        assert!(!debug.contains("user:token"));
    }

    #[test]
    fn worker_pool_does_not_cancel_on_pow_job() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let pool = AggregationWorkerPool::new(test_config(dir.path(), true, ClaimStrategy::Oldest))
            .unwrap();
        assert!(!pool.should_cancel_for_new_pow_job());
    }

    #[tokio::test]
    async fn dry_run_does_not_submit_claim() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let group = group(1, 1);
        let client =
            MockChainClient::with_group(group.clone(), vec![candidate(1, group.clone(), 3, 10)]);
        let pool = AggregationWorkerPool::new(test_config(dir.path(), true, ClaimStrategy::Oldest))
            .unwrap();

        let outcome = pool
            .run_once(&client, &SummaryProofValidator, &StaticProofGenerator)
            .await
            .unwrap();

        assert!(matches!(
            outcome,
            AggregationRunOutcome::DryRunValidated {
                candidate_count: 1,
                ..
            }
        ));
        assert_eq!(client.counters(), (0, 0, 0));
    }

    #[tokio::test]
    async fn claim_strategy_selects_oldest_group() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let old_group = group(1, 1);
        let new_group = group(2, 2);
        let client = MockChainClient::default();
        {
            let mut state = client.state.lock().unwrap();
            state.current_block = 1;
            state.groups = vec![new_group.clone(), old_group.clone()];
            state.candidates.insert(
                old_group.clone(),
                vec![candidate(1, old_group.clone(), 2, 10)],
            );
            state.candidates.insert(
                new_group.clone(),
                vec![candidate(2, new_group.clone(), 20, 1000)],
            );
            state.proofs.insert([1u8; 32], vec![1]);
            state.proofs.insert([2u8; 32], vec![2]);
        }
        let pool = AggregationWorkerPool::new(test_config(dir.path(), true, ClaimStrategy::Oldest))
            .unwrap();

        let batch = pool.next_candidate_batch(&client).await.unwrap().unwrap();
        assert_eq!(batch.group, old_group);
    }

    #[tokio::test]
    async fn reward_density_strategy_selects_best_reward_per_cost() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let dense_group = group(1, 1);
        let high_reward_group = group(2, 2);
        let mut dense_candidate = candidate(1, dense_group.clone(), 10, 50);
        dense_candidate.estimated_proving_cost = 1;
        let mut costly_candidate = candidate(2, high_reward_group.clone(), 2, 100);
        costly_candidate.estimated_proving_cost = 10;
        let client = MockChainClient::default();
        {
            let mut state = client.state.lock().unwrap();
            state.current_block = 1;
            state.groups = vec![high_reward_group.clone(), dense_group.clone()];
            state
                .candidates
                .insert(dense_group.clone(), vec![dense_candidate]);
            state
                .candidates
                .insert(high_reward_group.clone(), vec![costly_candidate]);
        }
        let pool =
            AggregationWorkerPool::new(test_config(dir.path(), true, ClaimStrategy::RewardDensity))
                .unwrap();

        let batch = pool.next_candidate_batch(&client).await.unwrap().unwrap();
        assert_eq!(batch.group, dense_group);
    }

    #[test]
    fn local_validation_rejects_incompatible_candidates() {
        let group = group(1, 1);
        let inputs = AggregatedPublicCircuitInputs {
            num_unique_exits: 0,
            asset_id: 99,
            volume_fee_bps: 10,
            block_data: qp_wormhole_verifier::BlockData {
                block_hash: BytesDigest::new_unchecked(group.block_hash),
                block_number: group.block_number,
            },
            account_data: Vec::new(),
            nullifiers: Vec::new(),
        };

        let err = validate_l0_public_inputs_against_group(&group, &inputs).unwrap_err();
        assert!(err.to_string().contains("asset_id"));
    }

    #[tokio::test]
    async fn local_validation_rejects_duplicate_nullifiers() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let group = group(1, 1);
        let mut candidate = candidate(1, group.clone(), 3, 10);
        candidate.nullifiers = vec![[7u8; 32], [7u8; 32]];
        let client = MockChainClient::with_group(group.clone(), vec![candidate]);
        let pool = AggregationWorkerPool::new(test_config(dir.path(), true, ClaimStrategy::Oldest))
            .unwrap();

        let err = pool
            .run_once(&client, &SummaryProofValidator, &StaticProofGenerator)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("duplicate nullifier"));
    }

    #[tokio::test]
    async fn mocked_worker_processes_claimed_bundle_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let group = group(1, 1);
        let client =
            MockChainClient::with_group(group.clone(), vec![candidate(1, group.clone(), 3, 10)]);
        let pool =
            AggregationWorkerPool::new(test_config(dir.path(), false, ClaimStrategy::Oldest))
                .unwrap();

        let outcome = pool
            .run_once(&client, &SummaryProofValidator, &StaticProofGenerator)
            .await
            .unwrap();

        assert!(matches!(
            outcome,
            AggregationRunOutcome::Submitted { proof_len: 3, .. }
        ));
        assert_eq!(client.counters(), (1, 1, 1));
    }

    #[tokio::test]
    async fn worker_revalidates_exact_claimed_proofs_after_claim() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let group = group(1, 1);
        let candidate = candidate(1, group.clone(), 3, 10);
        let candidate_id = candidate.candidate_id;
        let client = MockChainClient::with_group(group.clone(), vec![candidate]);
        {
            let mut state = client.state.lock().unwrap();
            state.post_claim_proofs.insert(candidate_id, vec![0]);
        }
        let pool =
            AggregationWorkerPool::new(test_config(dir.path(), false, ClaimStrategy::Oldest))
                .unwrap();

        let err = pool
            .run_once(&client, &SummaryProofValidator, &StaticProofGenerator)
            .await
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("candidate proof bytes do not match"));
        assert_eq!(client.counters(), (1, 1, 0));
    }

    #[tokio::test]
    async fn worker_does_not_submit_l1_proof_that_fails_local_verification() {
        let dir = tempfile::tempdir().unwrap();
        write_required_artifacts(dir.path());
        let group = group(1, 1);
        let client =
            MockChainClient::with_group(group.clone(), vec![candidate(1, group.clone(), 3, 10)]);
        let pool =
            AggregationWorkerPool::new(test_config(dir.path(), false, ClaimStrategy::Oldest))
                .unwrap();

        let err = pool
            .run_once(&client, &SummaryProofValidator, &UnverifiedProofGenerator)
            .await
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("local L1 aggregate proof verification failed"));
        assert_eq!(client.counters(), (1, 1, 0));
    }

    #[test]
    fn zk_aggregation_prove_generates_l1_proof_from_fixture_when_configured() {
        let Ok(bins_dir) = std::env::var("ZK_AGGREGATION_TEST_BINS_DIR") else {
            eprintln!("skipping proving test: ZK_AGGREGATION_TEST_BINS_DIR is not set");
            return;
        };
        let Ok(l0_proof_path) = std::env::var("ZK_AGGREGATION_TEST_L0_PROOF") else {
            eprintln!("skipping proving test: ZK_AGGREGATION_TEST_L0_PROOF is not set");
            return;
        };

        let proof_hex = fs::read_to_string(l0_proof_path).unwrap();
        let proof_bytes = hex::decode(proof_hex.trim()).unwrap();
        let group = group(1, 1);
        let bundle = ClaimedBundle {
            bundle_id: [9u8; 32],
            group_key: group,
            ordered_candidate_ids: vec![[1u8; 32]],
            aggregator_address: {
                let mut address = [0u8; 32];
                address[..8].copy_from_slice(&2u64.to_le_bytes());
                address
            },
            deadline_block: 50,
        };
        let prover = ZkBinsL1ProofGenerator::new(PathBuf::from(bins_dir)).unwrap();
        let proof = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(prover.prove_l1(&bundle, vec![proof_bytes]))
            .unwrap();

        assert!(!proof.is_empty());
    }
}
