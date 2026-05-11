# Delegated Aggregation Miner Guide

The miner delegated aggregation worker watches for compatible L0 Wormhole
candidates, claims a bundle, validates the exact claimed proofs, generates an
L1 aggregate proof, verifies that proof locally, and submits it to the chain.

## Configuration

Dry-run mode is allowed without signing credentials:

```bash
quantus-miner serve \
  --enable-zk-aggregation \
  --dry-run-zk-aggregation \
  --node-rpc ws://127.0.0.1:9944 \
  --zk-bins-dir ../qp-zk-circuits/generated-bins
```

Non-dry-run mode requires an aggregation account and secure keystore path:

```bash
quantus-miner serve \
  --enable-zk-aggregation \
  --node-rpc ws://127.0.0.1:9944 \
  --aggregation-account <SS58_ACCOUNT> \
  --aggregation-keystore /secure/path/aggregation-keystore \
  --zk-bins-dir ../qp-zk-circuits/generated-bins \
  --zk-miner-bond 1000000000000 \
  --min-aggregation-reward 0
```

Do not pass raw signing key material through CLI arguments or environment
variables in non-dry-run mode.

## Worker Flow

1. Select compatible pending L0 candidates.
2. Validate candidate proof bytes before claim.
3. Claim a bundle and lock nullifiers on chain.
4. Fetch the exact claimed bundle.
5. Re-fetch and revalidate the exact claimed candidate proofs.
6. Generate the L1 aggregate proof.
7. Locally verify the L1 aggregate proof.
8. Submit the L1 aggregate to chain.
9. Retry or let the runtime timeout/challenge path clean up failures.

The worker must never submit an L1 proof that fails local verification.

## Operational Notes

- Use `--min-aggregation-reward` to avoid uneconomical bundles.
- Use `--max-active-zk-jobs` conservatively until the E2E restart/timeout tests
  pass on a live devnet.
- RPC URLs with embedded credentials are redacted in debug/log output, but avoid
  credential-bearing URLs where possible.
- The `--zk-bins-dir` directory must contain matching L0 and L1 artifacts for
  the same circuits revision used by the chain runtime.

## Troubleshooting

- Missing artifact error: regenerate `generated-bins`, including `layer1_*`
  artifacts.
- Keystore rejected: ensure the path exists, is a file or directory, and on
  Unix has no group/other permissions.
- Worker skips a batch: check reward threshold, candidate compatibility, and
  active job limits.
- Claim succeeds but submit does not: inspect local proof validation and L1
  verification errors; do not bypass verification.
- Restart mid-bundle: the runtime timeout path must release locks if the miner
  cannot resume and settle.
