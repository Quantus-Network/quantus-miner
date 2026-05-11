# Miner Key Management

Delegated aggregation signing must use a secure signer, keystore, or provider.
Raw private key CLI arguments and raw private key environment variables are not
accepted for non-dry-run delegated aggregation.

## Requirements

- Non-dry-run mode requires `--aggregation-account`.
- Non-dry-run mode requires `--aggregation-keystore`.
- Dry-run mode may omit signing credentials.
- If a keystore path is supplied, it must exist and be a file or directory.
- On Unix, the keystore path must not be readable, writable, or executable by
  group or other users.

Example Unix setup:

```bash
install -d -m 700 /secure/path/aggregation-keystore
quantus-miner serve \
  --enable-zk-aggregation \
  --aggregation-account <SS58_ACCOUNT> \
  --aggregation-keystore /secure/path/aggregation-keystore \
  --node-rpc ws://127.0.0.1:9944 \
  --zk-bins-dir ../qp-zk-circuits/generated-bins \
  --zk-miner-bond 1000000000000
```

## Secret Handling

- Do not commit keystores.
- Do not put signing keys in shell history.
- Do not pass raw signing keys through process args.
- Do not print credential-bearing RPC URLs in support bundles.
- Prefer isolated operator accounts funded only for delegated aggregation bond
  and fee needs.

## Rotation

1. Register or fund the replacement aggregation account.
2. Start a miner with the replacement keystore in dry-run mode.
3. Stop new claims from the old miner.
4. Let old active bundles settle or timeout.
5. Start non-dry-run mode for the replacement account.

## Incident Response

- If a signing key is suspected compromised, stop the miner immediately.
- Let active bundle locks resolve via settlement, timeout, or challenge.
- Rotate to a new aggregation account.
- Review reward and slash events around the incident window.
