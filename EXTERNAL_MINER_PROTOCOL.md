# External Miner Protocol

This file is no longer the source of truth.

The external-miner QUIC protocol (including authenticated `Ready { token }`,
TLS cert pinning, ALPN `quantus-miner/2`, and the 1 KB frame limit) is defined
by:

1. The published [`quantus-miner-api`](https://crates.io/crates/quantus-miner-api) crate
2. The Quantus node’s [`MINING.md`](https://github.com/Quantus-Network/chain/blob/main/MINING.md)

Use those for any new miner or pool implementation.
