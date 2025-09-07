/*!
Deprecated root binary entrypoint.

This repository has been restructured into a Cargo workspace.

Run the external miner using the new CLI binary:

  - cargo run -p miner-cli -- [args...]
  - cargo build -p miner-cli --release

Examples:
  - cargo run -p miner-cli -- --port 9833
  - cargo run -p miner-cli -- --port 9833 --metrics-port 9900
  - cargo run -p miner-cli -- --num-cores 4

See crates/miner-cli for the active entrypoint and crates/miner-service for the service layer.
*/

fn main() {
    eprintln!(
        "\n[DEPRECATED] This entrypoint has moved.\n\
         Use the new CLI binary in the workspace instead:\n\
         \n\
         - cargo run -p miner-cli -- [args...]\n\
         - cargo build -p miner-cli --release\n\
         \n\
         Examples:\n\
         - cargo run -p miner-cli -- --port 9833\n\
         - cargo run -p miner-cli -- --port 9833 --metrics-port 9900\n\
         - cargo run -p miner-cli -- --num-cores 4\n"
    );
    std::process::exit(1);
}
