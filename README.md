# External Miner Service for Quantus Network

Note: This repository is now a Cargo workspace. Build and run the CLI with (the --num-cores flag remains available as an alias for --cores):
- cargo build -p miner-cli --release
- cargo run -p miner-cli -- --port 9833 [--metrics-port 9900] [--cores N]

This crate provides an external mining service that can be used with a Quantus Network node. It exposes an HTTP API for
managing mining jobs.

## Building

To build the external miner service, navigate to the `miner` directory within the repository and use Cargo:

```bash
cd quantus-miner
cargo build --release
```

This will compile the binary and place it in the `target/release/` directory.

## Configuration

The service can be configured using command-line arguments or environment variables.

| Argument          | Environment Variable | Description                                | Default       |
|-------------------|----------------------|--------------------------------------------|---------------|
| `--port <PORT>`   | `MINER_PORT`         | The port for the HTTP server to listen on. | `9833`        |
| `--workers <N>` | `MINER_WORKERS` | The number of worker threads (logical CPUs) to use for mining. | Auto-detected (leaves ~half available) |

Example:

```bash
# Run on the default port 9833 with all available cores
../target/release/quantus-miner

# Run on a custom port with 4 workers (logical CPUs)
../target/release/quantus-miner --port 8000 --workers 4

# Equivalent using environment variables
export MINER_PORT=8000
export MINER_WORKERS=4
../target/release/quantus-miner
```

## Running

After building the service, you can run it directly from the command line:

```bash
# Run with default settings
RUST_LOG=info ../target/release/quantus-miner

# Run with a specific port and 2 workers
RUST_LOG=info ../target/release/quantus-miner --port 12345 --workers 2

# Run in debug mode
RUST_LOG=info,miner=debug ../target/release/quantus-miner --workers 4

```

The service will start and log messages to the console, indicating the port it's listening on and the number of cores
it's using.

Example output:

```
INFO  external_miner > Starting external miner service...
INFO  external_miner > Using auto-detected workers (leaving headroom): 4
INFO  external_miner > Server starting on 0.0.0.0:9833 
```

## API Specification

The detailed API specification is defined using OpenAPI 3.0 and can be found in the `api/openapi.yaml` file.

This specification details all endpoints, request/response formats, and expected status codes.
You can use tools like [Swagger Editor](https://editor.swagger.io/)
or [Swagger UI](https://swagger.io/tools/swagger-ui/) to view and interact with the API definition.

## A note on workers

The miner previously used a flag named `--num-cores`. To better reflect intent, this has been replaced by `--workers`, which specifies the number of worker threads (logical CPUs). When not provided, the miner auto-detects an effective CPU set (honoring cgroup cpusets when present) and defaults to a value that leaves roughly half of the system resources available to other processes.

## API Endpoints (Summary)

* `POST /mine`: Submits a new mining job.
* `GET /result/{job_id}`: Retrieves the status and result of a specific mining job.
* `POST /cancel/{job_id}`: Cancels an ongoing mining job.

## Implementation and PR review docs

These documents provide reviewers with the authoritative context for changes. Commits and pull requests should link to the relevant prompt/response entry.

- Reviewer index: docs/implementation/readme.md
- Authoring/process guide: agents.md