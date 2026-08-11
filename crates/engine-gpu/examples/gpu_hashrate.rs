//! Fast GPU mining benchmark + correctness harness.
//!
//! Loads the mining WGSL from disk at runtime, so shader edits need no rebuild.
//!
//! Usage:
//!   cargo run -p engine-gpu --release --example gpu_hashrate -- [options]
//! Options:
//!   --wgsl <path>       shader path (default: src/mining.wgsl in this crate)
//!   --batches <n>       number of timed batches (default: 5)
//!   --batch-size <n>    nonces per batch (default: 10000000)
//!   --workgroups <n>    workgroups to dispatch (default: mirror engine heuristic)
//!   --skip-check        skip the CPU-vs-GPU correctness check

use engine_gpu::precompute_header_state;
use pow_core::{hash_from_nonce, JobContext};
use primitive_types::U512;
use std::time::Instant;

struct Args {
    wgsl_path: String,
    batches: u32,
    batch_size: u32,
    workgroups: Option<u32>,
    skip_check: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        wgsl_path: format!("{}/src/mining.wgsl", env!("CARGO_MANIFEST_DIR")),
        batches: 5,
        batch_size: 10_000_000,
        workgroups: None,
        skip_check: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--wgsl" => args.wgsl_path = it.next().expect("--wgsl needs a value"),
            "--batches" => {
                args.batches = it.next().expect("--batches needs a value").parse().unwrap()
            }
            "--batch-size" => {
                args.batch_size = it
                    .next()
                    .expect("--batch-size needs a value")
                    .parse()
                    .unwrap()
            }
            "--workgroups" => {
                args.workgroups = Some(
                    it.next()
                        .expect("--workgroups needs a value")
                        .parse()
                        .unwrap(),
                )
            }
            "--skip-check" => args.skip_check = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    args
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .parse_default_env()
        .init();
    let args = parse_args();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(run(args));
}

fn u512_to_u32s_le(v: U512) -> [u32; 16] {
    let bytes = v.to_little_endian();
    let mut out = [0u32; 16];
    for i in 0..16 {
        out[i] = u32::from_le_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap());
    }
    out
}

const THREADS_PER_WORKGROUP: u32 = 256;
const RESULTS_SIZE: u64 = (1 + 16 + 16) * 4;

struct Bench {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    results_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    header_buffer: wgpu::Buffer,
    start_nonce_buffer: wgpu::Buffer,
    target_buffer: wgpu::Buffer,
    dispatch_config_buffer: wgpu::Buffer,
    max_threads: u32,
}

impl Bench {
    fn set_header(&self, header: &[u8; 32]) {
        let header_state = precompute_header_state(header);
        self.queue
            .write_buffer(&self.header_buffer, 0, bytemuck::cast_slice(&header_state));
    }

    fn set_target(&self, target: U512) {
        self.queue.write_buffer(
            &self.target_buffer,
            0,
            bytemuck::cast_slice(&u512_to_u32s_le(target)),
        );
    }

    /// Run one batch exactly like the engine's run_single_batch.
    /// Returns Some((nonce, hash)) when a solution was found.
    async fn run_batch(&self, start_nonce: U512, batch_size: u32) -> Option<(U512, U512)> {
        let logical_threads = (batch_size as u64).min(self.max_threads as u64).max(1);
        let num_workgroups = ((logical_threads as u32).div_ceil(THREADS_PER_WORKGROUP)).max(1);
        let total_threads = num_workgroups * THREADS_PER_WORKGROUP;
        let nonces_per_thread = ((batch_size as u64).div_ceil(total_threads as u64)).max(1) as u32;

        let dispatch_config = [total_threads, nonces_per_thread, batch_size];
        self.queue.write_buffer(
            &self.dispatch_config_buffer,
            0,
            bytemuck::cast_slice(&dispatch_config),
        );
        self.queue.write_buffer(
            &self.start_nonce_buffer,
            0,
            bytemuck::cast_slice(&u512_to_u32s_le(start_nonce)),
        );
        self.queue
            .write_buffer(&self.results_buffer, 0, &[0u8; RESULTS_SIZE as usize]);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = self.encoder_begin_compute_pass(&mut encoder);
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.results_buffer,
            0,
            &self.staging_buffer,
            0,
            RESULTS_SIZE,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.await.unwrap().unwrap();

        let result = {
            let data = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&data);
            if words[0] != 0 {
                let nonce =
                    U512::from_little_endian(bytemuck::cast_slice::<u32, u8>(&words[1..17]));
                let hash =
                    U512::from_little_endian(bytemuck::cast_slice::<u32, u8>(&words[17..33]));
                Some((nonce, hash))
            } else {
                None
            }
        };
        let _ = slice;
        self.staging_buffer.unmap();
        result
    }

    fn encoder_begin_compute_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> wgpu::ComputePass<'a> {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        })
    }
}

async fn run(args: Args) {
    let shader_source = std::fs::read_to_string(&args.wgsl_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", args.wgsl_path));

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("no adapter");
    let info = adapter.get_info();
    println!("GPU: {} ({:?})", info.name, info.device_type);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("bench"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mining"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mining"),
        layout: None,
        module: &shader,
        entry_point: Some("mining_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Mirror the engine's dispatch heuristic (Apple fallback tier: divisor 4, min 160).
    let max_workgroups = device
        .limits()
        .max_compute_workgroups_per_dimension
        .min(65535);
    let heuristic = (max_workgroups / 4).max(160);
    let num_workgroups = args
        .workgroups
        .unwrap_or(heuristic)
        .min(max_workgroups)
        .max(1);
    let max_threads = num_workgroups * THREADS_PER_WORKGROUP;
    println!("Dispatch: up to {num_workgroups} workgroups x {THREADS_PER_WORKGROUP} threads = {max_threads} threads");

    let mk = |size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let results_buffer = mk(
        RESULTS_SIZE,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    // Header buffer oversized (128B) so future precomputed-state layouts fit.
    let header_buffer = mk(
        128,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let start_nonce_buffer = mk(
        64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let target_buffer = mk(
        64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let dispatch_config_buffer = mk(
        12,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let staging_buffer = mk(
        RESULTS_SIZE,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: results_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: header_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: start_nonce_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: target_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dispatch_config_buffer.as_entire_binding(),
            },
        ],
    });

    let bench = Bench {
        device,
        queue,
        pipeline,
        bind_group,
        results_buffer,
        staging_buffer,
        header_buffer,
        start_nonce_buffer,
        target_buffer,
        dispatch_config_buffer,
        max_threads,
    };

    let header = [7u8; 32];
    bench.set_header(&header);

    // Correctness check: difficulty 1 (target = MAX) -> every nonce wins.
    if !args.skip_check {
        println!("Correctness check (difficulty 1, nonce 123456789+)...");
        let easy_ctx = JobContext::new(header, U512::from(1u64));
        bench.set_target(easy_ctx.target);
        let (nonce, hash) = bench
            .run_batch(U512::from(123456789u64), 256)
            .await
            .expect("GPU found no solution at difficulty 1");
        let cpu = hash_from_nonce(&easy_ctx, nonce);
        assert_eq!(cpu, hash, "GPU hash != CPU hash for nonce {nonce}");
        println!("  nonce {nonce}: GPU hash == CPU hash OK");

        // Mid difficulty: solution must satisfy hash < target, CPU re-verified.
        let mid_ctx = JobContext::new(header, U512::from(1_000_000u64));
        bench.set_target(mid_ctx.target);
        match bench.run_batch(U512::from(0u64), 100_000_000).await {
            Some((nonce, hash)) => {
                let cpu = hash_from_nonce(&mid_ctx, nonce);
                assert_eq!(cpu, hash, "mid-diff GPU hash != CPU hash");
                assert!(hash < mid_ctx.target, "mid-diff hash not below target");
                println!("  mid-difficulty nonce {nonce}: valid seal OK");
            }
            None => println!("  mid-difficulty: no solution in range (unlikely)"),
        }
    }

    // Timed batches: high difficulty, no solutions expected.
    let ctx = JobContext::new(header, U512::from(u64::MAX));
    bench.set_target(ctx.target);
    println!(
        "Running {} batches x {} nonces (high difficulty)...",
        args.batches, args.batch_size
    );
    let mut total_hashes = 0u64;
    let wall_start = Instant::now();
    let mut nonce = U512::from(1u64 << 40);
    for b in 0..args.batches {
        let t = Instant::now();
        let found = bench.run_batch(nonce, args.batch_size).await;
        let dt = t.elapsed();
        total_hashes += args.batch_size as u64;
        let rate = args.batch_size as f64 / dt.as_secs_f64() / 1e6;
        println!(
            "  batch {b}: {:.3}s ({:.2} MH/s){}",
            dt.as_secs_f64(),
            rate,
            if found.is_some() { " FOUND" } else { "" }
        );
        assert!(found.is_none(), "unexpected solution at high difficulty");
        nonce = nonce.saturating_add(U512::from(args.batch_size));
    }
    let total_dt = wall_start.elapsed().as_secs_f64();
    println!(
        "TOTAL: {} hashes in {:.3}s = {:.3} MH/s",
        total_hashes,
        total_dt,
        total_hashes as f64 / total_dt / 1e6
    );
}
