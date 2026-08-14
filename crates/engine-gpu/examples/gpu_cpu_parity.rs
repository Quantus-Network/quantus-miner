//! GPU/CPU parity fuzzer.
//!
//! Phase 1 (bulk): a GPU kernel resumes the sponge from the per-batch midstate
//! (the real mining datapath) and writes the FULL 512-bit hash for every nonce;
//! every hash is verified against the canonical CPU implementation. Randomized
//! headers and 512-bit range starts per chunk.
//!
//! Phase 2 (seals): randomized mining jobs against the real GpuEngine, verifying
//! every found seal (and hash-count exactness, boundary crossings, carry edges,
//! and CPU-known solutions for false negatives).
//!
//! Usage: gpu_cpu_parity [jobs] [seed] [bulk_hashes]

use engine_cpu::{AtomicBoolCancelCheck, EngineStatus, MinerEngine, Range};
use engine_gpu::GpuEngine;
use pow_core::JobContext;
use primitive_types::U512;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::atomic::AtomicBool;

const BULK_KERNEL: &str = r#"
@group(0) @binding(5) var<storage, read_write> hash_out: array<u32>;

@compute @workgroup_size(256)
fn bulk_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dispatch_config[2]) {
        return;
    }
    var current_nonce: array<u32, 16>;
    let val0 = start_nonce[0];
    let sum0 = val0 + i;
    current_nonce[0] = sum0;
    var carry = select(0u, 1u, sum0 < val0);
    for (var k = 1u; k < 8u; k++) {
        let v = start_nonce[k];
        let s = v + carry;
        current_nonce[k] = s;
        carry = select(0u, 1u, s < v);
    }
    for (var k = 8u; k < 16u; k++) {
        current_nonce[k] = start_nonce[k];
    }

    var state: array<GoldilocksField, 12>;
    for (var k = 0u; k < 12u; k++) {
        state[k] = gf_from_limbs(midstate[2u * k], midstate[2u * k + 1u]);
    }
    for (var k = 0u; k < 8u; k++) {
        state[k] = gf_add(state[k], gf_from_u32(bswap32(current_nonce[7u - k])));
    }
    poseidon2_permute(&state);
    state[0] = gf_add(state[0], gf_one());
    state[1] = gf_add(state[1], gf_one());
    poseidon2_permute(&state);
    let first = field_elements_to_bytes(array<GoldilocksField, 4>(
        state[0], state[1], state[2], state[3]
    ));
    poseidon2_permute(&state);
    let second = field_elements_to_bytes(array<GoldilocksField, 4>(
        state[0], state[1], state[2], state[3]
    ));
    let base = i * 16u;
    for (var k = 0u; k < 8u; k++) {
        hash_out[base + 15u - k] = bswap32(first[k]);
        hash_out[base + k] = bswap32(second[7u - k]);
    }
}
"#;

fn random_u512(rng: &mut ChaCha8Rng, zero_top_byte: bool) -> U512 {
    let mut bytes = [0u8; 64];
    rng.fill_bytes(&mut bytes);
    if zero_top_byte {
        bytes[0] = 0;
    }
    U512::from_big_endian(&bytes)
}

fn u512_to_u32s_le(v: U512) -> [u32; 16] {
    let bytes = v.to_little_endian();
    let mut out = [0u32; 16];
    for i in 0..16 {
        out[i] = u32::from_le_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap());
    }
    out
}

/// Which shader variant a bulk chunk runs on.
#[derive(Clone, Copy)]
enum BulkVariant {
    U32,
    U64,
}

impl BulkVariant {
    fn name(self) -> &'static str {
        match self {
            BulkVariant::U32 => "32-bit",
            BulkVariant::U64 => "native-u64",
        }
    }
}

struct BulkRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_u32: wgpu::ComputePipeline,
    pipeline_u64: Option<wgpu::ComputePipeline>,
    midstate: wgpu::Buffer,
    start_nonce: wgpu::Buffer,
    cfg: wgpu::Buffer,
    hash_out: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl BulkRunner {
    async fn new(chunk: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("no adapter");
        let has_u64 = adapter.features().contains(wgpu::Features::SHADER_INT64);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: if has_u64 {
                    wgpu::Features::SHADER_INT64
                } else {
                    wgpu::Features::empty()
                },
                ..Default::default()
            })
            .await
            .unwrap();
        let mk_pipeline = |base: &str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(format!("{base}\n{BULK_KERNEL}").into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("bulk_hash"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        // Always build the 32-bit fallback; add the native-u64 variant when supported
        // so one run exercises both shader variants.
        let pipeline_u32 = mk_pipeline(include_str!("../src/mining.wgsl"));
        let pipeline_u64 = has_u64.then(|| mk_pipeline(include_str!("../src/mining_u64.wgsl")));
        println!(
            "bulk: 32-bit shader enabled, native-u64 shader {}",
            if pipeline_u64.is_some() {
                "enabled"
            } else {
                "unavailable"
            }
        );
        let mk = |size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        let out_size = chunk as u64 * 64;
        BulkRunner {
            pipeline_u32,
            pipeline_u64,
            midstate: mk(96, U::STORAGE | U::COPY_DST),
            start_nonce: mk(64, U::STORAGE | U::COPY_DST),
            cfg: mk(12, U::STORAGE | U::COPY_DST),
            hash_out: mk(out_size, U::STORAGE | U::COPY_SRC),
            staging: mk(out_size, U::MAP_READ | U::COPY_DST),
            device,
            queue,
        }
    }

    fn has_u64(&self) -> bool {
        self.pipeline_u64.is_some()
    }

    fn run_chunk(
        &self,
        header: [u8; 32],
        start: U512,
        count: u32,
        variant: BulkVariant,
    ) -> Vec<u32> {
        let pipeline = match variant {
            BulkVariant::U32 => &self.pipeline_u32,
            BulkVariant::U64 => self
                .pipeline_u64
                .as_ref()
                .expect("native-u64 variant requested but unsupported"),
        };
        let nonce_be = start.to_big_endian();
        let mid = pow_core::mining_midstate(header, nonce_be[..32].try_into().unwrap());
        let mut mid_u32 = [0u32; 24];
        for (i, f) in mid.iter().enumerate() {
            mid_u32[2 * i] = *f as u32;
            mid_u32[2 * i + 1] = (*f >> 32) as u32;
        }
        self.queue
            .write_buffer(&self.midstate, 0, bytemuck::cast_slice(&mid_u32));
        self.queue.write_buffer(
            &self.start_nonce,
            0,
            bytemuck::cast_slice(&u512_to_u32s_le(start)),
        );
        self.queue
            .write_buffer(&self.cfg, 0, bytemuck::cast_slice(&[count, 1u32, count]));

        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.midstate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.start_nonce.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.cfg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.hash_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.hash_out, 0, &self.staging, 0, count as u64 * 64);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging.slice(..count as u64 * 64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        loop {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            if rx.try_recv().is_ok() {
                break;
            }
        }
        let data = slice.get_mapped_range();
        let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging.unmap();
        out
    }
}

fn run_bulk_phase(rng: &mut ChaCha8Rng, total: u64, seed: u64) {
    let chunk: u32 = 65_536;
    let runner = pollster_block(BulkRunner::new(chunk));
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut verified = 0u64;
    let mut verified_u64 = 0u64;
    let mut chunk_idx = 0u64;

    while verified < total {
        // Clamp in u64 first: `(total - verified) as u32` truncates and can wrap to 0
        let count = (total - verified).min(u64::from(chunk)) as u32;
        let mut header = [0u8; 32];
        rng.fill_bytes(&mut header);
        let mut start_bytes = [0u8; 64];
        rng.fill_bytes(&mut start_bytes);
        // Keep headroom so start + count cannot carry into the high half
        start_bytes[0] = 0;
        start_bytes[32] = 0;
        let start = U512::from_big_endian(&start_bytes);

        // Alternate shader variants per chunk so both are exercised in one run
        let variant = if runner.has_u64() && !chunk_idx.is_multiple_of(2) {
            BulkVariant::U64
        } else {
            BulkVariant::U32
        };
        let gpu_hashes = runner.run_chunk(header, start, count, variant);

        std::thread::scope(|s| {
            let gpu = &gpu_hashes;
            let per = (count as usize).div_ceil(threads);
            for t in 0..threads {
                let lo = t * per;
                let hi = ((t + 1) * per).min(count as usize);
                s.spawn(move || {
                    for i in lo..hi {
                        let nonce = start + U512::from(i as u64);
                        let expected = pow_core::get_nonce_hash(header, nonce.to_big_endian());
                        let words = &gpu[i * 16..(i + 1) * 16];
                        let got = U512::from_little_endian(bytemuck::cast_slice(words));
                        assert_eq!(
                            got, expected,
                            "seed {seed} bulk chunk {chunk_idx} ({}) index {i}: GPU hash != CPU hash for nonce {nonce}",
                            variant.name()
                        );
                    }
                });
            }
        });

        verified += count as u64;
        if let BulkVariant::U64 = variant {
            verified_u64 += count as u64;
        }
        chunk_idx += 1;
        if chunk_idx.is_multiple_of(4) {
            println!("  bulk: {verified}/{total} hashes verified");
        }
    }
    println!(
        "BULK OK: {verified} full hashes verified vs CPU (seed {seed}): {} 32-bit, {verified_u64} native-u64",
        verified - verified_u64
    );
}

fn pollster_block<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Runtime::new().unwrap().block_on(fut)
}

struct Stats {
    found: usize,
    exhausted: usize,
    skipped: usize,
}

fn verify_candidate(
    ctx: &JobContext,
    candidate: &engine_cpu::Candidate,
    range: &Range,
    label: &str,
) {
    let cpu_hash = pow_core::hash_from_nonce(ctx, candidate.nonce);
    assert_eq!(
        cpu_hash, candidate.hash,
        "{label}: GPU hash != CPU hash for nonce {}",
        candidate.nonce
    );
    assert!(
        cpu_hash < ctx.target,
        "{label}: hash {:x} not below target {:x}",
        cpu_hash,
        ctx.target
    );
    assert!(
        candidate.nonce >= range.start && candidate.nonce <= range.end,
        "{label}: nonce outside range"
    );
    assert_eq!(
        candidate.work,
        candidate.nonce.to_big_endian(),
        "{label}: work bytes do not match nonce"
    );
}

fn main() {
    env_logger::init();
    let jobs: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("job count"))
        .unwrap_or(200);
    let seed: u64 = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("seed"))
        .unwrap_or_else(|| rand::rng().random());
    let bulk: u64 = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("bulk hash count"))
        .unwrap_or(1_000_000);
    println!("fuzzing with seed {seed}: {bulk} bulk hashes, then {jobs} jobs");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    if bulk > 0 {
        run_bulk_phase(&mut rng, bulk, seed);
    }

    let engine = GpuEngine::try_new(1_000_000, 0, false).expect("GPU init failed");
    let cancel_flag = AtomicBool::new(false);
    let cancel = AtomicBoolCancelCheck(&cancel_flag);
    let mut stats = Stats {
        found: 0,
        exhausted: 0,
        skipped: 0,
    };

    for job in 0..jobs {
        let mut header = [0u8; 32];
        rng.fill_bytes(&mut header);
        let kind = rng.random_range(0..100u32);
        let label = format!("seed {seed} job {job} kind {kind}");

        if kind < 55 {
            // random
            let difficulty = U512::from(10u64.pow(rng.random_range(4..=6)));
            let ctx = JobContext::new(header, difficulty);
            let start = random_u512(&mut rng, true);
            let len = rng.random_range(1_000_000u64..=10_000_000);
            let range = Range {
                start,
                end: start + U512::from(len),
            };
            match engine.search_range(&ctx, range.clone(), &cancel) {
                EngineStatus::Found { candidate, .. } => {
                    verify_candidate(&ctx, &candidate, &range, &label);
                    stats.found += 1;
                }
                EngineStatus::Exhausted { .. } => stats.exhausted += 1,
                other => panic!("{label}: unexpected status {other:?}"),
            }
        } else if kind < 70 {
            // carry-edge: low limbs saturated so nonce increments cascade carries
            let saturated_limbs = rng.random_range(1usize..=8);
            let mut le = random_u512(&mut rng, true).to_little_endian();
            for limb in 0..saturated_limbs {
                for b in 0..4 {
                    le[limb * 4 + b] = 0xFF;
                }
            }
            let margin = rng.random_range(0u64..1000);
            let start = U512::from_little_endian(&le) - U512::from(margin);
            let difficulty = U512::from(50_000u64);
            let ctx = JobContext::new(header, difficulty);
            let range = Range {
                start,
                end: start + U512::from(3_000_000u64),
            };
            match engine.search_range(&ctx, range.clone(), &cancel) {
                EngineStatus::Found { candidate, .. } => {
                    verify_candidate(&ctx, &candidate, &range, &label);
                    stats.found += 1;
                }
                EngineStatus::Exhausted { .. } => stats.exhausted += 1,
                other => panic!("{label}: unexpected status {other:?}"),
            }
        } else if kind < 75 {
            // boundary: cross a 2^256 boundary mid-range
            let hi = rng.random_range(1u64..1000);
            let back = rng.random_range(1u64..2_000_000);
            let start = (U512::from(hi) << 256) - U512::from(back);
            let difficulty = U512::from(50_000u64);
            let ctx = JobContext::new(header, difficulty);
            let range = Range {
                start,
                end: start + U512::from(4_000_000u64),
            };
            match engine.search_range(&ctx, range.clone(), &cancel) {
                EngineStatus::Found { candidate, .. } => {
                    verify_candidate(&ctx, &candidate, &range, &label);
                    stats.found += 1;
                }
                EngineStatus::Exhausted { .. } => stats.exhausted += 1,
                other => panic!("{label}: unexpected status {other:?}"),
            }
        } else if kind < 85 {
            // impossible: no solutions, and the hash count must be exact
            let ctx = JobContext::new(header, U512::MAX);
            let start = random_u512(&mut rng, true);
            let len = rng.random_range(100_000u64..=3_000_000);
            let range = Range {
                start,
                end: start + U512::from(len - 1),
            };
            match engine.search_range(&ctx, range, &cancel) {
                EngineStatus::Exhausted { hash_count } => {
                    assert_eq!(hash_count, len, "{label}: inexact hash count");
                    stats.exhausted += 1;
                }
                other => panic!("{label}: unexpected status {other:?}"),
            }
        } else {
            // known-solution: CPU finds a seal, GPU must not miss it
            let difficulty = U512::from(30_000u64);
            let ctx = JobContext::new(header, difficulty);
            let start = random_u512(&mut rng, true);
            match pow_core::mine_nonce_range(&ctx, start, 300_000) {
                Some((cpu_nonce, cpu_hash)) => {
                    assert!(cpu_hash < ctx.target);
                    let range = Range {
                        start,
                        end: cpu_nonce,
                    };
                    match engine.search_range(&ctx, range.clone(), &cancel) {
                        EngineStatus::Found { candidate, .. } => {
                            verify_candidate(&ctx, &candidate, &range, &label);
                            stats.found += 1;
                        }
                        other => panic!(
                            "{label}: GPU missed a solution the CPU found at {cpu_nonce} ({other:?})"
                        ),
                    }
                }
                None => stats.skipped += 1,
            }
        }

        if (job + 1) % 25 == 0 {
            println!(
                "  {}/{jobs} jobs: {} found, {} exhausted, {} skipped",
                job + 1,
                stats.found,
                stats.exhausted,
                stats.skipped
            );
        }
    }

    assert!(
        stats.found >= jobs / 4,
        "too few solutions found ({} of {jobs}) - fuzz not exercising the claim path",
        stats.found
    );
    println!(
        "PARITY OK: {jobs} jobs (seed {seed}): {} found (all verified vs CPU), {} exhausted, {} skipped",
        stats.found, stats.exhausted, stats.skipped
    );
}
