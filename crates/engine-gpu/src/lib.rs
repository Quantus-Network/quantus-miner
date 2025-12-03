#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

use engine_cpu::{Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use futures::executor::block_on;
use pow_core::JobContext;
use primitive_types::U512;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Represents a single GPU device context.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    // Reusable buffers
    header_buffer: wgpu::Buffer,
    target_buffer: wgpu::Buffer,
    results_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
}

pub struct GpuEngine {
    contexts: Vec<Arc<GpuContext>>,
}

impl GpuEngine {
    pub fn new() -> Self {
        block_on(Self::init()).expect("Failed to initialize GPU engine")
    }

    async fn init() -> Result<Self, Box<dyn std::error::Error>> {
        log::info!(target: "gpu_engine", "Initializing WGPU...");
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

        // Collect adapters to a vector to check count and iterate with index
        let adapters: Vec<_> = adapters.into_iter().collect();

        if adapters.is_empty() {
            log::error!(target: "gpu_engine", "No suitable GPU adapters found.");
            return Err("No suitable GPU adapters found".into());
        }

        let mut contexts = Vec::new();
        for (i, adapter) in adapters.into_iter().enumerate() {
            let info = adapter.get_info();
            log::info!(target: "gpu_engine", "Initializing adapter {}: {} (Backend: {:?})", i, info.name, info.backend);
            log::debug!(target: "gpu_engine", "Adapter {} info: {:?}", i, info);

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Mining Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                    ..Default::default()
                })
                .await?;

            log::debug!(target: "gpu_engine", "Device and Queue requested for adapter {}", i);

            let shader_source = include_str!("mining.wgsl");
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Mining Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Mining Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("mining_main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);

            // Pre-allocate buffers
            // Header: 8 u32s
            let header_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Header Buffer"),
                size: 32,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Target: 16 u32s
            let target_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Target Buffer"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Results: [flag (1), nonce (16), hash (16)] = 33 u32s
            let results_size = (1 + 16 + 16) * 4;
            let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Results Buffer"),
                size: results_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: results_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            log::debug!(target: "gpu_engine", "Buffers initialized for adapter {}", i);

            contexts.push(Arc::new(GpuContext {
                device,
                queue,
                pipeline,
                bind_group_layout,
                header_buffer,
                target_buffer,
                results_buffer,
                staging_buffer,
            }));
        }

        log::info!(target: "gpu_engine", "GPU engine initialized with {} devices", contexts.len());

        Ok(Self { contexts })
    }

    fn run_batch(
        ctx: &GpuContext,
        job_ctx: &JobContext,
        start_nonce: U512,
        batch_size: u32,
    ) -> Option<Candidate> {
        // Update Header Buffer
        let mut header_u32s = [0u32; 8];
        for i in 0..8 {
            let chunk = &job_ctx.header[i * 4..(i + 1) * 4];
            header_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        ctx.queue
            .write_buffer(&ctx.header_buffer, 0, bytemuck::cast_slice(&header_u32s));

        // Update Target Buffer
        let target_bytes = job_ctx.difficulty.to_little_endian();
        let mut target_u32s = [0u32; 16];
        for i in 0..16 {
            let chunk = &target_bytes[i * 4..(i + 1) * 4];
            target_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        ctx.queue
            .write_buffer(&ctx.target_buffer, 0, bytemuck::cast_slice(&target_u32s));

        // Create Start Nonce Buffer (per batch)
        let start_nonce_bytes = start_nonce.to_little_endian();
        let mut start_nonce_u32s = [0u32; 16];
        for i in 0..16 {
            let chunk = &start_nonce_bytes[i * 4..(i + 1) * 4];
            start_nonce_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        let start_nonce_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Start Nonce Buffer"),
                contents: bytemuck::cast_slice(&start_nonce_u32s),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Reset Results Buffer
        let results_size = (1 + 16 + 16) * 4;
        let zeros = vec![0u8; results_size as usize];
        ctx.queue.write_buffer(&ctx.results_buffer, 0, &zeros);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mining Bind Group"),
            layout: &ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: start_nonce_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ctx.target_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Mining Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mining Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (batch_size + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&ctx.results_buffer, 0, &ctx.staging_buffer, 0, results_size);

        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = ctx.staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        let _ = ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Ok(Ok(())) = block_on(receiver) {
            let data = buffer_slice.get_mapped_range();
            let result_u32s: &[u32] = bytemuck::cast_slice(&data);

            if result_u32s[0] != 0 {
                let mut nonce_bytes = [0u8; 64];
                for i in 0..16 {
                    let bytes = result_u32s[1 + i].to_le_bytes();
                    nonce_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
                }
                let nonce = U512::from_little_endian(&nonce_bytes);

                let mut hash_bytes = [0u8; 64];
                for i in 0..16 {
                    let bytes = result_u32s[17 + i].to_le_bytes();
                    hash_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
                }
                // GPU returns hash in Little Endian u32 array (normalized in shader)
                let hash = U512::from_little_endian(&hash_bytes);

                let work = nonce.to_big_endian();
                log::info!(target: "gpu_engine", "Solution found on GPU! Nonce: {}, Hash: {:x}", nonce, hash);

                drop(data);
                ctx.staging_buffer.unmap();
                return Some(Candidate { nonce, work, hash });
            }
            drop(data);
            ctx.staging_buffer.unmap();
        } else {
            log::error!(target: "gpu_engine", "Failed to map staging buffer");
        }

        None
    }
}

impl MinerEngine for GpuEngine {
    fn name(&self) -> &'static str {
        "gpu-wgpu"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus {
        // Simple multi-gpu strategy: split range among available GPUs
        let num_gpus = self.contexts.len();
        log::info!(target: "gpu_engine", "Starting search on {} GPUs. Range: {} - {}", num_gpus, range.start, range.end);

        if num_gpus == 0 {
            log::warn!(target: "gpu_engine", "No GPUs available for search.");
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        let total_range = range.end - range.start + 1;
        let range_per_gpu = total_range / U512::from(num_gpus);
        let remainder = total_range % U512::from(num_gpus);

        let atomic_hash_count = std::sync::atomic::AtomicU64::new(0);
        let found_candidate = std::sync::Mutex::new(None);

        std::thread::scope(|s| {
            for i in 0..num_gpus {
                let gpu_ctx = &self.contexts[i];
                let atomic_hash_count = &atomic_hash_count;
                let found_candidate = &found_candidate;

                let start = range.start + range_per_gpu * U512::from(i);
                let mut end = start + range_per_gpu - 1;
                if i == num_gpus - 1 {
                    end = end + remainder;
                }

                s.spawn(move || {
                    log::debug!(target: "gpu_engine", "GPU {} started. Range: {} - {}", i, start, end);
                    let mut current_start = start;
                    let batch_size = 65536 * 4; // Larger batch size for efficiency

                    while current_start <= end {
                        if cancel.load(Ordering::Relaxed) {
                            log::debug!(target: "gpu_engine", "GPU {} cancelled.", i);
                            break;
                        }
                        if found_candidate.lock().unwrap().is_some() {
                            break;
                        }

                        let remaining = end - current_start + 1;
                        let current_batch_size = if remaining < U512::from(batch_size) {
                            remaining.as_u64() as u32
                        } else {
                            batch_size
                        };

                        if let Some(candidate) =
                            Self::run_batch(gpu_ctx, ctx, current_start, current_batch_size)
                        {
                            let mut lock = found_candidate.lock().unwrap();
                            if lock.is_none() {
                                log::info!(target: "gpu_engine", "GPU {} found candidate!", i);
                                *lock = Some(candidate);
                            }
                            return;
                        }

                        atomic_hash_count.fetch_add(current_batch_size as u64, Ordering::Relaxed);
                        current_start = current_start + U512::from(current_batch_size);
                    }
                    log::debug!(target: "gpu_engine", "GPU {} finished.", i);
                });
            }
        });

        let hash_count = atomic_hash_count.load(Ordering::Relaxed);
        let lock = found_candidate.lock().unwrap();
        if let Some(candidate) = &*lock {
            return EngineStatus::Found {
                candidate: candidate.clone(),
                hash_count,
                origin: FoundOrigin::GpuG1,
            };
        }

        if cancel.load(Ordering::Relaxed) {
            EngineStatus::Cancelled { hash_count }
        } else {
            EngineStatus::Exhausted { hash_count }
        }
    }
}
