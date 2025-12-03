#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

use engine_cpu::{Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use futures::executor::block_on;
use pow_core::JobContext;
use primitive_types::U512;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
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

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("No suitable GPU adapter found: {:?}", e))?;

        log::info!(target: "gpu_engine", "Selected adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Mining Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                ..Default::default()
            })
            .await?;

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
        log::info!(target: "gpu_engine", "GPU engine initialized successfully");

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    fn run_batch(&self, ctx: &JobContext, start_nonce: U512, batch_size: u32) -> Option<Candidate> {
        // Prepare buffers
        let mut header_u32s = [0u32; 8];
        for i in 0..8 {
            let chunk = &ctx.header[i * 4..(i + 1) * 4];
            header_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        let start_nonce_bytes = start_nonce.to_little_endian();
        let mut start_nonce_u32s = [0u32; 16];
        for i in 0..16 {
            let chunk = &start_nonce_bytes[i * 4..(i + 1) * 4];
            start_nonce_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        let target_bytes = ctx.difficulty.to_little_endian();
        let mut target_u32s = [0u32; 16];
        for i in 0..16 {
            let chunk = &target_bytes[i * 4..(i + 1) * 4];
            target_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        let header_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Header Buffer"),
                contents: bytemuck::cast_slice(&header_u32s),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let start_nonce_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Start Nonce Buffer"),
                    contents: bytemuck::cast_slice(&start_nonce_u32s),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let target_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Target Buffer"),
                contents: bytemuck::cast_slice(&target_u32s),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Results: [flag (1), nonce (16), hash (16)] = 33 u32s
        let results_size = (1 + 16 + 16) * 4;
        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: results_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize results to 0
        let zeros = vec![0u8; results_size as usize];
        self.queue.write_buffer(&results_buffer, 0, &zeros);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mining Bind Group"),
            layout: &self.bind_group_layout,
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
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Mining Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mining Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (batch_size + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: results_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, results_size);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        let _ = self.device.poll(wgpu::PollType::Wait {
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

                return Some(Candidate { nonce, work, hash });
            }
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
        let mut current_start = range.start;
        let mut hash_count = 0;
        let batch_size = 65536; // 256 * 256

        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        while current_start <= range.end {
            if cancel.load(Ordering::Relaxed) {
                return EngineStatus::Cancelled { hash_count };
            }

            let remaining = range.end - current_start + 1;
            let current_batch_size = if remaining < U512::from(batch_size) {
                remaining.as_u64() as u32
            } else {
                batch_size
            };

            if let Some(candidate) = self.run_batch(ctx, current_start, current_batch_size) {
                // We found a candidate!
                // The hash_count is approximate since we don't know exactly which thread found it
                // without reading back the index, but we can just say we did the whole batch.
                // Or we can calculate it from candidate.nonce - current_start.
                let found_offset = (candidate.nonce - current_start).as_u64();
                return EngineStatus::Found {
                    candidate,
                    hash_count: hash_count + found_offset + 1,
                    origin: FoundOrigin::GpuG1,
                };
            }

            hash_count += current_batch_size as u64;
            current_start = current_start + U512::from(current_batch_size);
        }

        EngineStatus::Exhausted { hash_count }
    }
}
