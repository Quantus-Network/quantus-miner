#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

use engine_cpu::{Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use futures::executor::block_on;
use pow_core::JobContext;
use primitive_types::U512;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Represents a single GPU device context.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    // Reusable buffers
    header_buffer: wgpu::Buffer,
    target_buffer: wgpu::Buffer,
    start_nonce_buffer: wgpu::Buffer,
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
        log::info!("Initializing WGPU...");
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

        // Collect adapters to a vector to check count and iterate with index
        let adapters: Vec<_> = adapters.into_iter().collect();

        if adapters.is_empty() {
            log::error!("No suitable GPU adapters found.");
            return Err("No suitable GPU adapters found".into());
        }

        let mut contexts = Vec::new();
        let mut adapter_infos = Vec::new();
        for (i, adapter) in adapters.into_iter().enumerate() {
            let info = adapter.get_info();
            log::info!(
                "Initializing GPU adapter {}: {} (Backend: {:?})",
                i,
                info.name,
                info.backend
            );
            log::debug!(target: "gpu_engine", "Adapter {} info: {:?}", i, info);
            adapter_infos.push(info.clone());

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

            // Start Nonce: 16 u32s
            let start_nonce_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Start Nonce Buffer"),
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

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mining Bind Group"),
                layout: &bind_group_layout,
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

            log::debug!(target: "gpu_engine", "Buffers and bind group initialized for adapter {}", i);

            contexts.push(Arc::new(GpuContext {
                device,
                queue,
                pipeline,
                bind_group,
                header_buffer,
                target_buffer,
                start_nonce_buffer,
                results_buffer,
                staging_buffer,
            }));
        }

        log::info!("GPU engine initialized with {} devices", contexts.len());

        // Set engine backend info for metrics
        #[cfg(feature = "metrics")]
        {
            metrics::set_gpu_device_count(contexts.len() as i64);

            for (i, adapter_info) in adapter_infos.iter().enumerate() {
                let device_id = format!("gpu-{}", i);
                let backend_str = format!("{:?}", adapter_info.backend);
                let vendor_str = format!("{}", adapter_info.vendor);
                let device_type_str = format!("{:?}", adapter_info.device_type);
                let clean_name = adapter_info.name.replace(" ", "_").replace(",", "");

                // Set general engine backend info
                metrics::set_engine_backend(&device_id, &backend_str);

                // Set detailed GPU device info
                metrics::set_gpu_device_info(
                    &device_id,
                    &clean_name,
                    &backend_str,
                    &vendor_str,
                    &device_type_str,
                );

                // Log GPU device info for monitoring
                log::info!(
                    "📊 GPU Device {}: {} | Backend: {:?} | Vendor: {} | Device Type: {:?}",
                    i,
                    adapter_info.name,
                    adapter_info.backend,
                    adapter_info.vendor,
                    adapter_info.device_type
                );
            }
        }

        Ok(Self { contexts })
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
        if self.contexts.is_empty() {
            log::warn!("No GPUs available for search.");
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        // Use first GPU only for now to simplify and avoid threading overhead
        let gpu_ctx = &self.contexts[0];
        log::info!(
            "Starting GPU search on device 0. Range: {} - {}",
            range.start,
            range.end
        );

        // Pre-convert header and target once (not per batch)
        let mut header_u32s = [0u32; 8];
        for i in 0..8 {
            let chunk = &ctx.header[i * 4..(i + 1) * 4];
            header_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        gpu_ctx.queue.write_buffer(
            &gpu_ctx.header_buffer,
            0,
            bytemuck::cast_slice(&header_u32s),
        );

        let target_bytes = ctx.target.to_little_endian();
        let mut target_u32s = [0u32; 16];
        for i in 0..16 {
            let chunk = &target_bytes[i * 4..(i + 1) * 4];
            target_u32s[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        gpu_ctx.queue.write_buffer(
            &gpu_ctx.target_buffer,
            0,
            bytemuck::cast_slice(&target_u32s),
        );

        let mut current_start = range.start;
        // GPU hardware limit: max 65535 workgroups per dimension
        // With workgroup_size(256): max_nonces = 65535 * 256 = 16,777,344
        let batch_size = 65535 * 256; // ~16.7M hashes per batch (max hardware allows)
        let mut hash_count = 0u64;

        // Batch multiple dispatches to reduce sync overhead
        const BATCHES_PER_SYNC: u32 = 4; // Process 4 batches before checking results
        const RESULTS_SIZE: usize = (1 + 16 + 16) * 4;
        const ZEROS: [u8; RESULTS_SIZE] = [0; RESULTS_SIZE];

        // Results buffer must be reset per batch to detect solutions correctly

        #[cfg(feature = "metrics")]
        {
            let device_id = "gpu-0";
            metrics::set_gpu_batch_size(device_id, (batch_size * BATCHES_PER_SYNC) as f64);
            let workgroups = (batch_size + 255) / 256;
            metrics::set_gpu_workgroups(device_id, (workgroups * BATCHES_PER_SYNC) as f64);
        }

        // Collect command buffers to submit in batches
        let mut command_buffers = Vec::new();

        while current_start <= range.end {
            if cancel.load(Ordering::Relaxed) {
                log::debug!(target: "gpu_engine", "GPU 0 cancelled.");
                return EngineStatus::Cancelled { hash_count };
            }

            let remaining = range.end - current_start + 1;
            let current_batch_size = if remaining < U512::from(batch_size) {
                remaining.as_u64() as u32
            } else {
                batch_size
            };

            // Update start nonce (simplified conversion)
            let start_nonce_bytes = current_start.to_little_endian();
            gpu_ctx
                .queue
                .write_buffer(&gpu_ctx.start_nonce_buffer, 0, &start_nonce_bytes);

            // Reset results buffer to detect solutions from this batch
            gpu_ctx
                .queue
                .write_buffer(&gpu_ctx.results_buffer, 0, &ZEROS);

            // Create command buffer for this batch
            let mut encoder = gpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&gpu_ctx.pipeline);
                cpass.set_bind_group(0, &gpu_ctx.bind_group, &[]);
                cpass.dispatch_workgroups((current_batch_size + 255) / 256, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &gpu_ctx.results_buffer,
                0,
                &gpu_ctx.staging_buffer,
                0,
                RESULTS_SIZE as u64,
            );

            // Store command buffer instead of immediate submit
            command_buffers.push(encoder.finish());

            hash_count += current_batch_size as u64;
            current_start = current_start + U512::from(current_batch_size);

            // Submit and check results less frequently to reduce sync overhead
            if hash_count % (batch_size * BATCHES_PER_SYNC) as u64 == 0 || current_start > range.end
            {
                // Submit all batched commands at once
                gpu_ctx.queue.submit(command_buffers.drain(..));

                let buffer_slice = gpu_ctx.staging_buffer.slice(..);
                buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
                let _ = gpu_ctx.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });

                let data = buffer_slice.get_mapped_range();
                let result_u32s: &[u32] = bytemuck::cast_slice(&data);

                if result_u32s[0] != 0 {
                    // Direct conversion from u32 slice to U512
                    let nonce_u32s = &result_u32s[1..17];
                    let hash_u32s = &result_u32s[17..33];

                    let nonce = U512::from_little_endian(bytemuck::cast_slice(nonce_u32s));
                    let hash = U512::from_little_endian(bytemuck::cast_slice(hash_u32s));
                    let work = nonce.to_big_endian();

                    log::info!("GPU 0 found solution! Nonce: {}, Hash: {:x}", nonce, hash);

                    drop(data);
                    gpu_ctx.staging_buffer.unmap();

                    return EngineStatus::Found {
                        candidate: Candidate { nonce, work, hash },
                        hash_count,
                        origin: FoundOrigin::GpuG1,
                    };
                }

                drop(data);
                gpu_ctx.staging_buffer.unmap();
            }
        }

        log::debug!(target: "gpu_engine", "GPU 0 finished.");
        EngineStatus::Exhausted { hash_count }
    }
}
