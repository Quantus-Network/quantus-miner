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
    adapter_info: wgpu::AdapterInfo,
    // Cached vendor configuration
    optimal_workgroups: u32,
    estimated_cores: u32,
    // Reusable buffers
    header_buffer: wgpu::Buffer,
    target_buffer: wgpu::Buffer,
    start_nonce_buffer: wgpu::Buffer,
    results_buffer: wgpu::Buffer,
    dispatch_config_buffer: wgpu::Buffer,
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
            log::info!(target: "gpu_engine", "Adapter {} detailed info:", i);
            log::info!(target: "gpu_engine", "  Name: {}", info.name);
            log::info!(target: "gpu_engine", "  Vendor: {}", info.vendor);
            log::info!(target: "gpu_engine", "  Device: {}", info.device);
            log::info!(target: "gpu_engine", "  Device Type: {:?}", info.device_type);
            log::info!(target: "gpu_engine", "  Driver: {}", info.driver);
            log::info!(target: "gpu_engine", "  Driver Info: {}", info.driver_info);
            log::info!(target: "gpu_engine", "  Backend: {:?}", info.backend);
            log::debug!(target: "gpu_engine", "Adapter {} full info: {:?}", i, info);
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
            log::info!(target: "gpu_engine", "Device limits for adapter {}:", i);
            let limits = device.limits();
            log::info!(target: "gpu_engine", "  Max workgroups per dimension: {}", limits.max_compute_workgroups_per_dimension);
            log::info!(target: "gpu_engine", "  Max workgroup size X: {}", limits.max_compute_workgroup_size_x);
            log::info!(target: "gpu_engine", "  Max workgroup size Y: {}", limits.max_compute_workgroup_size_y);
            log::info!(target: "gpu_engine", "  Max workgroup size Z: {}", limits.max_compute_workgroup_size_z);
            log::info!(target: "gpu_engine", "  Max compute invocations per workgroup: {}", limits.max_compute_invocations_per_workgroup);
            log::info!(target: "gpu_engine", "  Max buffer size: {}", limits.max_buffer_size);

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

            // Dispatch config: [total_threads, nonces_per_thread, workgroups, threads_per_workgroup] = 4 u32s
            let dispatch_config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dispatch Config Buffer"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dispatch_config_buffer.as_entire_binding(),
                    },
                ],
            });

            log::debug!(target: "gpu_engine", "Buffers and bind group initialized for adapter {}", i);

            // Calculate vendor-specific configuration once during initialization
            let (optimal_workgroups, estimated_cores) =
                Self::get_vendor_specific_dispatch(&info, &device);

            contexts.push(Arc::new(GpuContext {
                device,
                queue,
                pipeline,
                bind_group,
                adapter_info: info,
                optimal_workgroups,
                estimated_cores,
                header_buffer,
                target_buffer,
                start_nonce_buffer,
                results_buffer,
                dispatch_config_buffer,
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
        log::debug!(
            "Starting continuous GPU search on device 0 from nonce {}",
            range.start
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

        // Get vendor-specific optimal dispatch configuration using stored adapter info.
        // We always dispatch the full set of optimal workgroups and let each thread
        // process multiple nonces so that:
        // - All nonces tested by this engine are disjoint (no overlap between batches)
        // - We amortize GPU dispatch and synchronization overhead over a lot of work
        let threads_per_workgroup = 256u32; // Must match shader @workgroup_size

        // Each thread will walk a strided sequence of nonces. We deliberately keep this
        // moderate to avoid TDR / watchdog timeouts on consumer GPUs.
        let nonces_per_thread = 256u32;

        // Use cached vendor-specific dispatch configuration
        let actual_workgroups = gpu_ctx.optimal_workgroups;
        let actual_threads = actual_workgroups * threads_per_workgroup;
        let work_per_batch = actual_threads as u64 * nonces_per_thread as u64;

        // Log effective configuration once per search_range
        let total_range_size = (range.end - range.start + 1).as_u64();
        log::info!(
            target: "gpu_engine",
            "GPU batch configuration: total_range={} nonces, work_per_batch={} ({} workgroups × {} threads × {} nonces/thread)",
            total_range_size,
            work_per_batch,
            actual_workgroups,
            threads_per_workgroup,
            nonces_per_thread
        );

        // Track how many nonces this engine has assigned within the requested range so
        // we don't keep dispatching once we've conceptually covered the whole window.
        let mut assigned = 0u64;

        let mut hash_count = 0u64;

        // Eliminate intermediate syncs - only sync at end or when solution found
        const RESULTS_SIZE: usize = (1 + 16 + 16) * 4;
        const ZEROS: [u8; RESULTS_SIZE] = [0; RESULTS_SIZE];

        #[cfg(feature = "metrics")]
        {
            let device_id = "gpu-0";
            metrics::set_gpu_batch_size(device_id, work_per_batch as f64);
            let workgroups = actual_workgroups; // Use actual workgroups for this GPU
            metrics::set_gpu_workgroups(device_id, workgroups as f64);
        }

        let mut batch_count = 0u32;
        let total_start = std::time::Instant::now();

        while assigned < total_range_size && current_start <= range.end {
            if cancel.load(Ordering::Relaxed) {
                log::debug!(target: "gpu_engine", "GPU 0 cancelled.");
                return EngineStatus::Cancelled { hash_count };
            }

            // Determine how much work to do in this batch, capped to the remaining range.
            let remaining = total_range_size - assigned;
            let batch_work = remaining.min(work_per_batch);

            // Update dispatch configuration for this batch:
            // [total_threads, nonces_per_thread, batch_work, threads_per_workgroup]
            let dispatch_config = [
                actual_threads,
                nonces_per_thread,
                batch_work as u32,
                threads_per_workgroup,
            ];
            gpu_ctx.queue.write_buffer(
                &gpu_ctx.dispatch_config_buffer,
                0,
                bytemuck::cast_slice(&dispatch_config),
            );

            // Update start nonce for this batch. Each batch walks a disjoint window of
            // size `batch_work` starting from `current_start`.
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
                // Always dispatch the full optimal configuration; each thread will
                // process `nonces_per_thread` nonces.
                cpass.dispatch_workgroups(actual_workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &gpu_ctx.results_buffer,
                0,
                &gpu_ctx.staging_buffer,
                0,
                RESULTS_SIZE as u64,
            );

            // Submit immediately to ensure correct buffer state
            gpu_ctx.queue.submit(Some(encoder.finish()));

            // Check results for this batch
            let buffer_slice = gpu_ctx.staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

            let _ = gpu_ctx.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            let data = buffer_slice.get_mapped_range();
            let result_u32s: &[u32] = bytemuck::cast_slice(&data);

            if result_u32s[0] != 0 {
                // Solution found!
                let nonce_u32s = &result_u32s[1..17];
                let hash_u32s = &result_u32s[17..33];

                let nonce = U512::from_little_endian(bytemuck::cast_slice(nonce_u32s));
                let hash = U512::from_little_endian(bytemuck::cast_slice(hash_u32s));
                let work = nonce.to_big_endian();

                log::info!("GPU 0 found solution! Nonce: {}, Hash: {:x}", nonce, hash);

                drop(data);
                gpu_ctx.staging_buffer.unmap();

                let batch_hashes = (hash_count + batch_work).min(total_range_size);
                return EngineStatus::Found {
                    candidate: Candidate { nonce, work, hash },
                    // Approximate: we assume a uniform distribution within the batch and
                    // cap reported work to the requested range length.
                    hash_count: batch_hashes,
                    origin: FoundOrigin::GpuG1,
                };
            }

            drop(data);
            gpu_ctx.staging_buffer.unmap();

            // Advance to the next disjoint batch of nonces. We deliberately do not
            // require contiguity between CPU and GPU engines, only that batches are
            // non-overlapping within this engine.
            hash_count += batch_work;
            assigned = assigned.saturating_add(batch_work);
            current_start = current_start + U512::from(batch_work);
            batch_count += 1;
        }

        let total_time = total_start.elapsed();
        let effective_hashes = hash_count.min(total_range_size);
        log::info!(
            target: "gpu_engine",
            "GPU finished range. Total time: {:.2}ms, Batches: {}, Hashes: {}, Performance: {:.0} hashes/sec",
            total_time.as_secs_f64() * 1000.0,
            batch_count,
            effective_hashes,
            effective_hashes as f64 / total_time.as_secs_f64()
        );

        log::debug!(target: "gpu_engine", "GPU 0 finished range.");
        EngineStatus::Exhausted {
            hash_count: effective_hashes,
        }
    }
}

impl GpuEngine {
    /// Get vendor-specific optimal dispatch configuration
    fn get_vendor_specific_dispatch(
        adapter_info: &wgpu::AdapterInfo,
        device: &wgpu::Device,
    ) -> (u32, u32) {
        let limits = device.limits();
        let max_workgroups = limits.max_compute_workgroups_per_dimension.min(65535);

        // Parse vendor from adapter info
        let vendor_name = adapter_info.name.to_lowercase();
        let _device_name = adapter_info.device.to_string().to_lowercase();

        // Vendor detection logic without logging (already logged during initialization)
        // Vendor-specific heuristics based on architecture knowledge
        let (optimal_workgroups, estimated_cores) = if vendor_name.contains("nvidia")
            || adapter_info.vendor == 4318
        {
            // NVIDIA GPUs (vendor ID 0x10DE = 4318)
            // CUDA cores benefit from high occupancy, typically thousands of simple cores
            let workgroups = if vendor_name.contains("rtx 40") || vendor_name.contains("rtx 4090") {
                (max_workgroups / 8).max(4096) // High-end: RTX 4090 has 16384 CUDA cores
            } else if vendor_name.contains("rtx 30") || vendor_name.contains("rtx 20") {
                (max_workgroups / 12).max(2048) // Mid-high-end: RTX 3080 has ~8700 CUDA cores
            } else if vendor_name.contains("gtx") || vendor_name.contains("rtx 16") {
                (max_workgroups / 16).max(1024) // Mid-range: GTX 1660 has 1408 CUDA cores
            } else {
                (max_workgroups / 20).max(512) // Lower-end NVIDIA
            };
            let cores = workgroups * 256 / 2; // NVIDIA benefits from ~2x oversubscription
            (workgroups, cores)
        } else if vendor_name.contains("amd") || adapter_info.vendor == 4098 {
            // AMD GPUs (vendor ID 0x1002 = 4098)
            // Compute Units with stream processors, different architecture than NVIDIA
            let workgroups = if vendor_name.contains("rx 7") || vendor_name.contains("rx 6900") {
                (max_workgroups / 10).max(3072) // High-end: RX 7900 XTX has 6144 stream processors
            } else if vendor_name.contains("rx 6") || vendor_name.contains("rx 5700") {
                (max_workgroups / 14).max(2048) // Mid-high-end: RX 6800 XT has 4608 stream processors
            } else if vendor_name.contains("rx 5") || vendor_name.contains("rx 580") {
                (max_workgroups / 18).max(1024) // Mid-range: RX 580 has 2304 stream processors
            } else {
                (max_workgroups / 24).max(512) // Lower-end AMD
            };
            let cores = workgroups * 256 / 3; // AMD typically benefits from ~3x oversubscription
            (workgroups, cores)
        } else if vendor_name.contains("intel") || adapter_info.vendor == 32902 {
            // Intel GPUs (vendor ID 0x8086 = 32902)
            // Newer Xe architecture, different from traditional designs
            let workgroups = if vendor_name.contains("arc a7") || vendor_name.contains("arc a770") {
                (max_workgroups / 12).max(2048) // High-end Arc A770
            } else if vendor_name.contains("arc a5") || vendor_name.contains("arc a380") {
                (max_workgroups / 16).max(1024) // Mid-range Arc A380
            } else if vendor_name.contains("iris xe") {
                (max_workgroups / 20).max(512) // Integrated Iris Xe
            } else {
                (max_workgroups / 24).max(256) // Other Intel integrated
            };
            let cores = workgroups * 256 / 4; // Conservative approach for Intel
            (workgroups, cores)
        } else if adapter_info.backend == wgpu::Backend::Metal {
            // Apple GPUs (detected by Metal backend)
            // Apple GPU cores are complex, powerful units - NOT like NVIDIA CUDA cores
            // Each Apple GPU core can handle many operations in parallel
            let (gpu_cores, workgroups) = if vendor_name.contains("m4 max") {
                (40, 800) // M4 Max: 40 GPU cores, ~20x workgroups
            } else if vendor_name.contains("m4 pro") {
                (20, 400) // M4 Pro: 20 GPU cores, ~20x workgroups
            } else if vendor_name.contains("m4") {
                (10, 200) // M4: 10 GPU cores, ~20x workgroups
            } else if vendor_name.contains("m3 max") {
                (40, 800) // M3 Max: 40 GPU cores
            } else if vendor_name.contains("m3 pro") {
                (18, 360) // M3 Pro: 18 GPU cores
            } else if vendor_name.contains("m3") {
                (10, 200) // M3: 10 GPU cores
            } else if vendor_name.contains("m2 ultra") {
                (76, 1520) // M2 Ultra: 76 GPU cores
            } else if vendor_name.contains("m2 max") {
                (38, 760) // M2 Max: 38 GPU cores
            } else if vendor_name.contains("m2 pro") {
                (19, 380) // M2 Pro: 19 GPU cores
            } else if vendor_name.contains("m2") {
                (10, 200) // M2: 10 GPU cores
            } else if vendor_name.contains("m1 ultra") {
                (64, 1280) // M1 Ultra: 64 GPU cores
            } else if vendor_name.contains("m1 max") {
                (32, 640) // M1 Max: 32 GPU cores
            } else if vendor_name.contains("m1 pro") {
                (16, 320) // M1 Pro: 16 GPU cores
            } else if vendor_name.contains("m1") {
                (8, 160) // M1: 8 GPU cores
            } else {
                (8, 160) // Conservative fallback for unknown Apple GPU
            };

            // Clamp to reasonable limits
            let clamped_workgroups = workgroups.min(max_workgroups / 4).max(64);

            // Each Apple GPU core is roughly equivalent to 64-128 "effective processing units"
            // This gives us a realistic estimate of parallel processing capability
            let estimated_cores = gpu_cores * 96; // Conservative middle ground

            // Apple GPU configuration calculated
            (clamped_workgroups, estimated_cores)
        } else {
            // Unknown/Generic GPU - use conservative defaults
            // Unknown/Generic GPU - use conservative defaults (logging moved to initialization)
            let workgroups = (max_workgroups / 16).max(512);
            let cores = workgroups * 256 / 4;
            (workgroups, cores)
        };

        log::info!(target: "gpu_engine", "Vendor-specific dispatch configuration:");
        log::info!(target: "gpu_engine", "  Max hardware workgroups: {}", max_workgroups);
        log::info!(target: "gpu_engine", "  Optimal workgroups: {}", optimal_workgroups);
        log::info!(target: "gpu_engine", "  Estimated effective cores: {}", estimated_cores);

        (optimal_workgroups, estimated_cores)
    }
}
