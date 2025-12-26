#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

use engine_cpu::{Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use futures::executor::block_on;
use pow_core::JobContext;
use primitive_types::U512;
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

/// Represents a single GPU device context.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,

    // Cached vendor configuration
    optimal_workgroups: u32,
}

#[derive(Clone)]
struct GpuResources {
    header_buffer: wgpu::Buffer,
    target_buffer: wgpu::Buffer,
    start_nonce_buffer: wgpu::Buffer,
    results_buffer: wgpu::Buffer,
    dispatch_config_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct GpuEngine {
    contexts: Vec<Arc<GpuContext>>,
    device_counter: AtomicUsize,
}

// Thread-local storage for consistent GPU device assignment per worker thread
thread_local! {
    static ASSIGNED_GPU_DEVICE: RefCell<Option<usize>> = const { RefCell::new(None) };
    static WORKER_RESOURCES: RefCell<Option<GpuResources>> = const { RefCell::new(None) };
}

impl GpuContext {
    fn create_resources(&self) -> GpuResources {
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);

        // Header: 8 u32s
        let header_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Header Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Target: 16 u32s
        let target_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Target Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Start Nonce: 16 u32s
        let start_nonce_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Start Nonce Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        // Dispatch config: [total_threads, nonces_per_thread, workgroups, threads_per_workgroup] = 4 u32s
        let dispatch_config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dispatch Config Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: results_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        GpuResources {
            header_buffer,
            target_buffer,
            start_nonce_buffer,
            results_buffer,
            dispatch_config_buffer,
            staging_buffer,
            bind_group,
        }
    }
}

impl Default for GpuEngine {
    fn default() -> Self {
        Self::new()
    }
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

            log::debug!(target: "gpu_engine", "Pipeline initialized for adapter {}", i);

            // Calculate vendor-specific configuration once during initialization
            let optimal_workgroups = Self::get_vendor_specific_dispatch(&info, &device);

            contexts.push(Arc::new(GpuContext {
                device,
                queue,
                pipeline,
                optimal_workgroups,
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

        Ok(Self {
            contexts,
            device_counter: AtomicUsize::new(0),
        })
    }

    /// Returns the number of GPU devices available
    pub fn device_count(&self) -> usize {
        self.contexts.len()
    }
}

impl MinerEngine for GpuEngine {
    fn name(&self) -> &'static str {
        "gpu-wgpu"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus {
        if self.contexts.is_empty() {
            log::warn!("No GPUs available for search.");
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        // Empty or inverted range: nothing to do.
        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        // Use thread-local assignment for consistent worker-to-GPU mapping
        let device_index = ASSIGNED_GPU_DEVICE.with(|assigned| {
            let mut assigned_ref = assigned.borrow_mut();
            if let Some(index) = *assigned_ref {
                // This thread already has a GPU assigned
                index
            } else {
                // First time this thread is calling search_range, assign a GPU device
                let index = if self.contexts.len() == 1 {
                    0
                } else {
                    self.device_counter.fetch_add(1, Ordering::SeqCst) % self.contexts.len()
                };
                *assigned_ref = Some(index);
                log::info!(
                    "Worker thread assigned to GPU device {} (of {} total devices)",
                    index,
                    self.contexts.len()
                );
                index
            }
        });

        let gpu_ctx = &self.contexts[device_index];
        log::debug!(
            "GPU device {} processing range {}..={} (inclusive)",
            device_index,
            range.start,
            range.end
        );

        // Ensure resources are initialized for this thread
        WORKER_RESOURCES.with(|resources_cell| {
            let mut resources = resources_cell.borrow_mut();
            if resources.is_none() {
                *resources = Some(gpu_ctx.create_resources());
            }
        });

        // Clone resources to use outside the closure (they are cheap handles)
        let resources = WORKER_RESOURCES.with(|resources_cell| {
            resources_cell.borrow().as_ref().unwrap().clone()
        });

        // Pre-convert header and target once (not per range)
        let mut header_u32s = [0u32; 8];
        for (i, item) in header_u32s.iter_mut().enumerate() {
            let chunk = &ctx.header[i * 4..(i + 1) * 4];
            *item = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        gpu_ctx.queue.write_buffer(
            &resources.header_buffer,
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
            &resources.target_buffer,
            0,
            bytemuck::cast_slice(&target_u32s),
        );

        // Length of the inclusive nonce range
        let total_range_size = (range.end - range.start + 1).as_u64();
        if total_range_size == 0 {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        // Thread configuration for a single dispatch over this range.
        let threads_per_workgroup = 256u32; // Must match shader @workgroup_size(256)
        let limits = gpu_ctx.device.limits();
        let max_workgroups = limits.max_compute_workgroups_per_dimension;

        // Vendor hint, clamped by hardware limits.
        let hinted_workgroups = gpu_ctx.optimal_workgroups.max(1).min(max_workgroups);
        let hinted_threads = hinted_workgroups as u64 * threads_per_workgroup as u64;

        // Choose a logical thread budget: enough threads to fill the GPU, but no more than
        // the range length (spawning more threads than nonces is wasteful).
        let mut logical_threads = total_range_size.min(hinted_threads);
        if logical_threads == 0 {
            logical_threads = 1;
        }

        // Round logical_threads up to a multiple of workgroup size so we have full workgroups.
        let mut num_workgroups = (logical_threads as u32).div_ceil(threads_per_workgroup);
        if num_workgroups == 0 {
            num_workgroups = 1;
        }
        let total_threads = (num_workgroups * threads_per_workgroup) as u64;

        // Derive how many nonces each logical thread should process so that the entire
        // range is covered in a single dispatch.
        let nonces_per_thread = total_range_size.div_ceil(total_threads).max(1) as u32;

        let total_threads_u32 = total_threads as u32;

        log::info!(
            target: "gpu_engine",
            "GPU dispatch configuration: total_range={} nonces, workgroups={}, threads={}, nonces_per_thread={}",
            total_range_size,
            num_workgroups,
            total_threads_u32,
            nonces_per_thread
        );

        // We'll process the full range in a single dispatch.
        let hash_count = total_range_size;

        // Eliminate intermediate syncs - only sync at end or when solution found
        const RESULTS_SIZE: usize = (1 + 16 + 16) * 4;
        const ZEROS: [u8; RESULTS_SIZE] = [0; RESULTS_SIZE];

        #[cfg(feature = "metrics")]
        {
            let device_id = "gpu-0";
            metrics::set_gpu_batch_size(device_id, total_range_size as f64);
            metrics::set_gpu_workgroups(device_id, num_workgroups as f64);
        }

        if cancel.load(Ordering::Relaxed) {
            log::debug!(target: "gpu_engine", "GPU {} cancelled before dispatch.", device_index);
            return EngineStatus::Cancelled { hash_count: 0 };
        }

        // Dispatch configuration for this range:
        // [total_threads, nonces_per_thread, total_nonces, threads_per_workgroup]
        let total_nonces_u32 = total_range_size.min(u32::MAX as u64) as u32;
        let dispatch_config = [
            total_threads_u32,
            nonces_per_thread,
            total_nonces_u32,
            threads_per_workgroup,
        ];
        gpu_ctx.queue.write_buffer(
            &resources.dispatch_config_buffer,
            0,
            bytemuck::cast_slice(&dispatch_config),
        );

        // Starting nonce for this range.
        let start_nonce_bytes = range.start.to_little_endian();
        gpu_ctx
            .queue
            .write_buffer(&resources.start_nonce_buffer, 0, &start_nonce_bytes);

        // Reset results buffer to detect solutions from this dispatch.
        gpu_ctx
            .queue
            .write_buffer(&resources.results_buffer, 0, &ZEROS);

        let total_start = std::time::Instant::now();

        // Create command buffer for this range
        let mut encoder = gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&gpu_ctx.pipeline);
            cpass.set_bind_group(0, &resources.bind_group, &[]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &resources.results_buffer,
            0,
            &resources.staging_buffer,
            0,
            RESULTS_SIZE as u64,
        );

        // Submit and wait for completion
        gpu_ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = resources.staging_buffer.slice(..);
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

            log::info!(
                "GPU {} found solution! Nonce: {}, Hash: {:x}",
                device_index,
                nonce,
                hash
            );

            drop(data);
            resources.staging_buffer.unmap();

            return EngineStatus::Found {
                candidate: Candidate { nonce, work, hash },
                // Approximate: we assume a uniform distribution within the dispatch
                // and report that we processed the entire range.
                hash_count,
                origin: FoundOrigin::GpuG1,
            };
        }

        drop(data);
        resources.staging_buffer.unmap();

        let total_time = total_start.elapsed();
        log::info!(
            target: "gpu_engine",
            "GPU finished range. Total time: {:.2}ms, Batches: 1, Hashes: {}, Performance: {:.0} hashes/sec",
            total_time.as_secs_f64() * 1000.0,
            hash_count,
            hash_count as f64 / total_time.as_secs_f64()
        );

        log::debug!(target: "gpu_engine", "GPU {} finished range with no solution.", device_index);
        EngineStatus::Exhausted { hash_count }
    }
}

impl GpuEngine {
    /// Get vendor-specific optimal dispatch configuration
    fn get_vendor_specific_dispatch(
        adapter_info: &wgpu::AdapterInfo,
        device: &wgpu::Device,
    ) -> u32 {
        let limits = device.limits();
        let max_workgroups = limits.max_compute_workgroups_per_dimension.min(65535);

        // Parse vendor from adapter info
        let vendor_name = adapter_info.name.to_lowercase();
        let _device_name = adapter_info.device.to_string().to_lowercase();

        // Vendor-specific heuristics based on architecture knowledge
        let optimal_workgroups = if vendor_name.contains("nvidia") || adapter_info.vendor == 4318 {
            // NVIDIA GPUs (vendor ID 0x10DE = 4318)
            if vendor_name.contains("rtx 40") || vendor_name.contains("rtx 4090") {
                (max_workgroups / 8).max(4096)
            } else if vendor_name.contains("rtx 30") || vendor_name.contains("rtx 20") {
                (max_workgroups / 12).max(2048)
            } else if vendor_name.contains("gtx") || vendor_name.contains("rtx 16") {
                (max_workgroups / 16).max(1024)
            } else {
                (max_workgroups / 20).max(512)
            }
        } else if vendor_name.contains("amd") || adapter_info.vendor == 4098 {
            // AMD GPUs (vendor ID 0x1002 = 4098)
            if vendor_name.contains("rx 7") || vendor_name.contains("rx 6900") {
                (max_workgroups / 10).max(3072)
            } else if vendor_name.contains("rx 6") || vendor_name.contains("rx 5700") {
                (max_workgroups / 14).max(2048)
            } else if vendor_name.contains("rx 5") || vendor_name.contains("rx 580") {
                (max_workgroups / 18).max(1024)
            } else {
                (max_workgroups / 24).max(512)
            }
        } else if vendor_name.contains("intel") || adapter_info.vendor == 32902 {
            // Intel GPUs (vendor ID 0x8086 = 32902)
            if vendor_name.contains("arc a7") || vendor_name.contains("arc a770") {
                (max_workgroups / 12).max(2048)
            } else if vendor_name.contains("arc a5") || vendor_name.contains("arc a380") {
                (max_workgroups / 16).max(1024)
            } else if vendor_name.contains("iris xe") {
                (max_workgroups / 20).max(512)
            } else {
                (max_workgroups / 24).max(256)
            }
        } else if adapter_info.backend == wgpu::Backend::Metal {
            // Apple GPUs (detected by Metal backend)
            let (gpu_cores, workgroups) = if vendor_name.contains("m4 max") {
                (40, 800)
            } else if vendor_name.contains("m4 pro") {
                (20, 400)
            } else if vendor_name.contains("m4") {
                (10, 200)
            } else if vendor_name.contains("m3 max") {
                (40, 800)
            } else if vendor_name.contains("m3 pro") {
                (18, 360)
            } else if vendor_name.contains("m3") {
                (10, 200)
            } else if vendor_name.contains("m2 ultra") {
                (76, 1520)
            } else if vendor_name.contains("m2 max") {
                (38, 760)
            } else if vendor_name.contains("m2 pro") {
                (19, 380)
            } else if vendor_name.contains("m2") {
                (10, 200)
            } else if vendor_name.contains("m1 ultra") {
                (64, 1280)
            } else if vendor_name.contains("m1 max") {
                (32, 640)
            } else if vendor_name.contains("m1 pro") {
                (16, 320)
            } else {
                (8, 160)
            };

            let clamped_workgroups = workgroups.min(max_workgroups / 4).max(64);
            let _ = gpu_cores; // gpu_cores currently unused but kept for potential future tuning
            clamped_workgroups
        } else {
            // Unknown/Generic GPU - use conservative defaults
            (max_workgroups / 16).max(512)
        };

        log::info!(target: "gpu_engine", "Vendor-specific dispatch configuration:");
        log::info!(target: "gpu_engine", "  Max hardware workgroups: {}", max_workgroups);
        log::info!(target: "gpu_engine", "  Optimal workgroups: {}", optimal_workgroups);

        optimal_workgroups
    }
}
