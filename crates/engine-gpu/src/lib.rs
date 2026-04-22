#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

use engine_cpu::{CancelCheck, Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use futures::executor::block_on;
use pow_core::{format_hashrate, format_u512, JobContext};
use primitive_types::U512;
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
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
    batch_size: u64,
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

        // Dispatch config: [total_threads, nonces_per_thread, total_nonces] = 3 u32s
        let dispatch_config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dispatch Config Buffer"),
            size: 12,
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

impl GpuEngine {
    /// Try to initialize the GPU engine with the given batch size.
    pub fn try_new(batch_size: u64) -> Result<Self, Box<dyn std::error::Error>> {
        block_on(Self::init(batch_size))
    }

    async fn init(batch_size: u64) -> Result<Self, Box<dyn std::error::Error>> {
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
        let mut adapter_infos = Vec::new();
        for (i, adapter) in adapters.into_iter().enumerate() {
            let info = adapter.get_info();
            log::debug!(target: "gpu_engine", "Adapter {} raw info: {:?}", i, info);
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

            // Log device limits at debug level
            let limits = device.limits();
            log::debug!(target: "gpu_engine", "Adapter {} limits: max_workgroups={}, max_workgroup_size={}x{}x{}, max_buffer={}",
                i,
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
                limits.max_buffer_size
            );

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

        log::info!(
            target: "gpu_engine",
            "GPU engine initialized with {} devices (batch size: {} nonces)",
            contexts.len(),
            batch_size
        );

        Ok(Self {
            contexts,
            device_counter: AtomicUsize::new(0),
            batch_size,
        })
    }

    /// Returns the number of GPU devices available
    pub fn device_count(&self) -> usize {
        self.contexts.len()
    }

    /// Explicitly clear thread-local GPU resources.
    /// Call this before thread exit to avoid TLS destruction order issues with wgpu.
    pub fn clear_worker_resources() {
        WORKER_RESOURCES.with(|resources| {
            *resources.borrow_mut() = None;
        });
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

    fn search_range(
        &self,
        ctx: &JobContext,
        range: Range,
        cancel: &dyn CancelCheck,
    ) -> EngineStatus {
        if self.contexts.is_empty() {
            log::warn!(target: "gpu_engine", "No GPUs available for search.");
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        // Empty or inverted range: nothing to do.
        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        // Check for pre-cancellation
        if cancel.is_cancelled() {
            return EngineStatus::Cancelled { hash_count: 0 };
        }

        // Use thread-local assignment for consistent worker-to-GPU mapping
        let device_index = ASSIGNED_GPU_DEVICE.with(|assigned| {
            let mut assigned_ref = assigned.borrow_mut();
            if let Some(index) = *assigned_ref {
                index
            } else {
                let index = if self.contexts.len() == 1 {
                    0
                } else {
                    self.device_counter.fetch_add(1, Ordering::SeqCst) % self.contexts.len()
                };
                *assigned_ref = Some(index);
                log::info!(
                    target: "gpu_engine",
                    "Worker thread assigned to GPU device {} (of {} total devices)",
                    index,
                    self.contexts.len()
                );
                index
            }
        });

        let gpu_ctx = &self.contexts[device_index];

        // Ensure resources are initialized for this thread
        WORKER_RESOURCES.with(|resources_cell| {
            let mut resources = resources_cell.borrow_mut();
            if resources.is_none() {
                *resources = Some(gpu_ctx.create_resources());
            }
        });

        let resources = WORKER_RESOURCES
            .with(|resources_cell| resources_cell.borrow().as_ref().unwrap().clone());

        // Pre-convert header and target (only needs to be done once per job)
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

        let search_start = std::time::Instant::now();
        let mut total_hashes: u64 = 0;
        let mut current_start = range.start;
        let mut batch_num = 0u64;

        log::info!(
            target: "gpu_engine",
            "GPU {} search started: range {}..{}, batch size: {} nonces",
            device_index,
            format_u512(range.start),
            format_u512(range.end),
            self.batch_size
        );

        // Process in batches, checking for cancellation between each batch
        while current_start <= range.end {
            // Check for cancellation at host level BEFORE starting each batch
            if cancel.is_cancelled() {
                let elapsed = search_start.elapsed();
                let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();
                log::info!(
                    target: "gpu_engine",
                    "GPU {} cancelled before batch {} ({} total hashes in {:.2}s, {})",
                    device_index,
                    batch_num,
                    total_hashes,
                    elapsed.as_secs_f64(),
                    format_hashrate(hash_rate)
                );
                return EngineStatus::Cancelled {
                    hash_count: total_hashes,
                };
            }

            // Calculate batch range
            let remaining = range
                .end
                .saturating_sub(current_start)
                .saturating_add(U512::one());
            let batch_size_u512 = U512::from(self.batch_size);
            let this_batch_size = if remaining > batch_size_u512 {
                self.batch_size
            } else {
                remaining.as_u64()
            };

            // Run single batch
            let batch_result =
                self.run_single_batch(gpu_ctx, &resources, current_start, this_batch_size);

            match batch_result {
                BatchResult::Found {
                    candidate,
                    hash_count,
                } => {
                    total_hashes += hash_count;
                    let elapsed = search_start.elapsed();
                    let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();

                    log::debug!(
                        target: "gpu_engine",
                        "GPU {} found solution in batch {}! Nonce: {}, Hash: {} ({} total hashes in {:.2}s, {})",
                        device_index,
                        batch_num,
                        format_u512(candidate.nonce),
                        format_u512(candidate.hash),
                        total_hashes,
                        elapsed.as_secs_f64(),
                        format_hashrate(hash_rate)
                    );

                    return EngineStatus::Found {
                        candidate,
                        hash_count: total_hashes,
                        origin: FoundOrigin::GpuG1,
                    };
                }
                BatchResult::NotFound { hash_count } => {
                    total_hashes += hash_count;
                }
            }

            // Move to next batch
            current_start = current_start.saturating_add(U512::from(this_batch_size));
            batch_num += 1;

            // Log progress periodically (every 10 batches)
            if batch_num.is_multiple_of(10) {
                let elapsed = search_start.elapsed();
                let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();
                log::debug!(
                    target: "gpu_engine",
                    "GPU {} batch {} complete: {} hashes so far ({:.2}s, {})",
                    device_index,
                    batch_num,
                    total_hashes,
                    elapsed.as_secs_f64(),
                    format_hashrate(hash_rate)
                );
            }
        }

        // Range exhausted without finding solution
        let elapsed = search_start.elapsed();
        let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();
        log::info!(
            target: "gpu_engine",
            "GPU {} search exhausted: {} hashes in {} batches ({:.2}s, {})",
            device_index,
            total_hashes,
            batch_num,
            elapsed.as_secs_f64(),
            format_hashrate(hash_rate)
        );

        EngineStatus::Exhausted {
            hash_count: total_hashes,
        }
    }
}

/// Result from a single GPU batch
enum BatchResult {
    Found {
        candidate: Candidate,
        hash_count: u64,
    },
    NotFound {
        hash_count: u64,
    },
}

impl GpuEngine {
    /// Run a single batch of GPU computation
    fn run_single_batch(
        &self,
        gpu_ctx: &GpuContext,
        resources: &GpuResources,
        batch_start: U512,
        batch_size: u64,
    ) -> BatchResult {
        // Calculate dispatch configuration for this batch
        let threads_per_workgroup = 256u32;
        let limits = gpu_ctx.device.limits();
        let max_workgroups = limits.max_compute_workgroups_per_dimension;

        let hinted_workgroups = gpu_ctx.optimal_workgroups.max(1).min(max_workgroups);
        let hinted_threads = hinted_workgroups as u64 * threads_per_workgroup as u64;

        let logical_threads = batch_size.min(hinted_threads).max(1);
        let num_workgroups = ((logical_threads as u32).div_ceil(threads_per_workgroup)).max(1);
        let total_threads = (num_workgroups * threads_per_workgroup) as u64;
        let nonces_per_thread = (batch_size.div_ceil(total_threads)).max(1) as u32;

        // Dispatch config: [total_threads, nonces_per_thread, total_nonces]
        let dispatch_config = [total_threads as u32, nonces_per_thread, batch_size as u32];

        // Write dispatch config
        gpu_ctx.queue.write_buffer(
            &resources.dispatch_config_buffer,
            0,
            bytemuck::cast_slice(&dispatch_config),
        );

        // Write start nonce for this batch
        let start_nonce_bytes = batch_start.to_little_endian();
        gpu_ctx
            .queue
            .write_buffer(&resources.start_nonce_buffer, 0, &start_nonce_bytes);

        // Reset results buffer
        const RESULTS_SIZE: usize = (1 + 16 + 16) * 4;
        const ZEROS: [u8; RESULTS_SIZE] = [0; RESULTS_SIZE];
        gpu_ctx
            .queue
            .write_buffer(&resources.results_buffer, 0, &ZEROS);

        // Create and submit command buffer
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

        gpu_ctx.queue.submit(Some(encoder.finish()));

        // Wait for GPU to complete (blocking)
        let buffer_slice = resources.staging_buffer.slice(..);
        let mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mapped_clone = mapped.clone();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                mapped_clone.store(true, Ordering::Release);
            }
        });

        // Poll until complete
        loop {
            let _ = gpu_ctx.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(10)),
            });

            if mapped.load(Ordering::Acquire) {
                break;
            }
        }

        // Read results
        let data = buffer_slice.get_mapped_range();
        let result_u32s: &[u32] = bytemuck::cast_slice(&data);

        // Calculate the actual number of nonces dispatched
        let dispatched_nonces = (total_threads * nonces_per_thread as u64).min(batch_size);

        if result_u32s[0] != 0 {
            // Solution found!
            let nonce_u32s = &result_u32s[1..17];
            let hash_u32s = &result_u32s[17..33];
            let nonce = U512::from_little_endian(bytemuck::cast_slice(nonce_u32s));
            let hash = U512::from_little_endian(bytemuck::cast_slice(hash_u32s));
            let work = nonce.to_big_endian();

            // Calculate hashes computed based on GPU parallel execution model
            let hashes_computed = if nonce >= batch_start {
                let logical_index = (nonce - batch_start).as_u64();
                let winning_iteration = logical_index % (nonces_per_thread as u64);
                (total_threads * (winning_iteration + 1)).min(dispatched_nonces)
            } else {
                dispatched_nonces
            };

            drop(data);
            resources.staging_buffer.unmap();

            return BatchResult::Found {
                candidate: Candidate { nonce, work, hash },
                hash_count: hashes_computed,
            };
        }

        drop(data);
        resources.staging_buffer.unmap();

        BatchResult::NotFound {
            hash_count: dispatched_nonces,
        }
    }

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
        // Returns (workgroups, tier_name, is_fallback)
        let (optimal_workgroups, tier, is_fallback) =
            if vendor_name.contains("nvidia") || adapter_info.vendor == 4318 {
                // NVIDIA GPUs (vendor ID 0x10DE = 4318)
                if vendor_name.contains("5090") || vendor_name.contains("5080") {
                    (
                        (max_workgroups / 6).max(5120),
                        "NVIDIA RTX 50 Flagship (Blackwell)",
                        false,
                    )
                } else if vendor_name.contains("5070")
                    || vendor_name.contains("5060")
                    || vendor_name.contains("rtx 50")
                {
                    (
                        (max_workgroups / 7).max(4608),
                        "NVIDIA RTX 50 (Blackwell)",
                        false,
                    )
                } else if vendor_name.contains("4090") || vendor_name.contains("4080") {
                    (
                        (max_workgroups / 8).max(4096),
                        "NVIDIA RTX 40 Flagship (Ada)",
                        false,
                    )
                } else if vendor_name.contains("rtx 40")
                    || vendor_name.contains("4070")
                    || vendor_name.contains("4060")
                {
                    (
                        (max_workgroups / 10).max(3072),
                        "NVIDIA RTX 40 (Ada)",
                        false,
                    )
                } else if vendor_name.contains("rtx 30")
                    || vendor_name.contains("rtx 20")
                    || vendor_name.contains("3090")
                    || vendor_name.contains("3080")
                    || vendor_name.contains("3070")
                    || vendor_name.contains("2080")
                    || vendor_name.contains("2070")
                    || vendor_name.contains("2060")
                {
                    (
                        (max_workgroups / 12).max(2048),
                        "NVIDIA RTX 30/20 (Ampere/Turing)",
                        false,
                    )
                } else if vendor_name.contains("gtx 16")
                    || vendor_name.contains("gtx 10")
                    || vendor_name.contains("1660")
                    || vendor_name.contains("1650")
                    || vendor_name.contains("1080")
                    || vendor_name.contains("1070")
                    || vendor_name.contains("1060")
                {
                    (
                        (max_workgroups / 16).max(1024),
                        "NVIDIA GTX 16/10 (Turing/Pascal)",
                        false,
                    )
                } else if vendor_name.contains("gtx") {
                    ((max_workgroups / 18).max(768), "NVIDIA GTX (Legacy)", false)
                } else if vendor_name.contains("quadro")
                    || vendor_name.contains("rtx a")
                    || vendor_name.contains("tesla")
                {
                    (
                        (max_workgroups / 10).max(2560),
                        "NVIDIA Quadro/Professional",
                        false,
                    )
                } else {
                    ((max_workgroups / 20).max(512), "NVIDIA Unknown", true)
                }
            } else if vendor_name.contains("amd")
                || vendor_name.contains("radeon")
                || adapter_info.vendor == 4098
            {
                // AMD GPUs (vendor ID 0x1002 = 4098)
                if vendor_name.contains("rx 9")
                    || vendor_name.contains("9070")
                    || vendor_name.contains("9080")
                {
                    (
                        (max_workgroups / 8).max(4096),
                        "AMD RX 9000 (RDNA 4)",
                        false,
                    )
                } else if vendor_name.contains("7900") {
                    (
                        (max_workgroups / 9).max(3584),
                        "AMD RX 7900 (RDNA 3 Flagship)",
                        false,
                    )
                } else if vendor_name.contains("rx 7")
                    || vendor_name.contains("7800")
                    || vendor_name.contains("7700")
                    || vendor_name.contains("7600")
                {
                    (
                        (max_workgroups / 10).max(3072),
                        "AMD RX 7000 (RDNA 3)",
                        false,
                    )
                } else if vendor_name.contains("6900") || vendor_name.contains("6800") {
                    (
                        (max_workgroups / 12).max(2560),
                        "AMD RX 6900/6800 (RDNA 2 Flagship)",
                        false,
                    )
                } else if vendor_name.contains("rx 6")
                    || vendor_name.contains("6700")
                    || vendor_name.contains("6600")
                {
                    (
                        (max_workgroups / 14).max(2048),
                        "AMD RX 6000 (RDNA 2)",
                        false,
                    )
                } else if vendor_name.contains("5700") {
                    (
                        (max_workgroups / 16).max(1536),
                        "AMD RX 5700 (RDNA 1)",
                        false,
                    )
                } else if vendor_name.contains("rx 5")
                    || vendor_name.contains("5600")
                    || vendor_name.contains("5500")
                {
                    (
                        (max_workgroups / 18).max(1024),
                        "AMD RX 5000 (RDNA 1)",
                        false,
                    )
                } else if vendor_name.contains("rx 4")
                    || vendor_name.contains("580")
                    || vendor_name.contains("570")
                {
                    (
                        (max_workgroups / 20).max(768),
                        "AMD RX 500/400 (Polaris)",
                        false,
                    )
                } else if vendor_name.contains("radeon pro")
                    || vendor_name.contains("instinct")
                    || vendor_name.contains("mi")
                {
                    (
                        (max_workgroups / 10).max(2560),
                        "AMD Radeon Pro/Instinct",
                        false,
                    )
                } else {
                    ((max_workgroups / 24).max(512), "AMD Unknown", true)
                }
            } else if vendor_name.contains("intel") || adapter_info.vendor == 32902 {
                // Intel GPUs (vendor ID 0x8086 = 32902)
                if vendor_name.contains("arc b")
                    || vendor_name.contains("b580")
                    || vendor_name.contains("b570")
                {
                    (
                        (max_workgroups / 10).max(2560),
                        "Intel Arc B-Series (Battlemage)",
                        false,
                    )
                } else if vendor_name.contains("a770") || vendor_name.contains("a750") {
                    (
                        (max_workgroups / 12).max(2048),
                        "Intel Arc A7 (Alchemist)",
                        false,
                    )
                } else if vendor_name.contains("a580")
                    || vendor_name.contains("a380")
                    || vendor_name.contains("arc a5")
                    || vendor_name.contains("arc a3")
                {
                    (
                        (max_workgroups / 16).max(1024),
                        "Intel Arc A5/A3 (Alchemist)",
                        false,
                    )
                } else if vendor_name.contains("a310") {
                    ((max_workgroups / 20).max(512), "Intel Arc A3 Entry", false)
                } else if vendor_name.contains("iris xe") || vendor_name.contains("iris plus") {
                    (
                        (max_workgroups / 24).max(384),
                        "Intel Iris Xe/Plus (Integrated)",
                        false,
                    )
                } else if vendor_name.contains("uhd") || vendor_name.contains("hd graphics") {
                    (
                        (max_workgroups / 28).max(256),
                        "Intel UHD/HD Graphics (Integrated)",
                        false,
                    )
                } else {
                    ((max_workgroups / 24).max(256), "Intel Unknown", true)
                }
            } else if adapter_info.backend == wgpu::Backend::Metal {
                // Apple GPUs (detected by Metal backend)
                let (gpu_cores, workgroups, tier) = if vendor_name.contains("m4 ultra") {
                    (80, 1600, "Apple M4 Ultra")
                } else if vendor_name.contains("m4 max") {
                    (40, 800, "Apple M4 Max")
                } else if vendor_name.contains("m4 pro") {
                    (20, 400, "Apple M4 Pro")
                } else if vendor_name.contains("m4") {
                    (10, 200, "Apple M4")
                } else if vendor_name.contains("m3 ultra") {
                    (76, 1520, "Apple M3 Ultra")
                } else if vendor_name.contains("m3 max") {
                    (40, 800, "Apple M3 Max")
                } else if vendor_name.contains("m3 pro") {
                    (18, 360, "Apple M3 Pro")
                } else if vendor_name.contains("m3") {
                    (10, 200, "Apple M3")
                } else if vendor_name.contains("m2 ultra") {
                    (76, 1520, "Apple M2 Ultra")
                } else if vendor_name.contains("m2 max") {
                    (38, 760, "Apple M2 Max")
                } else if vendor_name.contains("m2 pro") {
                    (19, 380, "Apple M2 Pro")
                } else if vendor_name.contains("m2") {
                    (10, 200, "Apple M2")
                } else if vendor_name.contains("m1 ultra") {
                    (64, 1280, "Apple M1 Ultra")
                } else if vendor_name.contains("m1 max") {
                    (32, 640, "Apple M1 Max")
                } else if vendor_name.contains("m1 pro") {
                    (16, 320, "Apple M1 Pro")
                } else if vendor_name.contains("m1") {
                    (8, 160, "Apple M1")
                } else {
                    (8, 160, "Apple Silicon Unknown")
                };

                let clamped_workgroups = workgroups.min(max_workgroups / 4).max(64);
                let _ = gpu_cores; // gpu_cores currently unused but kept for potential future tuning
                let is_fallback = tier == "Apple Silicon Unknown";
                (clamped_workgroups, tier, is_fallback)
            } else {
                // Unknown/Generic GPU - use conservative defaults
                ((max_workgroups / 16).max(512), "Unknown GPU", true)
            };

        // Log GPU detection result
        log::info!(
            target: "gpu_engine",
            "GPU detected: {} | tier: {} | workgroups: {} (max: {})",
            adapter_info.name,
            tier,
            optimal_workgroups,
            max_workgroups
        );

        if is_fallback {
            log::warn!(
                target: "gpu_engine",
                "GPU not recognized, using fallback config. Please report: name='{}', vendor=0x{:04X}, device={}",
                adapter_info.name,
                adapter_info.vendor,
                adapter_info.device
            );
            log::warn!(target: "gpu_engine", "Report at: https://github.com/Quantus-Network/quantus-miner/issues");
        }

        optimal_workgroups
    }
}
