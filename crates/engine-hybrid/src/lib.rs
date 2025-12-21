#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

//! Hybrid CPU+GPU mining engine for the Quantus External Miner.
//!
//! This crate provides a `HybridEngine` that combines CPU and GPU mining
//! engines by routing different worker threads to different engine types.
//! This avoids concurrent GPU operations while still utilizing both CPU and GPU.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use pow_core::JobContext;
use primitive_types::U512;

// Re-export common types from engine-cpu for compatibility
pub use engine_cpu::{Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};

/// Configuration for hybrid mining engine worker allocation.
#[derive(Debug, Clone, Default)]
pub struct HybridConfig {
    /// Number of CPU workers to use (None = auto-detect)
    pub cpu_workers: Option<usize>,
    /// Number of GPU workers to use (None = auto-detect)
    pub gpu_workers: Option<usize>,
}

impl HybridConfig {
    /// Create a new hybrid configuration with specified worker counts.
    pub fn new(cpu_workers: Option<usize>, gpu_workers: Option<usize>) -> Self {
        Self {
            cpu_workers,
            gpu_workers,
        }
    }

    /// Get CPU worker count (with default if not set).
    pub fn cpu_workers(&self) -> usize {
        self.cpu_workers.unwrap_or_else(num_cpus::get)
    }

    /// Get GPU worker count (with default if not set).
    pub fn gpu_workers(&self) -> usize {
        self.gpu_workers.unwrap_or({
            #[cfg(feature = "gpu")]
            {
                // Auto-detect GPU device count by creating a temporary GPU engine
                log::debug!("🔍 HybridConfig::gpu_workers() auto-detection called");
                let gpu_engine = engine_gpu::GpuEngine::new();
                let device_count = gpu_engine.device_count();
                log::debug!("🔍 Auto-detected {} GPU device(s)", device_count);
                device_count
            }
            #[cfg(not(feature = "gpu"))]
            {
                0
            }
        })
    }

    /// Check if this configuration would result in any workers.
    pub fn has_workers(&self) -> bool {
        self.cpu_workers() > 0 || self.gpu_workers() > 0
    }
}

/// Hybrid mining engine that routes threads to CPU or GPU engines.
///
/// This engine maintains separate CPU and GPU engines and routes incoming
/// search_range calls to the appropriate engine based on a round-robin
/// thread assignment strategy. This ensures no concurrent GPU operations
/// while still utilizing both CPU and GPU resources.
pub struct HybridEngine {
    config: HybridConfig,
    cpu_engine: Arc<dyn MinerEngine>,
    #[cfg(feature = "gpu")]
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    thread_counter: Arc<AtomicUsize>,
}

impl HybridEngine {
    /// Create a hybrid engine with the specified configuration.
    pub fn new(config: HybridConfig) -> anyhow::Result<Self> {
        if !config.has_workers() {
            return Err(anyhow::anyhow!(
                "Hybrid engine requires at least one CPU or GPU worker"
            ));
        }

        let cpu_workers = config.cpu_workers();
        let gpu_workers = config.gpu_workers();

        log::info!(
            "🔀 Initializing hybrid engine: {} CPU workers, {} GPU workers",
            cpu_workers,
            gpu_workers
        );

        let engine = Self {
            config,
            cpu_engine: Arc::new(engine_cpu::FastCpuEngine::new()),
            #[cfg(feature = "gpu")]
            gpu_engine: if gpu_workers > 0 {
                log::info!("🎮 Creating GPU engine for hybrid mining...");
                let gpu_engine = engine_gpu::GpuEngine::new();
                let detected_devices = gpu_engine.device_count();
                log::info!(
                    "🎮 Created GPU engine: {} devices detected, {} workers configured",
                    detected_devices,
                    gpu_workers
                );
                Some(Arc::new(gpu_engine))
            } else {
                None
            },
            thread_counter: Arc::new(AtomicUsize::new(0)),
        };

        Ok(engine)
    }

    /// Create a hybrid engine with simple worker counts.
    pub fn with_workers(
        cpu_workers: Option<usize>,
        gpu_workers: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::new(HybridConfig::new(cpu_workers, gpu_workers))
    }

    /// Get the total number of workers across all engines.
    pub fn total_workers(&self) -> usize {
        self.config.cpu_workers() + self.config.gpu_workers()
    }

    /// Get the configuration for this hybrid engine.
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Route a thread to the appropriate engine based on worker allocation.
    fn route_to_engine(&self) -> RouteDecision {
        let cpu_workers = self.config.cpu_workers();
        let gpu_workers = self.config.gpu_workers();

        // If only one engine type is available, route directly
        if gpu_workers == 0 {
            return RouteDecision::Cpu;
        }
        if cpu_workers == 0 {
            return RouteDecision::Gpu;
        }

        // For hybrid mode, use round-robin assignment
        let thread_id = self.thread_counter.fetch_add(1, Ordering::Relaxed);
        let total_workers = cpu_workers + gpu_workers;
        let worker_slot = thread_id % total_workers;

        if worker_slot < cpu_workers {
            RouteDecision::Cpu
        } else {
            RouteDecision::Gpu
        }
    }
}

#[derive(Debug)]
enum RouteDecision {
    Cpu,
    Gpu,
}

impl MinerEngine for HybridEngine {
    fn name(&self) -> &'static str {
        "hybrid"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        // Use the same context preparation as the underlying engines
        JobContext::new(header_hash, difficulty)
    }

    fn search_range(
        &self,
        ctx: &JobContext,
        range: Range,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> EngineStatus {
        let route = self.route_to_engine();
        let thread_id = thread::current().id();

        match route {
            RouteDecision::Cpu => {
                log::debug!("🖥️  Thread {:?} routed to CPU engine", thread_id);
                self.cpu_engine.search_range(ctx, range, cancel)
            }
            RouteDecision::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    if let Some(ref gpu_engine) = self.gpu_engine {
                        log::debug!("🎮 Thread {:?} routed to GPU engine", thread_id);
                        return gpu_engine.search_range(ctx, range, cancel);
                    }
                }

                // Fallback to CPU if GPU engine is not available
                log::debug!(
                    "🖥️  Thread {:?} fallback to CPU engine (GPU unavailable)",
                    thread_id
                );
                self.cpu_engine.search_range(ctx, range, cancel)
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert!(config.cpu_workers.is_none());
        assert!(config.gpu_workers.is_none());
    }

    #[test]
    fn test_hybrid_config_with_workers() {
        let config = HybridConfig::new(Some(4), Some(2));
        assert_eq!(config.cpu_workers(), 4);
        assert_eq!(config.gpu_workers(), 2);
        assert!(config.has_workers());
    }

    #[test]
    fn test_hybrid_engine_creation() {
        let config = HybridConfig::new(Some(2), Some(0));
        let engine = HybridEngine::new(config).expect("Failed to create hybrid engine");
        assert_eq!(engine.total_workers(), 2);
    }

    #[test]
    fn test_thread_routing_cpu_only() {
        let config = HybridConfig::new(Some(4), Some(0));
        let engine = HybridEngine::new(config).expect("Failed to create hybrid engine");

        // All routes should go to CPU
        for _ in 0..10 {
            match engine.route_to_engine() {
                RouteDecision::Cpu => {}
                RouteDecision::Gpu => panic!("Should not route to GPU when gpu_workers = 0"),
            }
        }
    }

    #[test]
    fn test_thread_routing_hybrid() {
        let config = HybridConfig::new(Some(2), Some(1));
        let engine = HybridEngine::new(config).expect("Failed to create hybrid engine");

        let mut cpu_count = 0;
        let mut gpu_count = 0;

        // Test routing pattern
        for _ in 0..6 {
            match engine.route_to_engine() {
                RouteDecision::Cpu => cpu_count += 1,
                RouteDecision::Gpu => gpu_count += 1,
            }
        }

        // Should get 4 CPU routes and 2 GPU routes (2:1 ratio)
        assert_eq!(cpu_count, 4);
        assert_eq!(gpu_count, 2);
    }
}
