//! Poseidon2 mining kernels.
//!
//! Same `mining_main` binding numbers; must stay bit-exact with `pow_core`.
//! Apple uses uniform bindings for inputs 1..4; other kernels use storage.
//! Input word layouts are identical, with dispatch config padded to 16 bytes.
//!
//! - Apple Metal + `SHADER_INT64` → Apple Metal u64 (`mining_u64_apple.wgsl`)
//! - other GPUs + `SHADER_INT64` → native u64 (`mining_u64.wgsl`)
//! - no `SHADER_INT64` → 32-bit fallback (`mining.wgsl`)

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kernel {
    U32,
    Default,
    Apple,
}

impl Kernel {
    pub const fn needs_int64(self) -> bool {
        !matches!(self, Self::U32)
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::Default => "u64",
            Self::Apple => "u64-apple",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::U32 => "32-bit",
            Self::Default => "native-u64",
            Self::Apple => "native-u64 Apple Metal",
        }
    }

    pub const fn source(self) -> &'static str {
        match self {
            Self::U32 => include_str!("mining.wgsl"),
            Self::Default => include_str!("mining_u64.wgsl"),
            Self::Apple => include_str!("mining_u64_apple.wgsl"),
        }
    }

    pub fn for_adapter(adapter: &wgpu::Adapter) -> Self {
        Self::for_adapter_info(&adapter.get_info(), adapter.features())
    }

    pub fn for_adapter_info(info: &wgpu::AdapterInfo, features: wgpu::Features) -> Self {
        if !features.contains(wgpu::Features::SHADER_INT64) {
            return Self::U32;
        }
        if info.backend == wgpu::Backend::Metal {
            Self::Apple
        } else {
            Self::Default
        }
    }

    pub const fn all() -> &'static [Self] {
        &[Self::U32, Self::Default, Self::Apple]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(backend: wgpu::Backend) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "test".into(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend,
        }
    }

    #[test]
    fn metal_with_int64_selects_apple() {
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Metal), wgpu::Features::SHADER_INT64),
            Kernel::Apple
        );
    }

    #[test]
    fn vulkan_with_int64_selects_default() {
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Vulkan), wgpu::Features::SHADER_INT64),
            Kernel::Default
        );
    }

    #[test]
    fn dx12_with_int64_selects_default() {
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Dx12), wgpu::Features::SHADER_INT64),
            Kernel::Default
        );
    }

    #[test]
    fn no_int64_falls_back_to_u32() {
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Metal), wgpu::Features::empty()),
            Kernel::U32
        );
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Vulkan), wgpu::Features::empty()),
            Kernel::U32
        );
    }
}
