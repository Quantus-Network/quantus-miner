fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    for adapter in instance.enumerate_adapters(wgpu::Backends::PRIMARY) {
        let info = adapter.get_info();
        let features = adapter.features();
        println!("{} ({:?}, {:?})", info.name, info.device_type, info.backend);
        println!(
            "  SHADER_INT64: {}",
            features.contains(wgpu::Features::SHADER_INT64)
        );
        println!(
            "  SUBGROUP: {}",
            features.contains(wgpu::Features::SUBGROUP)
        );
        println!(
            "  TIMESTAMP_QUERY: {}",
            features.contains(wgpu::Features::TIMESTAMP_QUERY)
        );
        let limits = adapter.limits();
        println!(
            "  max_workgroup_size_x: {}, max_invocations: {}, max_workgroups_per_dim: {}",
            limits.max_compute_workgroup_size_x,
            limits.max_compute_invocations_per_workgroup,
            limits.max_compute_workgroups_per_dimension
        );
    }
}
