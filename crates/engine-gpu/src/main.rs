mod end_to_end_tests;
mod tests;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("no GPU adapter");

    let mut failures = 0usize;

    {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;
        failures += run_suite(
            &device,
            &queue,
            include_str!("mining.wgsl"),
            "32-bit (mining.wgsl)",
        )
        .await;
    }

    if adapter.features().contains(wgpu::Features::SHADER_INT64) {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("u64 Test Device"),
                required_features: wgpu::Features::SHADER_INT64,
                ..Default::default()
            })
            .await?;
        failures += run_suite(
            &device,
            &queue,
            include_str!("mining_u64.wgsl"),
            "native-u64 (mining_u64.wgsl)",
        )
        .await;
    } else {
        println!("\nSHADER_INT64 not supported on this adapter; skipping mining_u64.wgsl suite");
    }

    if failures > 0 {
        return Err(format!("{failures} test group(s) failed").into());
    }
    println!("\nAll tests completed!");
    Ok(())
}

async fn run_suite(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_src: &str,
    label: &str,
) -> usize {
    println!("\n==== Running Poseidon2 GPU tests against {label} ====\n");
    let mut failures = 0usize;
    macro_rules! run {
        ($name:literal, $fut:expr) => {
            if let Err(e) = $fut.await {
                eprintln!("❌ {} failed: {}", $name, e);
                failures += 1;
            }
        };
    }

    run!(
        "gf_from_const",
        tests::test_gf_from_const(device, queue, shader_src)
    );
    run!("gf_mul", tests::test_gf_mul(device, queue, shader_src));
    run!("sbox", tests::test_sbox(device, queue, shader_src));
    run!(
        "mds matrix",
        tests::test_mds_matrix(device, queue, shader_src)
    );
    run!(
        "internal linear layer",
        tests::test_internal_linear_layer(device, queue, shader_src)
    );
    run!(
        "external linear layer",
        tests::test_external_linear_layer(device, queue, shader_src)
    );
    run!(
        "initial external rounds",
        tests::test_poseidon2_initial_external_rounds(device, queue, shader_src)
    );
    run!(
        "terminal external rounds",
        tests::test_poseidon2_terminal_external_rounds(device, queue, shader_src)
    );
    run!(
        "constants verification",
        tests::test_poseidon2_constants_verification(device, queue, shader_src)
    );
    run!(
        "internal constants verification",
        tests::test_poseidon2_internal_constants_verification(device, queue, shader_src)
    );
    run!(
        "internal rounds only",
        tests::test_poseidon2_internal_rounds_only(device, queue, shader_src)
    );
    run!(
        "terminal external constants verification",
        tests::test_poseidon2_terminal_external_constants_verification(device, queue, shader_src)
    );
    run!(
        "poseidon2 permutation",
        tests::test_poseidon2_permutation(device, queue, shader_src)
    );
    run!(
        "bytes to field elements",
        tests::test_bytes_to_field_elements(device, queue, shader_src)
    );
    run!(
        "field elements to bytes",
        tests::test_field_elements_to_bytes(device, queue, shader_src)
    );
    run!(
        "poseidon2 squeeze-twice",
        tests::test_poseidon2_squeeze_twice(device, queue, shader_src)
    );
    run!(
        "hash squeeze twice",
        tests::test_hash_squeeze_twice(device, queue, shader_src)
    );
    run!(
        "end-to-end mining",
        end_to_end_tests::test_end_to_end_mining(device, queue, shader_src)
    );

    failures
}
