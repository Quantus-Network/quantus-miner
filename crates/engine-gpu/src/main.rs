use futures::executor::block_on;

mod end_to_end_tests;
mod tests;

fn main() {
    block_on(run()).unwrap();
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Setup GPU device and queue
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL, // Force Metal on Apple
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    println!("Running Poseidon2 GPU Component Tests...\n");

    // Run all component tests
    if let Err(e) = tests::test_gf_from_const(&device, &queue).await {
        eprintln!("❌ gf_from_const tests failed: {}", e);
    }

    if let Err(e) = tests::test_gf_mul(&device, &queue).await {
        eprintln!("❌ gf_mul tests failed: {}", e);
    }

    if let Err(e) = tests::test_sbox(&device, &queue).await {
        eprintln!("❌ S-box tests failed: {}", e);
    }

    if let Err(e) = tests::test_mds_matrix(&device, &queue).await {
        eprintln!("❌ MDS matrix tests failed: {}", e);
    }

    if let Err(e) = tests::test_internal_linear_layer(&device, &queue).await {
        eprintln!("❌ Internal linear layer tests failed: {}", e);
    }

    if let Err(e) = tests::test_external_linear_layer(&device, &queue).await {
        eprintln!("❌ External linear layer tests failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_initial_external_rounds(&device, &queue).await {
        eprintln!("❌ Initial external rounds tests failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_terminal_external_rounds(&device, &queue).await {
        eprintln!("❌ Terminal external rounds tests failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_constants_verification(&device, &queue).await {
        eprintln!("❌ Constants verification test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_internal_constants_verification(&device, &queue).await {
        eprintln!("❌ Internal constants verification test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_internal_rounds_only(&device, &queue).await {
        eprintln!("❌ Internal rounds only test failed: {}", e);
    }

    if let Err(e) =
        tests::test_poseidon2_terminal_external_constants_verification(&device, &queue).await
    {
        eprintln!(
            "❌ Terminal external constants verification test failed: {}",
            e
        );
    }

    if let Err(e) = tests::test_poseidon2_permutation(&device, &queue).await {
        eprintln!("❌ Poseidon2 permutation tests failed: {}", e);
    }

    if let Err(e) = tests::test_bytes_to_field_elements(&device, &queue).await {
        eprintln!("❌ Bytes to field elements tests failed: {}", e);
    }

    if let Err(e) = tests::test_field_elements_to_bytes(&device, &queue).await {
        eprintln!("❌ Field elements to bytes tests failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_squeeze_twice(&device, &queue).await {
        eprintln!("❌ Poseidon2 squeeze-twice tests failed: {}", e);
    }

    if let Err(e) = tests::test_double_hash(&device, &queue).await {
        eprintln!("❌ Double hash tests failed: {}", e);
    }

    if let Err(e) = end_to_end_tests::test_end_to_end_mining(&device, &queue).await {
        eprintln!("❌ End-to-end mining test failed: {}", e);
    }

    println!("\nAll tests completed!");
    if let Err(e) = end_to_end_tests::test_end_to_end_mining(&device, &queue).await {
        eprintln!("❌ End-to-end mining test failed: {}", e);
    }

    println!("\nAll tests completed!");

    // generate_correct_wgsl_constants();
    Ok(())
}

#[allow(dead_code)]
fn generate_correct_wgsl_constants() {
    use qp_poseidon_constants::*;

    println!("🔧 Generating correct WGSL constants...");

    println!("// Initial external round constants (4 rounds x 12 elements)");
    println!("const INITIAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(");

    for (round_idx, round) in POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW.iter().enumerate() {
        println!("    array<array<u32, 2>, 12>(");
        for (elem_idx, &value) in round.iter().enumerate() {
            let low = value as u32;
            let high = (value >> 32) as u32;
            if elem_idx == 11 {
                println!("        array<u32, 2>({}u, {}u)", low, high);
            } else {
                println!("        array<u32, 2>({}u, {}u),", low, high);
            }
        }
        if round_idx == 3 {
            println!("    )");
        } else {
            println!("    ),");
        }
    }
    println!(");");

    println!("\n// Terminal external round constants (4 rounds x 12 elements)");
    println!("const TERMINAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(");

    for (round_idx, round) in POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW.iter().enumerate() {
        println!("    array<array<u32, 2>, 12>(");
        for (elem_idx, &value) in round.iter().enumerate() {
            let low = value as u32;
            let high = (value >> 32) as u32;
            if elem_idx == 11 {
                println!("        array<u32, 2>({}u, {}u)", low, high);
            } else {
                println!("        array<u32, 2>({}u, {}u),", low, high);
            }
        }
        if round_idx == 3 {
            println!("    )");
        } else {
            println!("    ),");
        }
    }
    println!(");");

    println!("\n// Internal round constants (22 values)");
    println!("const INTERNAL_CONSTANTS: array<array<u32, 2>, 22> = array<array<u32, 2>, 22>(");
    for (idx, &value) in POSEIDON2_INTERNAL_CONSTANTS_RAW.iter().enumerate() {
        let low = value as u32;
        let high = (value >> 32) as u32;
        if idx == 21 {
            println!("    array<u32, 2>({}u, {}u)", low, high);
        } else {
            println!("    array<u32, 2>({}u, {}u),", low, high);
        }
    }
    println!(");");
}
