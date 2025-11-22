use bytemuck;
use futures::executor::block_on;
use qp_plonky2_field::goldilocks_field::GoldilocksField;
use qp_plonky2_field::types::{Field, PrimeField64};

use wgpu::{self, util::DeviceExt};

mod tests;

fn u8_array_to_u32_array(bytes: &[u8; 96]) -> [u32; 24] {
    let mut result = [0u32; 24];
    for i in 0..24 {
        let idx = i * 4;
        result[i] =
            u32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
    }
    result
}

fn u8_array_to_u32_array_64(bytes: &[u8; 64]) -> [u32; 16] {
    let mut result = [0u32; 16];
    for i in 0..16 {
        let idx = i * 4;
        result[i] =
            u32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
    }
    result
}

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

    if let Err(e) = tests::test_poseidon2_first_round_debug(&device, &queue).await {
        eprintln!("❌ First round debug failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_initial_external_rounds(&device, &queue).await {
        eprintln!("❌ Initial external rounds test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_constants_verification(&device, &queue).await {
        eprintln!("❌ Constants verification test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_rounds_0_and_1_debug(&device, &queue).await {
        eprintln!("❌ Rounds 0-1 debug test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_internal_constants_verification(&device, &queue).await {
        eprintln!("❌ Internal constants verification test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_internal_linear_layer_debug(&device, &queue).await {
        eprintln!("❌ Internal linear layer debug test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_first_internal_rounds_debug(&device, &queue).await {
        eprintln!("❌ First internal rounds debug test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_terminal_external_rounds_debug(&device, &queue).await {
        eprintln!("❌ Terminal external rounds debug test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_internal_rounds(&device, &queue).await {
        eprintln!("❌ Internal rounds test failed: {}", e);
    }

    if let Err(e) = tests::test_poseidon2_sequential_cpu_vs_gpu_internal(&device, &queue).await {
        eprintln!("❌ Sequential CPU vs GPU internal test failed: {}", e);
    }

    if let Err(e) =
        tests::test_poseidon2_terminal_external_constants_verification(&device, &queue).await
    {
        eprintln!(
            "❌ Terminal external constants verification test failed: {}",
            e
        );
    }

    if let Err(e) = tests::test_poseidon2_sequential_step_by_step_debug(&device, &queue).await {
        eprintln!("❌ Sequential step by step debug test failed: {}", e);
    }

    println!("🎯 Testing Poseidon2 permutation with proper conversion...");

    if let Err(e) = tests::test_poseidon2_permutation(&device, &queue).await {
        eprintln!("❌ Poseidon2 permutation tests failed: {}", e);
    }

    println!("\nAll tests completed!");
    Ok(())
}

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
