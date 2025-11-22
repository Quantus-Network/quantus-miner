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

    if let Err(e) = tests::test_poseidon2_permutation(&device, &queue).await {
        eprintln!("❌ Poseidon2 permutation tests failed: {}", e);
    }

    println!("\nAll tests completed!");
    Ok(())
}
