use pow_core::{hash_from_nonce, mining_midstate, JobContext};
use primitive_types::U512;
use std::time::Instant;

fn u512_to_u32s_le(v: U512) -> [u32; 16] {
    let bytes = v.to_little_endian();
    let mut out = [0u32; 16];
    for i in 0..16 {
        out[i] = u32::from_le_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap());
    }
    out
}

struct Runner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    results: wgpu::Buffer,
    midstate: wgpu::Buffer,
    start_nonce: wgpu::Buffer,
    target: wgpu::Buffer,
    cfg: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl Runner {
    async fn new(trusted: bool) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("no adapter");
        assert!(
            adapter.features().contains(wgpu::Features::SHADER_INT64),
            "SHADER_INT64 required"
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SHADER_INT64,
                ..Default::default()
            })
            .await
            .unwrap();

        let desc = wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../src/mining_u64.wgsl").into()),
        };
        let shader = if trusted {
            unsafe {
                device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
            }
        } else {
            device.create_shader_module(desc)
        };
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("mining_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mk = |size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        Runner {
            pipeline,
            results: mk(132, U::STORAGE | U::COPY_SRC | U::COPY_DST),
            midstate: mk(96, U::STORAGE | U::COPY_DST),
            start_nonce: mk(64, U::STORAGE | U::COPY_DST),
            target: mk(64, U::STORAGE | U::COPY_DST),
            cfg: mk(12, U::STORAGE | U::COPY_DST),
            staging: mk(132, U::MAP_READ | U::COPY_DST),
            device,
            queue,
        }
    }

    fn run_batch(&self, header: [u8; 32], start: U512, batch: u32, target: U512) -> f64 {
        let nonce_be = start.to_big_endian();
        let mid = mining_midstate(header, nonce_be[..32].try_into().unwrap());
        let mut mid_u32 = [0u32; 24];
        for (i, f) in mid.iter().enumerate() {
            mid_u32[2 * i] = *f as u32;
            mid_u32[2 * i + 1] = (*f >> 32) as u32;
        }
        self.queue
            .write_buffer(&self.midstate, 0, bytemuck::cast_slice(&mid_u32));
        self.queue.write_buffer(
            &self.start_nonce,
            0,
            bytemuck::cast_slice(&u512_to_u32s_le(start)),
        );
        self.queue.write_buffer(
            &self.target,
            0,
            bytemuck::cast_slice(&u512_to_u32s_le(target)),
        );
        self.queue
            .write_buffer(&self.cfg, 0, bytemuck::cast_slice(&[batch, 1u32, batch]));
        self.queue.write_buffer(&self.results, 0, &[0u8; 132]);

        let layout = self.pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry<'_>> = [
            (0, &self.results),
            (1, &self.midstate),
            (2, &self.start_nonce),
            (3, &self.target),
            (4, &self.cfg),
        ]
        .iter()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: *i,
            resource: b.as_entire_binding(),
        })
        .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });

        let t0 = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(batch.div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.results, 0, &self.staging, 0, 132);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        loop {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            if rx.try_recv().is_ok() {
                break;
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        self.staging.unmap();
        elapsed
    }

    fn read_results(&self) -> [u32; 33] {
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let mut out = [0u32; 33];
        out.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        self.staging.unmap();
        out
    }
}

fn main() {
    let batch: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(8_000_000);
    let iters = 4u32;
    let rt = tokio::runtime::Runtime::new().unwrap();

    for trusted in [false, true] {
        let label = if trusted {
            "trusted (no checks)"
        } else {
            "checked (default)"
        };
        let runner = rt.block_on(Runner::new(trusted));

        let header = [9u8; 32];
        let ctx = JobContext::new(header, U512::one());
        let start = (U512::from(7u64) << 300) | U512::from(123456789u64);
        runner.run_batch(header, start, 256, ctx.target);
        let r = runner.read_results();
        assert_eq!(r[0], 1, "{label}: no solution with target=MAX");
        let nonce = U512::from_little_endian(bytemuck::cast_slice(&r[1..17]));
        let hash = U512::from_little_endian(bytemuck::cast_slice(&r[17..33]));
        assert_eq!(hash, hash_from_nonce(&ctx, nonce), "{label}: hash mismatch");
        println!("{label}: correctness OK");

        let header = [42u8; 32];
        let start = U512::from(1u64) << 200;
        let target = U512::one();
        runner.run_batch(header, start, batch, target);
        let mut total = 0.0;
        for i in 0..iters {
            total += runner.run_batch(
                header,
                start + U512::from((i as u64 + 1) * batch as u64),
                batch,
                target,
            );
        }
        let mhs = (batch as f64 * iters as f64) / total / 1e6;
        println!("{label}: {batch} nonces x{iters} in {total:.3}s = {mhs:.3} MH/s");
    }
}
