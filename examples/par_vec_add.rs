use std::{sync::Arc, time::Instant};

use gpgpu_test::{DescriptorSet, Framework, GpuBuffer, GpuBufferUsage, Kernel, Program, Shader};
use rayon::prelude::*;

lazy_static::lazy_static! {
    static ref FW: Framework = futures::executor::block_on(Framework::default());
}

const THREADING: usize = 8;

pub fn run_cpu(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let cpu_vec: Vec<u32> = a.iter().zip(b).map(|(a, b)| a + b).collect();
    cpu_vec
}

pub fn run_cpu_par(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let par_cpu_vec: Vec<u32> = a.par_iter().zip(b).map(|(a, b)| a + b).collect();
    par_cpu_vec
}

pub fn run_gpu(
    host_vec_a: Vec<u32>,
    host_vec_b: Vec<u32>,
    size: usize,
    shader: Arc<Shader>,
) -> Vec<u32> {
    let gpu_vec_a = GpuBuffer::from_slice(&FW, host_vec_a.as_slice());
    let gpu_vec_b = GpuBuffer::from_slice(&FW, host_vec_b.as_slice());
    let gpu_vec_mult: GpuBuffer<u32> = GpuBuffer::with_capacity(&FW, size as u64);

    let bindings = DescriptorSet::default() // bindgroup 0
        .bind_buffer(&gpu_vec_a, GpuBufferUsage::ReadOnly) // bindentry 0
        .bind_buffer(&gpu_vec_b, GpuBufferUsage::ReadOnly) // bindentry 1
        .bind_buffer(&gpu_vec_mult, GpuBufferUsage::ReadWrite); // bindentry 2

    let program = Program::new(&shader, "main").add_descriptor(bindings);

    let kernel = Kernel::new(&FW, program);

    kernel.enqueue(size as u32 / THREADING as u32, 1, 1);

    let result_vec_mult = gpu_vec_mult.read_vec_blocking().unwrap();

    result_vec_mult
}

pub fn run() {
    let shader = Arc::new(Shader::from_wgsl_file(&FW, "./examples/add.wgsl").unwrap());

    let size = 32000usize;
    let work_per_thread = size / THREADING;

    let a = (0..size as u32).into_iter().collect::<Vec<u32>>();
    let b = (0..size as u32).into_iter().rev().collect::<Vec<u32>>();

    let mut handles = Vec::with_capacity(THREADING);
    let mut gpu_vec = Vec::new();

    println!("\nCompute using CPU");
    let now = Instant::now();
    let cpu_vec = run_cpu(&a, &b);
    let elapsed = now.elapsed();
    println!("Done in {:.2?}\n", elapsed);

    println!("Compute using CPU (parallel)");
    let now = Instant::now();
    let par_cpu_vec = run_cpu_par(&a, &b);
    let elapsed = now.elapsed();
    println!("Done in {:.2?}\n", elapsed);

    println!("Compute using GPU");
    let now = Instant::now();
    for idx in 0..THREADING {
        let shader = shader.clone();
        let a =
            a.as_slice()[work_per_thread * idx..(work_per_thread * idx) + work_per_thread].to_vec();
        let b =
            b.as_slice()[work_per_thread * idx..(work_per_thread * idx) + work_per_thread].to_vec();

        let handle = std::thread::spawn(move || (run_gpu(a, b, size, shader)));
        handles.push(handle);
    }
    for handle in handles {
        let output = handle.join().unwrap();
        gpu_vec = output;
    }
    let elapsed = now.elapsed();
    println!("Done in {:.2?}\n", elapsed);

    // assert_eq!(&cpu_vec, &gpu_vec);

    println!(
        "cpu-only:     {:?}",
        &cpu_vec[(size - 5) as usize..(size) as usize]
    );
    println!(
        "cpu-parallel: {:?}",
        &par_cpu_vec[(size - 5) as usize..(size) as usize]
    );
    println!(
        "gpu-only:     {:?}",
        &gpu_vec[(work_per_thread - 5)..work_per_thread]
    );
}

fn main() {
    let _ = pretty_env_logger::init();
    run()
}
