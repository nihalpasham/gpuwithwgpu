use std::time::Instant;

use gpgpu_test::{DescriptorSet, Framework, GpuBuffer, GpuBufferUsage, Kernel, Program, Shader};
use rayon::prelude::*;


pub fn run_cpu(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let cpu_vec: Vec<u32> = a.iter().zip(b).map(|(a, b)| a * b).collect();
    cpu_vec
}

pub async fn run_cpu_par(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let par_cpu_vec: Vec<u32> = a.par_iter().zip(b).map(|(a, b)| a * b).collect();
    par_cpu_vec
}

pub async fn run_gpu(host_vec_a: &Vec<u32>, host_vec_b: &Vec<u32>, size: u32) -> Vec<u32> {
    let fw = Framework::default().await;
    let shader = Shader::from_wgsl_file(&fw, "./examples/vecmult.wgsl").unwrap();

    let gpu_vec_a = GpuBuffer::from_slice(&fw, host_vec_a.as_slice());
    let gpu_vec_b = GpuBuffer::from_slice(&fw, host_vec_b.as_slice());
    let gpu_vec_mult: GpuBuffer<u32> = GpuBuffer::with_capacity(&fw, size as u64);  

    let bindings = DescriptorSet::default() // bindgroup 0
        .bind_buffer(&gpu_vec_a, GpuBufferUsage::ReadOnly) // bindentry 0
        .bind_buffer(&gpu_vec_b, GpuBufferUsage::ReadOnly) // bindentry 1
        .bind_buffer(&gpu_vec_mult, GpuBufferUsage::ReadWrite); // bindentry 2

    let program = Program::new(&shader, "main").add_descriptor(bindings);

    let kernel = Kernel::new(&fw, program);

    kernel.enqueue(size, 10, 1);

    let result_vec_mult = gpu_vec_mult.read_vec().await.unwrap();

    result_vec_mult
}

async fn run() {
    let size = 50000;

    let a = (1..size).into_iter().map(|x| x as u32).collect::<Vec<u32>>();
    let b = (1..size).into_iter().map(|x| x as u32).rev().collect::<Vec<u32>>();

    println!("\nCompute using CPU");
    let now = Instant::now();
    let cpu_vec = run_cpu(&a, &b);
    let elapsed = now.elapsed();
    println!("Done in {:.2?}\n", elapsed);

    println!("Compute using CPU (parallel)");
    let now = Instant::now();
    let par_cpu_vec = run_cpu_par(&a, &b).await;
    let elapsed = now.elapsed();
    println!("Done in {:.2?}\n", elapsed);

    println!("Compute using GPU");
    let now = Instant::now();
    let gpu_vec = run_gpu(&a, &b, size).await;
    let elapsed = now.elapsed();
    println!("Done in {:.2?}\n", elapsed);

    // assert_eq!(&cpu_vec, &gpu_vec);
    println!("cpu-only:     {:?}", &cpu_vec[(size - 5) as usize..(size - 1) as usize]);
    println!("cpu-parallel: {:?}", &par_cpu_vec[(size - 5) as usize..(size -1) as usize]);
    println!("gpu-only:     {:?}", &gpu_vec[(size - 5) as usize..(size -1) as usize]);
}

fn main() {
    let _ = pretty_env_logger::init();
    futures::executor::block_on(run())
}
