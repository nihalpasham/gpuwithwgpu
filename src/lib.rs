#![allow(warnings)]

use std::{marker::PhantomData, sync::Arc};

use wgpu::{BindGroupLayoutEntry, BindGroupEntry, BindGroup};

pub mod shader;
pub mod framework;
pub mod kernel;
 
/// Represents a shader.
///
/// It's just a wrapper around [`wgpu::ShaderModule`].
pub struct Shader(wgpu::ShaderModule);

/// Entry point of `gpgpu`. A [`Framework`] must be created
/// first as all GPU primitives needs it to be created.
pub struct Framework {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
}

/// Holds `BindGroupEntries` and their layout. DescriptorSet is basically used to bind resources as entries.  
#[derive(Default, Debug)]
pub struct DescriptorSet<'a> {
    layout: Vec<BindGroupLayoutEntry>,
    set: Vec<BindGroupEntry<'a>>
}

#[derive(PartialEq, Eq)]
pub enum GpuBufferUsage {
    /// Read-only object.
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[group(0), binding(0)]] var<storage, read> input: Vector;
    /// ```
    ReadOnly,
    /// Read-write object.
    /// ### Example WGSL syntax:
    /// ```ignore
    /// [[group(0), binding(0)]] var<storage, read_write> input: Vector;
    /// ```
    ReadWrite,
}

/// Vector of contiguous homogeneous elements on GPU memory.
/// Its elements must implement [`bytemuck::Pod`].
///
/// Equivalent to OpenCL's Buffer objects.
///
/// Basically wraps a [`wgpu::Buffer`] i.e. a gpu accessible buffer.
pub struct GpuBuffer<'fw, T> {
    fw: &'fw Framework,
    buf: wgpu::Buffer,
    size: u64,
    marker: PhantomData<T>,
}

/// Represents a shader with its bindings and entry-point
pub struct Program<'sha, 'res> {
    shader: &'sha Shader,
    entry_point: String,
    descriptors: Vec<DescriptorSet<'res>>,
}

/// dispatches the shader with its bindings
pub struct Kernel<'fw> {
    fw: &'fw Framework,
    pipeline: wgpu::ComputePipeline,
    bindgroups: Vec<BindGroup>,
    entry_point: String
}