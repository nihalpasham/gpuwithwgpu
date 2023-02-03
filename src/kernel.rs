use std::marker::PhantomData;

use thiserror::Error;
use wgpu::{util::DeviceExt, MapMode};

use crate::{DescriptorSet, Framework, GpuBuffer, GpuBufferUsage, Kernel, Program, Shader};

const GPU_BUFFER_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
    wgpu::BufferUsages::STORAGE.bits()
        | wgpu::BufferUsages::COPY_SRC.bits()
        | wgpu::BufferUsages::COPY_DST.bits()
        | wgpu::BufferUsages::MAP_READ.bits()
        | wgpu::BufferUsages::MAP_WRITE.bits(),
);

pub type BufferResult<T> = Result<T, BufferError>;

#[derive(Error, Debug)]
pub enum BufferError {
    #[error(transparent)]
    AsyncMapError(#[from] wgpu::BufferAsyncError),
}

impl<'res> DescriptorSet<'res> {
    pub fn bind_buffer<T>(mut self, storage_buf: &'res GpuBuffer<T>, usage: GpuBufferUsage) -> Self
    where
        T: bytemuck::Pod,
    {
        let bind_id = self.layout.len() as u32;
        // describe a bindgroupentry's layout
        let entry_layout = wgpu::BindGroupLayoutEntry {
            binding: bind_id,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: usage == GpuBufferUsage::ReadOnly,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // construct a bindgroup entry
        let entry = wgpu::BindGroupEntry {
            binding: bind_id,
            resource: storage_buf.as_binding_resource(),
        };

        // push them into a DescriptorSet, which is a collection of bindgroup entries and their layouts.
        self.layout.push(entry_layout);
        self.set.push(entry);

        self
    }
}

impl<'fw, T> GpuBuffer<'fw, T>
where
    T: bytemuck::Pod,
{
    /// Returns a [`wgpu::BindingResource`] of all the elements in the buffer.
    fn as_binding_resource(&self) -> wgpu::BindingResource {
        self.as_gpu_buffer().as_entire_binding()
    }

    fn as_gpu_buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    fn size(&self) -> u64 {
        self.size
    }

    /// Returns the number of elements the buffer can hold.
    fn capacity(&self) -> u64 {
        self.size() / std::mem::size_of::<T>() as u64
    }
    /// get a GPU accessible buffer from a slice
    pub fn from_slice(fw: &'fw Framework, slice: &[T]) -> Self {
        let size = (slice.len() * std::mem::size_of::<T>()) as u64;
        let buf = fw
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(""),
                contents: bytemuck::cast_slice(slice),
                usage: GPU_BUFFER_USAGES,
            });
        Self {
            fw,
            buf,
            size,
            marker: PhantomData,
        }
    }

    pub fn with_capacity(fw: &'fw Framework, capacity: u64) -> Self {
        let size = capacity * std::mem::size_of::<T>() as u64;
        let buf = fw.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer: with capacity"),
            size,
            usage: GPU_BUFFER_USAGES,
            mapped_at_creation: false,
        });
        Self {
            fw,
            buf,
            size,
            marker: PhantomData,
        }
    }

    /// Pulls some elements from the [`GpuBuffer`] into `buf`, returning how many elements were read.
    pub async fn read(&self, buf: &mut [T]) -> BufferResult<u64> {
        let output_size = (buf.len() * std::mem::size_of::<T>()) as u64;
        let download_size = if output_size > self.size {
            self.size
        } else {
            output_size
        };

        let download = self.buf.slice(..download_size as u64);

        let (tx, rx) = futures::channel::oneshot::channel();
        download.map_async(MapMode::Read, |result| {
            tx.send(result).expect("GpuBuffer reading error!");
        });
        rx.await
            .expect("GpuBuffer futures::channel::oneshot error")?;
        buf.copy_from_slice(bytemuck::cast_slice(&download.get_mapped_range()));

        Ok(download_size)
    }

    /// Pulls all the elements from the [`GpuBuffer`] into a [`Vec`].
    pub async fn read_vec(&self) -> BufferResult<Vec<T>> {
        // Safety: Since T is Pod: Zeroed + ... it is safe to use zeroed() to init it.
        let mut buf = vec![unsafe { std::mem::zeroed() }; self.capacity() as usize];
        self.read(&mut buf).await?;

        Ok(buf)
    }
    // /// Blocking version of `GpuBuffer::read_vec()`.
    // pub fn read_vec_blocking(&self) -> BufferResult<Vec<T>> {
    //     futures::executor::block_on(self.read_vec())
    // }
}

impl<'sha, 'res> Program<'sha, 'res> {
    pub fn new(shader: &'sha Shader, entry_point: impl Into<String>) -> Self {
        Self {
            shader,
            entry_point: entry_point.into(),
            descriptors: Vec::new(),
        }
    }

    /// add a descriptor set to the program. another way of saying the descriptor set contains
    /// the resources the shader needs access to
    pub fn add_descriptor(mut self, desc: DescriptorSet<'res>) -> Self {
        self.descriptors.push(desc);
        self
    }
}

impl<'sha, 'res, 'fw> Kernel<'fw> {
    pub fn new(fw: &'fw Framework, program: Program<'sha, 'res>) -> Self {
        let mut bindgroup_layouts = Vec::new();
        let mut bindgroups = Vec::new();

        for (set_id, desc) in program.descriptors.iter().enumerate() {
            let bindgroup_layout =
                fw.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &desc.layout,
                    });
            let bind_group = fw.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bindgroup_layout,
                entries: &desc.set,
            });

            log::debug!("Binding set = {} with {:#?}", set_id, &desc.set);

            bindgroup_layouts.push(bindgroup_layout);
            bindgroups.push(bind_group);
        }

        let bindgroup_layouts = bindgroup_layouts.iter().collect::<Vec<_>>();

        // create the pipeline
        let compute_pipeline_layout =
            fw.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(""),
                    bind_group_layouts: &bindgroup_layouts,
                    push_constant_ranges: &[],
                });
        let compute_pipeline =
            fw.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(""),
                    layout: Some(&compute_pipeline_layout),
                    module: &program.shader.0,
                    entry_point: &program.entry_point,
                });

        Self {
            fw,
            pipeline: compute_pipeline,
            bindgroups,
            entry_point: program.entry_point,
        }
    }

    pub fn enqueue(&self, x: u32, y: u32, z: u32) {
        let mut encoder = self
            .fw
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("begin compute pass encoding"),
            });

            compute_pass.set_pipeline(&self.pipeline);

            for (id, bindgroup) in self.bindgroups.iter().enumerate() {
                compute_pass.set_bind_group(id as u32, bindgroup, &[]);
            }

            compute_pass.insert_debug_marker(&self.entry_point);
            compute_pass.dispatch_workgroups(x, y, z);
        }

        self.fw.queue.submit(Some(encoder.finish()));
    }
}
