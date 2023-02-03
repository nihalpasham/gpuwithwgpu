use std::{sync::Arc, time::Duration};

// use std::sync::Arc;
use crate::Framework;

impl Framework {
    pub async fn default() -> Self {
        // specify the backend, we'll just pick one from the environment i.e. whatever's available.
        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        // specify a power profile for the gpu, again pick one from the environment.
        let power_preference = wgpu::util::power_preference_from_env()
            .unwrap_or(wgpu::PowerPreference::HighPerformance);
        // initialize a wgpu instance to create the adapter type
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backend,
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        });

        // create the adapter. `request_adapter` returns a Future, so must be awaited
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                ..Default::default()
            })
            .await
            .expect("Failed to find an appropriate adapter");
        // create the device and queue. device is reponsible for the creation of most
        // rendering and compute resources. These are then used in commands, which are submitted to a [`Queue`].
        // `request_device` returns a Future, so must be awaited
        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("M1 Max 24 core"),
                    features: adapter.features(),
                    limits: adapter.limits(),
                },
                None,
            )
            .await
        {
            Ok((d, v)) => (d, v),
            Err(e) => panic!("Failed at device creation: {}", e),
        };

        let device = Arc::new(device);
        let polling_device = Arc::clone(&device);

        std::thread::spawn(move || loop {
            polling_device.poll(wgpu::Maintain::Poll);
            std::thread::sleep(Duration::from_millis(10));
        });

        // constuct and return a framework
        Framework {
            device,
            queue,
            adapter,
        }
    }
}
