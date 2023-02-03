
use crate::{Shader, Framework};

use std::{path::Path, borrow::Cow};


impl Shader {
/// Initialises a [`Shader`] from a `WGSL` file.
pub fn from_wgsl_file(fw: &Framework, path: impl AsRef<Path>) -> std::io::Result<Self> {
    let source_string = std::fs::read_to_string(&path)?;
    let shader_name = path.as_ref().to_str();

    Ok(Self(fw.device.create_shader_module(
        wgpu::ShaderModuleDescriptor {
            label: shader_name,
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source_string)),
        },
    )))
}}