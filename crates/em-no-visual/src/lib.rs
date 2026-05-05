use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use em_no_train::MaxwellSampleVis;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 2],
    pub color: [f32; 3],
}

struct State {
    _window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vb_e_true: wgpu::Buffer,
    vb_e_pred: wgpu::Buffer,
    vb_eps: wgpu::Buffer,      // eps_r
    len_e_true: u32,
    len_e_pred: u32,
    len_eps: u32,
}

impl State {
    async fn new(window: Arc<Window>, sample: &MaxwellSampleVis) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();

        // 关键：传 Arc<Window>，让 surface 可以是 Surface<'static>
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::default(),
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("line.wgsl").into()),
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ---------- build vertex buffers from sample ----------
        let xs = &sample.xs;
        let e_true = &sample.e_true;
        let e_pred = &sample.e_pred;
        let eps = &sample.eps;

        let x_min = *xs.first().unwrap_or(&0.0);
        let x_max = *xs.last().unwrap_or(&1.0);

        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;

        // 把 eps 也纳入 y 范围，三条线同一坐标系
        for v in e_true.iter().chain(e_pred.iter()).chain(eps.iter()) {
            y_min = y_min.min(*v);
            y_max = y_max.max(*v);
        }

        let y_range = (y_max - y_min).max(1e-6);

        let to_ndc = |x: f32, y: f32| -> [f32; 2] {
            let xn = if x_max > x_min {
                (x - x_min) / (x_max - x_min) * 2.0 - 1.0
            } else {
                0.0
            };

            let yn = (y - y_min) / y_range * 2.0 - 1.0;

            [xn, yn]
        };

        let vertices_e_true: Vec<LineVertex> = xs
            .iter()
            .zip(e_true.iter())
            .map(|(x, y)| LineVertex {
                pos: to_ndc(*x, *y),
                color: [0.0, 0.0, 1.0], // 蓝：真值
            })
            .collect();

        let vertices_e_pred: Vec<LineVertex> = xs
            .iter()
            .zip(e_pred.iter())
            .map(|(x, y)| LineVertex {
                pos: to_ndc(*x, *y),
                color: [1.0, 0.0, 0.0], // 红：预测
            })
            .collect();

        let vertices_eps: Vec<LineVertex> = xs
            .iter()
            .zip(eps.iter())
            .map(|(x, eps_val)| LineVertex {
                pos: to_ndc(*x, *eps_val),
                color: [0.5, 0.5, 0.5], // 灰：eps_r
            })
            .collect();

        let len_e_true = vertices_e_true.len() as u32;
        let len_e_pred = vertices_e_pred.len() as u32;
        let len_eps = vertices_eps.len() as u32;

        let vb_e_true = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vb e_true"),
            contents: bytemuck::cast_slice(&vertices_e_true),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let vb_e_pred = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vb e_pred"),
            contents: bytemuck::cast_slice(&vertices_e_pred),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let vb_eps = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vb eps"),
            contents: bytemuck::cast_slice(&vertices_eps),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            _window: window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vb_e_true,
            vb_e_pred,
            vb_eps,
            len_e_true,
            len_e_pred,
            len_eps,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;

        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) -> Result<()> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,

            // 可以先继续画；但下次最好重新 configure。
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.surface.configure(&self.device, &self.config);
                frame
            }

            // 这两种情况跳过当前 frame。
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }

            // Surface 状态变了，重新 configure 后下一帧再画。
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }

            wgpu::CurrentSurfaceTexture::Validation => {
                anyhow::bail!("surface texture validation error");
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.render_pipeline);

            // 先画 eps_r 灰线
            rpass.set_vertex_buffer(0, self.vb_eps.slice(..));
            rpass.draw(0..self.len_eps, 0..1);

            // 再画真值（蓝）
            rpass.set_vertex_buffer(0, self.vb_e_true.slice(..));
            rpass.draw(0..self.len_e_true, 0..1);

            // 最后画预测（红）
            rpass.set_vertex_buffer(0, self.vb_e_pred.slice(..));
            rpass.draw(0..self.len_e_pred, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }

    fn request_redraw(&self) {
        self._window.request_redraw();
    }
}

struct App {
    sample: MaxwellSampleVis,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        WindowAttributes::default()
                            .with_title("Maxwell1D FNO Visualization")
                            .with_inner_size(PhysicalSize::new(900, 600)),
                    )
                    .expect("Failed to create window"),
            );

            let sample_ref = &self.sample;

            let mut state =
                pollster::block_on(State::new(window.clone(), sample_ref))
                    .expect("init state failed");

            // 先画一帧
            state.render().ok();

            // 关键：保存 state。否则 window/surface/state 会掉出 scope。
            self.state = Some(state);

            // 再请求一次 redraw，避免某些平台第一帧没显示。
            if let Some(state) = &self.state {
                state.request_redraw();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            winit::event::WindowEvent::Resized(new_size) => {
                if let Some(state) = &mut self.state {
                    state.resize(new_size);
                    state.request_redraw();
                }
            }

            winit::event::WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    state.render().ok();
                }
            }

            _ => {}
        }
    }
}

pub async fn run_vis(sample: MaxwellSampleVis) -> Result<()> {
    let event_loop: EventLoop<()> = EventLoop::new().unwrap();

    let mut app = App {
        sample,
        state: None,
    };

    event_loop.run_app(&mut app).unwrap();

    Ok(())
}