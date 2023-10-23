mod texture;
use winit::window::Window;
use wgpu::util::DeviceExt;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)] // vertex needs to be Copy so we can create a buffer with it
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

// Upside Down Texture
// const VERTICES: &[Vertex] = &[
//     Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.99240386], }, // A
//     Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.56958647], }, // B
//     Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.05060294], }, // C
//     Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.1526709], }, // D
//     Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.7347359], }, // E
// ];

// Right Side Up Texture
const VERTICES: &[Vertex] = &[
    // Changed
    Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.00759614], }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.43041354], }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.949397], }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.84732914], }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.2652641], }, // E
];

const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];

impl Vertex {
    // consider using wpgu vertex_attr_array macro to clean this up
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                }
            ]
        }
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    window: Window,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    diffuse_bind_group: wgpu::BindGroup,
    challenge_diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    toggle_texture: bool,
}

impl State {
    // Creating some of the wgpu types requires async 
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // instance main purpose is to create Adapters and Surfaces
        // Backends::all => Vulkan + Metal + Dx12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety

        // surface needs to live as long as the window that creates it
        // State owns the window so it should be safe

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        // adapter is handle to actual graphics card
        // will use to create our Device and Queue
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(), // has two variants: LowPower and HighPerformance
                compatible_surface: Some(&surface),
                force_fallback_adapter: false, // forces wgpu to pick an adapter that will work on all hardware, usually means that rendering backend will be "software" rather than hardware such as GPU
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None, // Trace path
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different one will result all color coming out darker.
        // If want to support non sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // RENDER_ATTACHMENT specifies that tectures will be used to write to the screen
            format: surface_format, //  how SurfaceTextures will be stored on the gpu
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0], // determines how to sync the surface with the display. Fifo will cap display rate to display framerate (VSync)
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![], // a list of TextureFormats that can use when creating TextureViews 
        };

        // if want to let users pick what PresentMode, can use SurfaceCapabilities::present_modes to get list 
        surface.configure(&device, &config);

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy_tree.png").unwrap();

        let challenge_diffuse_bytes = include_bytes!("challenge.png");
        let challenge_diffuse_texture = texture::Texture::from_bytes(&device, &queue, challenge_diffuse_bytes, "challenge.png").unwrap();

        // a BindGroup describes a set of resources and how they can be accessed by a shader
        // create a BindGroup using a BindGroupLayout
        // two entries: one sampled texture at binding 0, and another at binding 1
        // both are visible only to the fragment shader as specified by FRAGMENT (other choices are: NONE, VERTEX, FRAGMENT, COMPUTE)
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        // BindGroup is more specific declaration of BindGroupLayout
        // separate since it allows us to swap out BindGroups on the fly, so long as they share the same Layout
        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );

        let challenge_diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&challenge_diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&challenge_diffuse_texture.sampler),
                    }
                ],
                label: Some("challenge_diffuse_bind_group")
            }
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // can specify which function inside the shader should be the entry point
                buffers: &[Vertex::desc()], // tells wgpu what type of vertices we want to pass to the vertex shader
            },
            fragment: Some(wgpu::FragmentState { // technically optional, will need if we want to store color data to the surface
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { // tells wgpu what color outputs it should set up
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // means that every three vertices will correspond to one triangle
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // tells wpgu how to determine whether a given triangle is facing forward or not, ccw means facing forward if vertices in counter-clockwise direction
                // triangles NOT considered facing forward (clockwise) are culled as specified below
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
                depth_stencil: None, 
                multisample: wgpu::MultisampleState {
                    count: 1, // determines how many samples the pipeline will use
                    mask: !0, // which samples should be active (in this case, all of them)
                    alpha_to_coverage_enabled: false, // has to do with anti-aliasing
                },
                multiview: None, // how many array layers the render attachments can have. Currently not rendering to array textures so None
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let num_indices = INDICES.len() as u32;

        let toggle_texture = true;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            window,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            challenge_diffuse_bind_group,
            diffuse_texture,
            toggle_texture
        }  
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    // returns a bool to indicate whether an event has been fully processed
    // if true, main loop won't process the event any further
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { input: KeyboardInput {
                state,
                virtual_keycode: Some(VirtualKeyCode::Space),
                ..
            }, ..
            } => {
                self.toggle_texture = *state == ElementState::Released;
                true
            }
        _ => false,
        }
    }

    fn update(&mut self) {
        // todo!()
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // get_current_texture will wait for the surface to provide a new SurfaceTexture that we will render to
        let output = self.surface.get_current_texture()?;
        // creates a TextureView with default settings. Need to do this because want to control how the render code interacts with the texture
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        // will need to create a CommandEncoder to create actual commands to send to gpu
        // most modern graphics frameworks expect commands to be stored in a command buffer before being sent to the gpu
        // encoder builds a command buffer that we can then send to gpu
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // use encoder to create a RenderPass. RenderPass has all methods for the actual drawing
        {
            // begin render pass borrows encoder mutable (aka &mut self)
            // can't call encoder.finish() until we release that mutable borrow
            // the block around _render_pass tells rust to drop any variables within it when the code
            // leaves that scope thus releasing the mutabel borrow on encoder and allowing us to finish() it
            // can also call drop(render_pass) to achieve the same effect
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true, // tells wgpu whether we want to store the rendered result to Texture behind our TextureView (in this case, SurfaceTexture)
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline); // set pipeline on the render pass using the one we just created
            if self.toggle_texture {
                render_pass.set_bind_group(0, &self.challenge_diffuse_bind_group, &[]);
            } else {
                render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            }
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..)); // .. specifies entire buffer; can store as many objects in a buffer as hardware allows
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1); // draw something with 3 vertices and 1 instance, this is where @builtin(vertex_index) comes from
        }
        // tell wgpu to finish command buffer and to submit it to the gpu's render queue
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    
        Ok(())
    }
}
// note: enumerate_adapters isn't available on WASM, so have to use request_adapter

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));
        
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut state = State::new(window).await;

    event_loop.run(move |event, _, control_flow| 
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size is &&mut so we have to dereference it twice
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window().request_redraw();
            }
            _ => {}
        }
    );
}

 