use winit::window::Window;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    clear_color: wgpu::Color,
    // window must be declared after the surface so it gets
    // dropped after it as the surface contains unsafe
    // references to the window's resources
    window: Window,
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

        let clear_color = wgpu::Color::BLACK;

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
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
            WindowEvent::CursorMoved { position, .. } => {
                self.clear_color = wgpu::Color {
                    r: position.x as f64 / self.size.width as f64,
                    g: position.y as f64 / self.size.height as f64,
                    b: 1.0,
                    a: 1.0,
                };
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
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true, // tells wgpu whether we want to store the rendered result to Texture behind our TextureView (in this case, SurfaceTexture)
                    },
                })],
                depth_stencil_attachment: None,
            });
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

 