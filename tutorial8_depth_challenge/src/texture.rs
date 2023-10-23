use image::GenericImageView;
use anyhow::*;

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, label: &str) -> Self {
        let size = wgpu::Extent3d { // depth texture needs to be the same size as our screen if we want things to render correctly. Can use config to make sure that depth texture is the same size as surface textures
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, // since rendering to this texture, need to add the RENDER_ATTACHMENT flag to it
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(
            &wgpu::SamplerDescriptor { // technically don't need sampler for a depth texture, but Texture struct requires it
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual), // if decide to render depth texture, need to use this. Due to how sampler_comparison and textureSampleCompare() interacts with texture() function in GLSL
                lod_min_clamp: 0.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }
        );

        Self { texture, view, sampler }
    }

    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8], 
        label: &str
    ) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        Self::from_image(device, queue, &img, Some(label))
    }

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>
    ) -> Result<Self> {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            &wgpu::TextureDescriptor {
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Most images are stored using sRGB so we need to reflect that here.
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
                // COPY_DST means that we want to copy data to this texture
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                // This is the same as with the SurfaceConfig. It
                // specifies what texture formats can be used to
                // create TextureViews for this texture. The base
                // texture format (Rgba8UnormSrgb in this case) is
                // always supported. Note that using a different
                // texture format is not supported on the WebGL2
                // backend.
                view_formats: &[],
            }
        );

        queue.write_texture(
            // tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            // the actual pixel data
            &rgba,
            // the layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        // TextureView offers a view into texture
        // Sampler controls how the Texture is sampled
        // sampling works similar to the eyedropper tool in Photoshop
        // program supplies a coordinate on the texture (texture toordinate), and the
        // sampler then returns the corresponding color based on the texture and some internal parameters

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { // address_mode_* determine what to do if the sampler gets a texture coordinate that's outside the texture itself
            address_mode_u: wgpu::AddressMode::ClampToEdge, // ClampToEdge: Any texture coordinates outside the texture will return the color of the nearest pixel on the edges of the texture.
            address_mode_v: wgpu::AddressMode::ClampToEdge, // Repeat: The texture will repeat as texture coordinates exceed the texture's dimensions.
            address_mode_w: wgpu::AddressMode::ClampToEdge, // MirrorRepeat: Similar to Repeat, but the image will flip when going over boundaries.
            mag_filter: wgpu::FilterMode::Linear, // mag and min filter descrube what to do when the sample footprint is smaller or larger than one texel, usually work when mapping in scene is far from or close to camera
            min_filter: wgpu::FilterMode::Nearest, // Linear: Select two texels in each dimension and return a linear interpolation between their values
            mipmap_filter: wgpu::FilterMode::Nearest, // Nearest: return value of texel nearest to the texture coordinates. Creates an image that's crisper from far away but pixelated up close (can be desirable however if textures are designed to be pixelated)
            ..Default::default()
            }
        );
        
        Ok(Self { texture, view, sampler })
    }
}