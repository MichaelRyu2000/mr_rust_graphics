// https://gpuweb.github.io/gpuweb/#coordinate-systems

// Vertex shader

// @builtin(position) tells WGPU that this is the value we want to use as the vertex's clip coordinates
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
};

@vertex 
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.position = vec2<f32>(x, y);
    return out; 
}

// Fragment shader

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.position, 0.5, 1.0); // sets color of the current fragment to brown
}

