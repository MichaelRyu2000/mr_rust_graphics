use tutorial7_instancing::run;

fn main() {
    pollster::block_on(run());
}

/*

Instancing allows us to draw the same object multiple times with different properties (position, orientation, size, color, etc.). 
There are multiple ways of doing instancing. 
One way would be to modify the uniform buffer to include these properties and then update it before we draw each instance of our object.
We don't want to use this method for performance reasons. 
Updating the uniform buffer for each instance would require multiple buffer copies for each frame. 
On top of that, our method to update the uniform buffer currently requires us to create a new buffer to store the updated data. 
That's a lot of time wasted between draw calls.

If we look at the parameters for the draw_indexed function in the wgpu docs (opens new window), we can see a solution to our problem.

pub fn draw_indexed(
    &mut self,
    indices: Range<u32>,
    base_vertex: i32,
    instances: Range<u32> // <-- This right here
)
    
The instances parameter takes a Range<u32>. This parameter tells the GPU how many copies, or instances, of the model we want to draw. 
Currently, we are specifying 0..1, which instructs the GPU to draw our model once, and then stop. 
If we used 0..5, our code would draw 5 instances.

The fact that instances is a Range<u32> may seem weird as using 1..2 for instances would still draw 1 instance of our object. 
Seems like it would be simpler to just use a u32 right? The reason it's a range is that sometimes we don't want to draw all of our objects. 
Sometimes we want to draw a selection of them, because others are not in frame, or we are debugging and want to look at a particular set of instances.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Make sure if you add new instances to the Vec, 
that you recreate the instance_buffer and as well as camera_bind_group, 
otherwise your new instances won't show up correctly.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
*/