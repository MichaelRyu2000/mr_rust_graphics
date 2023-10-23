use tutorial8_depth::run;

fn main() {
    pollster::block_on(run());
}

/*

Models in the back are being rendered in front of ones in the front.
We can either sort the data from back to front, or use depth buffer.

Sorting is the go-to method for 2d rendering. Can just use the z order.
Sort all objects by their distance to the camera's position. (There are issues with this)

A depth buffer is a black and whtie texture that stores the z-coord of rendered pixels.
Technique is called depth testing.

*/