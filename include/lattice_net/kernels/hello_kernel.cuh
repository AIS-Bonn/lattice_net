// #include "lattice/kernels/dummy_device_func.cuh"
// #include "/media/rosu/Data/phd/c_ws/src/surfel_renderer/include/surfel_renderer/lattice/kernels/dummy_device_func.cuh" //works!
// #include "lattice/kernels/dummy_device_func.cuh"
#include "dummy_device_func.cuh"
__global__ void kernel_hello(){
    // determine where in the thread grid we are
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // unmangle output
    // str[idx] += idx;
    printf("Hello!\n");
    
    float x=3;
    float x_squared=square(x);
    printf("x_squared is %f !\n", x_squared );

}