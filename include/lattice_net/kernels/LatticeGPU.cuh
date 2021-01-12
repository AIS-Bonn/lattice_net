#pragma once

#include "lattice_net/kernels/HashTableGPU.cuh"

#ifndef __CUDACC_RTC__ 
    #include "lattice_net/jitify_helper/jitify_helper.cuh"
#endif

#if !defined(__CUDACC_RTC__)
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
    #include "device_launch_parameters.h" //needed for threadIdx and blockDim 
#endif

#ifndef __CUDACC_RTC__ 
//Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
#define ENABLE_CUDA_PROFILING 1
#include "Profiler.h" 
#endif


#define BLOCK_SIZE 256

class LatticeGPU { 
public:

    //during nvrtc compilation we do not want to compile this code as it does not work due to not including vector definition
    #ifndef __CUDACC_RTC__ 
        LatticeGPU(){
            create_program_handles();
            cudaEventCreate (&m_event_nr_vertices_lattice_changed);
        }

        //it uses Jittify to get a handle for the programs. The programs can contain more than one kernel.. It doesnt actually compile them, they will get jit compiled the first time you run them
        void create_program_handles(){
            m_lattice_program=create_jitify_program( std::string(CMAKE_SOURCE_DIR)+"/include/lattice_net/kernels/LatticeGPU.cuh" );
        }


        // void splat_standalone(const float* positions, const float* values, const int nr_positions, const int pos_dim, const int val_dim, const float* splatting_indices_and_weights, const HashTableGPU& hash_table_gpu){
        void splat_standalone(const float* positions, const float* values, const int nr_positions, const int pos_dim, const int val_dim, const int* splatting_indices, const float* splatting_weights, const HashTableGPU& hash_table_gpu){
   
            TIME_START("kernel_splat");
            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res_1= m_lattice_program.kernel("kernel_splat")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        // .launch( positions, values, nr_positions, splatting_indices_and_weights, hash_table_gpu );
                        .launch( positions, nr_positions, splatting_indices, splatting_weights, hash_table_gpu, true );
            cudaEventRecord (m_event_nr_vertices_lattice_changed);
            TIME_END("kernel_splat");
            CUDA_CHECK_CURESULT(res_1);
            CUDA_CHECK_ERROR();



            TIME_START("splatCacheNaive");
            blocks=dim3((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            blockSize=dim3(BLOCK_SIZE, 1, 1);
            CUresult res_2= m_lattice_program.kernel("splatCacheNaive")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        // .launch( nr_positions, values, splatting_indices_and_weights, hash_table_gpu );
                        .launch( nr_positions, values, splatting_indices, splatting_weights, hash_table_gpu );
            TIME_END("splatCacheNaive");
            CUDA_CHECK_CURESULT(res_2);
            CUDA_CHECK_ERROR()


        }


        void just_create_verts(const float* positions, const int nr_positions, const int pos_dim, const int val_dim,  const int* splatting_indices, const float* splatting_weights, const HashTableGPU& hash_table_gpu){
   
            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_lattice_program.kernel("kernel_splat")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( positions, nr_positions, splatting_indices, splatting_weights, hash_table_gpu, false );
            cudaEventRecord (m_event_nr_vertices_lattice_changed);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }



        void distribute(const float* positions, const float* values, const float* distributed, const int nr_positions, const int pos_dim, const int val_dim,  const int* splatting_indices, const float* splatting_weights, const HashTableGPU& hash_table_gpu){
   
            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_lattice_program.kernel("distribute")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( positions, values, nr_positions, splatting_indices, splatting_weights, distributed, hash_table_gpu );
            cudaEventRecord (m_event_nr_vertices_lattice_changed);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }

        void create_splatting_mask(bool* mask,  const int* splatting_indices, const int* nr_points_per_simplex, const int max_nr_points, const int nr_positions, const int pos_dim){
  
            int size_of_indices_vector=nr_positions*(pos_dim+1);
            dim3 blocks(( nr_positions*(pos_dim+1) - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_lattice_program.kernel("create_splatting_mask")
                        .instantiate(pos_dim)
                        .configure(blocks, blockSize)
                        .launch( mask, splatting_indices, nr_points_per_simplex, max_nr_points, size_of_indices_vector);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }


        void blur_standalone(const int hash_table_capacity, const int pos_dim, const int val_full_dim, const float* new_values, const int remainder, const HashTableGPU& hash_table_gpu){

            dim3 blocks((hash_table_capacity - 1) / BLOCK_SIZE + 1, 1, 1);

            CUresult res= m_lattice_program.kernel("blur")
                    .instantiate(pos_dim, val_full_dim)
                    .configure(blocks, BLOCK_SIZE)
                    .launch( hash_table_capacity, new_values, remainder, hash_table_gpu );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }

        void convolve_standalone(const int hash_table_capacity, const int pos_dim, const int val_dim, float* new_values, const float* filter_bank, const int nr_filters, const int filter_extent, const HashTableGPU& hash_table_gpu){

            dim3 blocks((hash_table_capacity - 1) / BLOCK_SIZE + 1, 1, 1);

            CUresult res= m_lattice_program.kernel("convolve")
                    .instantiate(pos_dim, val_dim)
                    .configure(blocks, BLOCK_SIZE)
                    .launch( hash_table_capacity, new_values, filter_bank, nr_filters, filter_extent, hash_table_gpu );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }


        void depthwise_convolve(const int nr_vertices, const int pos_dim, const int val_dim, float*filter_bank, const int dilation, float* values_out, const int filter_extent, const HashTableGPU& hash_table_query, const HashTableGPU& hash_table_neighbours, const int query_lvl, const int neighbours_lvl, const bool use_center_vertex_from_lattice_neighbours, const bool flip_neighbours, const bool debug_kernel){

            int nr_blocks=nr_vertices/BLOCK_SIZE;
            // check for partial block at the end
            if(nr_vertices % BLOCK_SIZE) ++nr_blocks; 

            CUresult res= m_lattice_program.kernel("depthwise_convolve")
                    .instantiate(pos_dim, val_dim, filter_extent)
                    .configure(nr_blocks, BLOCK_SIZE)
                    .launch( nr_vertices, values_out, filter_bank, dilation, hash_table_query, hash_table_neighbours, query_lvl, neighbours_lvl, use_center_vertex_from_lattice_neighbours, flip_neighbours, debug_kernel);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }

        //creates a lattice rowified by grabbing the values of the neighbours from the hash_table_neighbours. The neigbhours are the neighbours of the keys in hash_table_query. Useful for lattices which are at different coarsenes levels
        void im2row(const int nr_vertices, const int pos_dim, const int val_dim, const int dilation, float* im2row_out, const int filter_extent, const HashTableGPU& hash_table_query, const HashTableGPU& hash_table_neighbours, const int query_lvl, const int neighbours_lvl, const bool flip_neighbours, const bool debug_kernel){

            int nr_blocks=nr_vertices/BLOCK_SIZE;
            // check for partial block at the end
            if(nr_vertices % BLOCK_SIZE) ++nr_blocks; 

            CUresult res= m_lattice_program.kernel("im2row")
                    .instantiate(pos_dim, val_dim)
                    .configure(nr_blocks, BLOCK_SIZE)
                    .launch( nr_vertices, im2row_out, filter_extent, dilation, hash_table_query, hash_table_neighbours, query_lvl, neighbours_lvl, flip_neighbours, debug_kernel);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }


        void row2im(const int hash_table_capacity, const int pos_dim, const int val_full_dim, const int dilation, float* im2row_in, const int filter_extent, const HashTableGPU& hash_table_query, const HashTableGPU& hash_table_neighbours, const int query_lvl, const int neighbours_lvl, const bool do_test){


            dim3 blocks((hash_table_capacity - 1) / BLOCK_SIZE + 1, 1, 1);

            CUresult res= m_lattice_program.kernel("row2im")
                    .instantiate(pos_dim, val_full_dim)
                    .configure(blocks, BLOCK_SIZE)
                    .launch( hash_table_capacity, im2row_in, filter_extent, dilation, hash_table_query, hash_table_neighbours, query_lvl, neighbours_lvl, do_test);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }

        void coarsen(const int hash_table_capacity, const int pos_dim, const HashTableGPU& fine_hash_table_gpu, const HashTableGPU& coarse_hash_table_gpu){
            dim3 blocks((hash_table_capacity - 1) / BLOCK_SIZE + 1, 1, 1);

            CUresult res= m_lattice_program.kernel("coarsen")
                    .instantiate(pos_dim)
                    .configure(blocks, BLOCK_SIZE)
                    .launch(hash_table_capacity, fine_hash_table_gpu, coarse_hash_table_gpu );
            cudaEventRecord (m_event_nr_vertices_lattice_changed);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }

        void slice_standalone_with_precomputation(const float* positions, float* sliced_values, const int pos_dim, const int val_dim, const int nr_positions, const int* splatting_indices, const float* splatting_weights,  const HashTableGPU& hash_table_gpu){

            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            int cleanBlockSize = 128;
            dim3 cleanBlocks((nr_positions - 1) / cleanBlockSize + 1, 2 * (pos_dim + 1), 1);

            blockSize.y = 1;
            CUresult res= m_lattice_program.kernel("slice_with_precomputation")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( positions, sliced_values, nr_positions, splatting_indices, splatting_weights, hash_table_gpu);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }


        void slice_standalone_no_precomputation(const float* positions, float* sliced_values, const int pos_dim, const int val_dim, const int nr_positions, int* splatting_indices, float* splatting_weights,  const HashTableGPU& hash_table_gpu){

            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            int cleanBlockSize = 128;
            dim3 cleanBlocks((nr_positions - 1) / cleanBlockSize + 1, 2 * (pos_dim + 1), 1);

            blockSize.y = 1;
            CUresult res= m_lattice_program.kernel("slice_no_precomputation")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( positions, sliced_values, nr_positions, splatting_indices, splatting_weights, hash_table_gpu);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }

        void gather_standalone_no_precomputation(const float* positions, float* gathered_values, const int pos_dim, const int val_dim, const int nr_positions, const int* splatting_indices, const float* splatting_weights,  const HashTableGPU& hash_table_gpu){

            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);

            CUresult res= m_lattice_program.kernel("gather_no_precomputation")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( positions, gathered_values, nr_positions, splatting_indices, splatting_weights, hash_table_gpu);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }

        void gather_standalone_with_precomputation(const float* positions, float* gathered_values, const int pos_dim, const int val_dim, const int nr_positions, const int* splatting_indices, const float* splatting_weights,  const HashTableGPU& hash_table_gpu){

            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);

            CUresult res= m_lattice_program.kernel("gather_with_precomputation")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( positions, gathered_values, nr_positions, splatting_indices, splatting_weights, hash_table_gpu);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }

        

        void gather_elevated_standalone_no_precomputation(const int* keys, float* gathered_values, const int pos_dim, const int val_full_dim, const int nr_vertices, const int* splatting_indices, const float* splatting_weights,  const HashTableGPU& hash_table_gpu_to_gather_from, const int lattice_to_gather_from_lvl, const int elevated_verts_lvl){

            dim3 blocks((nr_vertices - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);

            CUresult res= m_lattice_program.kernel("gather_elevated_no_precomputation")
                        .instantiate(pos_dim, val_full_dim)
                        .configure(blocks, blockSize)
                        .launch( keys, gathered_values, nr_vertices, splatting_indices, splatting_weights, hash_table_gpu_to_gather_from, lattice_to_gather_from_lvl,  elevated_verts_lvl);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }



        void slice_elevated_verts(const int hash_table_capacity, float* sliced_values, const int pos_dim, const int val_full_dim, int* splatting_indices, float* splatting_weights, const HashTableGPU& hash_table_to_slice_from, const HashTableGPU& hash_table_elevated_verts, const int lattice_to_slice_from_lvl, const int elevated_vert_lvl){

            dim3 blocks((hash_table_capacity - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);

            blockSize.y = 1;
            CUresult res= m_lattice_program.kernel("slice_elevated_verts")
                        .instantiate(pos_dim, val_full_dim)
                        .configure(blocks, blockSize)
                        .launch( sliced_values, splatting_indices, splatting_weights, hash_table_to_slice_from, hash_table_elevated_verts, lattice_to_slice_from_lvl, elevated_vert_lvl);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }

        void slice_classify_no_precomputation(const float* positions, float* class_logits, const float* delta_weights, const float* linear_clasify_weight, const float* linear_clasify_bias, const int nr_classes, const int pos_dim, const int val_dim, const int nr_positions, const int* splatting_indices, const float* splatting_weights,  const HashTableGPU& hash_table_gpu){

            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);


            CUresult res= m_lattice_program.kernel("slice_classify_no_precomputation")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( positions, class_logits, delta_weights, linear_clasify_weight, linear_clasify_bias, nr_classes, nr_positions, splatting_indices, splatting_weights, hash_table_gpu);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }


        void slice_classify_with_precomputation(const float* positions, float* class_logits, const float* delta_weights, const float* linear_clasify_weight, const float* linear_clasify_bias, const int nr_classes, const int pos_dim, const int val_dim, const int nr_positions, const int* splatting_indices, const float* splatting_weights,  const HashTableGPU& hash_table_gpu){

            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);


            CUresult res= m_lattice_program.kernel("slice_classify_with_precomputation")
                        .instantiate(pos_dim, val_dim, nr_classes)
                        .configure(blocks, blockSize)
                        .launch( positions, class_logits, delta_weights, linear_clasify_weight, linear_clasify_bias, nr_positions, splatting_indices, splatting_weights, hash_table_gpu);
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();
        }

        

        void slice_backwards_standalone_with_precomputation(float* sliced_values_hom, float* grad_sliced_values, int* splatting_indices, float* splatting_weights,  const int pos_dim, const int val_full_dim, const int nr_positions, const HashTableGPU& hash_table_gpu){


            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_lattice_program.kernel("slice_backwards_with_precomputation")
                        .instantiate(pos_dim, val_full_dim)
                        .configure(blocks, blockSize)
                        .launch( nr_positions, sliced_values_hom, grad_sliced_values, splatting_indices, splatting_weights, hash_table_gpu );
            CUDA_CHECK_ERROR()

        }

        void slice_backwards_standalone_with_precomputation_no_homogeneous(float* grad_sliced_values, int* splatting_indices, float* splatting_weights,  const int pos_dim, const int val_dim, const int nr_positions, const HashTableGPU& hash_table_gpu){


            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_lattice_program.kernel("slice_backwards_with_precomputation_no_homogeneous")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( nr_positions, grad_sliced_values, splatting_indices, splatting_weights, hash_table_gpu );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR()

        }

 
        void slice_classify_backwards_with_precomputation(float* grad_class_logits, float* initial_values, int* splatting_indices, float* splatting_weights,  const int pos_dim, const int val_dim, const int nr_positions, 
        float* delta_weights, float* linear_clasify_weight, float* linear_clasify_bias, const int nr_classes, float* grad_lattice_values, float* grad_delta_weights, float* grad_linear_clasify_weight, float* grad_linear_clasify_bias,
         const HashTableGPU& hash_table_gpu){

            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_lattice_program.kernel("slice_classify_backwards_with_precomputation")
                        .instantiate(pos_dim, val_dim, nr_classes)
                        .configure(blocks, blockSize)
                        .launch( nr_positions, grad_class_logits, initial_values, splatting_indices, splatting_weights, delta_weights, linear_clasify_weight, linear_clasify_bias, grad_lattice_values, grad_delta_weights,
                         grad_linear_clasify_weight, grad_linear_clasify_bias, hash_table_gpu );

            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR()

        }

        void gather_backwards_standalone_with_precomputation(float* grad_sliced_values, int* splatting_indices, float* splatting_weights,  const int pos_dim, const int val_dim, const int nr_positions, const HashTableGPU& hash_table_gpu){


            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res = m_lattice_program.kernel("gather_backwards_with_precomputation")
                        .instantiate(pos_dim, val_dim)
                        .configure(blocks, blockSize)
                        .launch( nr_positions, grad_sliced_values, splatting_indices, splatting_weights, hash_table_gpu );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR()

        }

        void elevate(float* positions, const int pos_dim, const int nr_positions, float* elevated){

            dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_lattice_program.kernel("elevate_points")
                        .instantiate(pos_dim)
                        .configure(blocks, blockSize)
                        .launch( nr_positions, positions, elevated );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR()

        }

        void wait_to_create_vertices(){
            cudaEventSynchronize(m_event_nr_vertices_lattice_changed);
        }
        

        jitify::Program m_lattice_program;

        //for syncronization
        cudaEvent_t m_event_nr_vertices_lattice_changed; //when doing splatting, distribute, or a create_coarse_verts, we must record this even after the kernel. Afterwards when we call nr_lattice_vertices we wait for this event to have finished indicating that the kernel has finished
    #endif



   

};


#if defined(__CUDACC_RTC__)

//elevated a vector from m_pos_dim to a m_pos_dim+1 space
template<int pos_dim>
__device__ void elevate(float* elevated, const float* position){
    //TODO the scale factor can be precomputed
    float scaleFactor[pos_dim];
    float invStdDev = (pos_dim + 1) * sqrt(2.0f / 3);
    // float invStdDev = 1.0;
    for (int i = 0; i < pos_dim; i++) {
        scaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
    }

    // embed position vector into the hyperplane
    // first rotate position into the (pd+1)-dimensional hyperplane
    // sm contains the sum of 1..n of our feature vector
    float sm = 0;
    for (int i = pos_dim; i > 0; i--) {
        float cf = position[i - 1] * scaleFactor[i - 1];
        // float cf = position[i - 1] ;
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;

}

//check if a vector has all coordinates integers like 1.0 or do they have a fractional part. Useful when having lattice keys from different scales
__device__ bool are_all_coords_integer(const float* vec, const int vec_size){
    for (int i = 0; i < vec_size; i++) {
        float val=vec[i];
        float integer_part=0.0;
        float decimal_part=0.0;
        decimal_part=modff(val, &integer_part);
        decimal_part=fabs(decimal_part);
        if( decimal_part>0.0001 ){ 
            return false;
        }
    }
    return true;

}


//I believe that when we embedd a fine lattice in a coarse one we can end up with keys of type 0.5, 0.5, -1.0 so making movements of 0.5 with them will end up in non integer keys. This helps me debug this 
__device__ bool is_only_one_coord_integer(const float* vec, const int vec_size){
    int nr_integer_coords=0;
    for (int i = 0; i < vec_size; i++) {
        float val=vec[i];
        float integer_part=0.0;
        float decimal_part=0.0;
        decimal_part=modff(val, &integer_part);
        decimal_part=fabs(decimal_part);
        if( decimal_part<0.0001 ){  //if the decimal part is very small we say it's integer
            return nr_integer_coords++;
        }
    }

    if (nr_integer_coords==1){
        return true;
    }else{
        return false;
    }

}
__device__ int nr_coords_integer(const float* vec, const int vec_size){
    int nr_integer_coords=0;
    for (int i = 0; i < vec_size; i++) {
        float val=vec[i];
        float integer_part=0.0;
        float decimal_part=0.0;
        decimal_part=modff(val, &integer_part);
        decimal_part=fabs(decimal_part);
        if( decimal_part<0.0001 ){  //if the decimal part is very small we say it's integer
            return nr_integer_coords++;
        }
    }

    return nr_integer_coords;

}


template<int pos_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
elevate_points(const int nr_positions,  const float* positions, float* elevated){

    // determine where in the thread grid we are
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }
    
    float* elevated_point = elevated + idx * (pos_dim + 1);
    const float *position = positions + idx * pos_dim;
    elevate<pos_dim>(elevated_point, position);

}


template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
distribute(float* positions, float* values, const int nr_positions, int* splatting_indices, float* splatting_weights, float* distributed, HashTableGPU hash_table){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new position from the input point cloud

    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }
    
    float elevated[pos_dim + 1];
    const float *position = positions + idx * pos_dim;
    elevate<pos_dim>(elevated, position);
    int rem0[pos_dim + 1];
    int rank[pos_dim + 1];

    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pos_dim; i++)
        rank[i] = 0;
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];


    int key[pos_dim+1];
    float* distributed_out_for_cur_position=distributed + idx*(pos_dim+1)*( pos_dim + val_dim +1 ); //each row of the distributed will have  ( pos_dim + val_dim +1 ) elements for each of the (pos_dim+1) vertices of a simplex. therefore the whole row of the distributed matrix is of size (pos_dim+1)*( pos_dim + val_dim +1 )
    float* value_cur_position= values + idx * val_dim;
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly. Including the last coordinate, even though it is redundant, as we know it sums up to 0. 
        for (int i = 0; i < pos_dim+1; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

      

        //using two matrices for indices and weights
        int index_in_m_entries=hash_table.insert(key); //the slot in which it will be inserted is linearly increasing with an atomicAdd
        if(index_in_m_entries>=0){ //if it got inserted correctly in the hashmap
            splatting_indices[idx * (pos_dim + 1) + remainder]=hash_table.m_entries[index_in_m_entries]; //it indexes in m_keys
            splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
        }


        //store the distributed values at this row of the distributed matrix, which for positions will be just the position in xyz and for values will be value. We also store a barycentric weight
        float* distributed_out_lattice_vertex = distributed_out_for_cur_position + remainder*( pos_dim + val_dim+1  );
        // distribute positions
        for(int i=0; i<pos_dim; i++){
           distributed_out_lattice_vertex[i] = position[i];  
        }
        //distribute values
        for(int i=0; i<val_dim; i++){
           distributed_out_lattice_vertex[ pos_dim + i] = value_cur_position[i]; 
        }
        //distribute barycentric
        distributed_out_lattice_vertex[ pos_dim+val_dim] = barycentric[remainder]; 
        
        
    

        

    }





}


template<int pos_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
create_splatting_mask(bool* mask, const int* splatting_indices,  const int* nr_points_per_simplex, const int max_nr_points, const int size_of_indices_vector){

    // determine where in the thread grid we are
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with an edge going form a point towards a lattice vertex
    if(idx>=size_of_indices_vector){ //don't go out of bounds
        return;
    } 

    int lattice_vertex_idx=splatting_indices[idx];
    if(lattice_vertex_idx<0){ //if the point didnt splat, then we have a -1 and we don't care about that
        return;
    }
    int nr_points=nr_points_per_simplex[lattice_vertex_idx];
    if(nr_points>max_nr_points){
        //roll a dice and write a True if we keep the edge(the contributuon from the point to the vertex) and a False if we dont
        // curandState localState = globalState[idx];
        // float random = curand_uniform( &localState ); //uniform between 0.0 and 1.0
        // unsigned int r_int=RNG(idx);

        //https://stackoverflow.com/a/12230158
        unsigned int m_w = idx;
        unsigned int m_z = size_of_indices_vector;
        m_z = 36969 * (m_z & 65535) + (m_z >> 16);
        m_w = 18000 * (m_w & 65535) + (m_w >> 16);
        unsigned int r_int=(m_z << 16) + m_w;



        // r_int=r_int%100000000;
        // float r_float=r_int/100000000.0;
        r_int=r_int%429496729;
        float r_float=r_int/429496729.0;
        // printf("%f\n", r_float);


        float overfill=nr_points/max_nr_points; //by how much we verfilled out cap
        //imagine we overifll by a factor of 100
        if (r_float < 1.0/overfill){
            mask[idx]=true; //keep 
        }else{
            mask[idx]=false; //kill
        }

        // globalState[idx] = localState;
    }else{
        mask[idx]=true; //we keep this edge(this contribution)
    }

}


template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
kernel_splat(const float* positions, const int nr_positions, int* splatting_indices, float* splatting_weights, HashTableGPU hash_table, bool write_new_indices_and_weights){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new position

    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }
    
    float elevated[pos_dim + 1];
    const float *position = positions + idx * pos_dim;
    int rem0[pos_dim + 1];
    int rank[pos_dim + 1];


    //TODO the scale factor can be precomputed
    float scaleFactor[pos_dim];
    float invStdDev = (pos_dim + 1) * sqrt(2.0f / 3);
    for (int i = 0; i < pos_dim; i++) {
        scaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
    }

    // embed position vector into the hyperplane
    // first rotate position into the (pd+1)-dimensional hyperplane
    // sm contains the sum of 1..n of our feature vector
    float sm = 0;
    for (int i = pos_dim; i > 0; i--) {
        float cf = position[i - 1] * scaleFactor[i - 1];
        // float cf = position[i - 1] ;
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pos_dim; i++)
        rank[i] = 0;
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];


    int key[pos_dim];
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // MatrixEntry r;
        // // unsigned int slot = static_cast<unsigned int>(idx * (pos_dim + 1) + remainder);
        // // r.index = hash_table.insert(key, slot);
        // r.index = hash_table.insert(key); //the slot in which it will be inserted is linearly increasing with an atomicAdd
        // r.weight = barycentric[remainder];
        // // if( threadIdx.x==31 && blockIdx.x==353){
        //     // printf("debug idx%d!\n",idx);
        // // }
        // // printf("debug indexing matrix at %d!\n",idx*(pos_dim+1));
        // matrix[idx * (pos_dim + 1) + remainder] = r;

        //TODO RESTORE THIS PART ABOVE

        //using two matrices for indices and weights
        int index_in_m_entries=hash_table.insert(key); //the slot in which it will be inserted is linearly increasing with an atomicAdd
        // printf("got index_in_m_entires %d \n", index_in_m_entries);
        // splatting_indices[idx * (pos_dim + 1) + remainder]=index_in_m_entries; //for the moment this insex indexes in m_entries but after splat_cache it will index in m_keys
        if(index_in_m_entries>=0){
            if( write_new_indices_and_weights ){
                splatting_indices[idx * (pos_dim + 1) + remainder]=hash_table.m_entries[index_in_m_entries]; //it indexes in m_keys
                splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
            }
        }else{
            // printf("position %d could not be inserted\n", idx);
        }

        // //store things 
        // float weight=barycentric[remainder];
        // splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + remainder*2 + 0] = index_in_m_entries;
        // splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + remainder*2 + 1] = weight;

    }


}




template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
splatCache(const int n, const float *values, float* splatting_indices_and_weights,  HashTableGPU hash_table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int threadId = threadIdx.x;
    const int color = blockIdx.y;
    const bool outOfBounds = (idx >= n);

    __shared__ int sharedOffsets[BLOCK_SIZE];
    __shared__ float sharedValues[BLOCK_SIZE * (val_dim+1)];
    int myOffset = -1;
    float *myValue = sharedValues + threadId * (val_dim+1);


    int splatting_idx=round(splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0]);

    //check if the pointer into the entries array is valid. it may not be valid for points that have not splatted due to some race condition
    if (!outOfBounds && splatting_idx>=0) {

        float *value = const_cast<float *>(values + idx * val_dim);
        // float weight = splatting_weights[idx * (pos_dim + 1) + color];
        float weight = splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 1];


        // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
        int index_to_hash_table = hash_table.m_entries[splatting_idx]; //not this indexes into m_keys and m_values
        // splatting_indices[ idx * (pos_dim + 1) + color] = index_to_hash_table; 
        splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0] = index_to_hash_table; 
        splatting_idx=index_to_hash_table;
        // matrix[idx * (pos_dim + 1) + color].index = r.index = hash_table.m_entries[splatting_idx];
        // record the offset into the keys/values array in shared space
        myOffset = sharedOffsets[threadId] = splatting_idx * (val_dim+1);

        for (int j = 0; j < val_dim; j++) {
            myValue[j] = value[j] * weight;
        }
        myValue[val_dim] = weight;

    } else {
        sharedOffsets[threadId] = -1;
    }

    __syncthreads();

    // am I the first thread in this block to care about this key?
    if (outOfBounds || splatting_idx<0)
        return;

    #pragma unroll // makes is somewhat faster from 54ms on the dog picture to 43
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (i < threadId) {
            if (myOffset == sharedOffsets[i]) {
                // somebody else with higher priority cares about this key
                return;
            }
        } else if (i > threadId) {
            if (myOffset == sharedOffsets[i]) {
                // someone else with lower priority cares about this key, accumulate it into mine
                for (int j = 0; j < val_dim+1; j++) {
                    sharedValues[threadId * (val_dim+1) + j] += sharedValues[i * (val_dim+1) + j];
                }
            }
        }
    }

    // only the threads with something to write to main memory are still going
    float *val = hash_table.m_values + myOffset;
    for (int j = 0; j < val_dim+1; j++) {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
            // #warning CUDA ARCH IS FINE
            atomicAdd(val + j, myValue[j]);
        #else 
            #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        #endif
    }
}

template<int pos_dim, int val_dim>
// __global__ void splatCache(const int n, const float *values, int* splatting_indices, float* splatting_weights, HashTableGPU hash_table) {
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
splatCacheNaive(const int nr_positions, float *values, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    //each positions will splat onto pos_dim+1 vertices
    float *my_value = values + idx * val_dim;
    for(int color=0; color<pos_dim+1; color++){
        // int index_into_m_entries=round(splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0]);
        // printf("indexing at %d!\n", idx * (pos_dim + 1) + color);
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];
        // if(splatting_idx>0 && splatting_idx<*hash_table.m_nr_filled){
        if(splatting_idx>=0 ){
            // int splatting_idx=hash_table.m_entries[index_into_m_entries];
            // splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0]=splatting_idx;
            // splatting_indices[ idx * (pos_dim + 1) + color]=splatting_idx;

            // float weight = splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 1];
            float weight = splatting_weights[ idx * (pos_dim + 1) + color];
            float *valOut = hash_table.m_values + splatting_idx * val_dim;

            // printf("idx is %d, color is %d my_value is %f and weight is %f \n", idx, color, *my_value, weight );

            //acumulate the values
            for (int j = 0; j < val_dim; j++) {
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
                    // #warning CUDA ARCH IS FINE
                    atomicAdd(valOut +j, my_value[j]*weight);
                    // atomicAdd(valOut +j, 0);
                #else 
                    #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
                #endif
            }
        
        }else{
            // printf("idx is %d, has an invalid splatting_idx at position %d and the splatting idx is %d \n", idx, idx * (pos_dim + 1) + color, splatting_idx );

        }

    }
    
}

template<int pos_dim, int val_full_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
blur(int n, float *newValues, int remainder,  HashTableGPU hash_table) {

    // const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a lattice vertex
    if (idx >= n)
        return;

    if (idx >= *hash_table.m_nr_filled){
        return;
    }

    // // Check if I'm valid
    // if (matrix[idx].index != idx)
    //     return;


    // find my key and the keys of my neighbors
    int myKey[pos_dim + 1];
    int np[pos_dim + 1];
    int nm[pos_dim + 1];

    
    //store the values of this current lattice vertex (the one at the center of the kernel)
    float *valMe = hash_table.m_values + val_full_dim * idx;
    float *valOut = newValues + val_full_dim * idx;


    for (int i = 0; i < pos_dim; i++) {
        myKey[i] = hash_table.m_keys[idx * pos_dim + i];
        np[i] = myKey[i] + 1;
        nm[i] = myKey[i] - 1;
    }
    np[remainder] -= pos_dim + 1;
    nm[remainder] += pos_dim + 1;
    // np[remainder] = myKey[remainder] - pos_dim;
    // nm[remainder] = myKey[remainder] + pos_dim;

    int offNp = hash_table.retrieve(np);
    int offNm = hash_table.retrieve(nm);


    //in case neighbours don't exist (lattice edges) offNp and offNm are -1
    float zeros[val_full_dim]{0};
    float *valNp = zeros; //or valMe? for edges?
    float *valNm = zeros;
    if(offNp >= 0){
        valNp = hash_table.m_values + val_full_dim * offNp;
    }
    if(offNm >= 0){
        valNm = hash_table.m_values + val_full_dim * offNm;
    }


    // printf(" idx %d !\n", idx);
    for (int i = 0; i < val_full_dim; i++){
        valOut[i] = 0.25 * valNp[i] + 0.5 * valMe[i] + 0.25 * valNm[i];
        // valOut[i] = 1.0 * valMe[i];
    }

}

template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
convolve(int n, float *newValues, const float* filter_bank, const int nr_filters, const int filter_extent, HashTableGPU hash_table) {

    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    if (idx >= n)
        return;

    if (idx >= *hash_table.m_nr_filled){
        return;
    }

    // // Check if I'm valid
    // if (matrix[idx].index != idx)
    //     return;


    // find my key and the keys of my neighbors
    int myKey[pos_dim + 1];
    for (int i = 0; i < pos_dim; i++) {
        myKey[i] = hash_table.m_keys[idx * pos_dim + i];
    }
    int np[pos_dim + 1];
    int nm[pos_dim + 1];

    
    //store the values of this current lattice vertex (the one at the center of the kernel)
    float *valMe = hash_table.m_values + (val_dim+1) * idx;
    float *valOut = newValues + (nr_filters+1) * idx;

    float zeros[val_dim+1]{0};
    int nr_immediate_neigbhours=2*(pos_dim+1);
    int nr_axes=pos_dim+1;
    int idx_neigbhour=0; //if there are 6 neighbours in total (in the case of pos_dim being 2), this will be in range [0,5]
    // printf("nr_axes is %d!\n",nr_axes);
    for(int axis=0; axis<nr_axes; axis++){
        //for each axis we have 2 neighbours

        for (int i = 0; i < pos_dim; i++) {
            np[i] = myKey[i] + 1;
            nm[i] = myKey[i] - 1;
        }
        np[axis] -= pos_dim + 1;
        nm[axis] += pos_dim + 1;

        int offNp = hash_table.retrieve(np);
        int offNm = hash_table.retrieve(nm);

        //in case neighbours don't exist (lattice edges) offNp and offNm are -1
        float *valNp = zeros; //or valMe? for edges?
        float *valNm = zeros;
        //each neigbhour gets multiplied with the weight in the filter bank sequencially from 0 to filter_extent-1 (the last weight is for the center lattice vertex)
        if(offNp >= 0){
            valNp = hash_table.m_values + (val_dim+1) * offNp;

            //multiply the values of neighbour 1
            // printf("nr_filters is %f!\n",nr_filters);
            for(int idx_filter=0; idx_filter<nr_filters; idx_filter++){
                for (int i = 0; i < val_dim+1; i++){
                    int idx_weight = idx_filter*(filter_extent*(val_dim+1) ) + idx_neigbhour*(val_dim+1) + i;  //filter bank has sizes nr_filters x filter_extent x in_val_dim where in_val_dim is the fastest changing index
                    float weight=filter_bank[idx_weight];
                    // printf("weight is %f!\n",weight);
                    // printf("valNp is %f!\n",valNp[i]);
                    valOut[idx_filter] += weight * valNp[i];
                }
            }

        }
        idx_neigbhour++;
        if(offNm >= 0){
            valNm = hash_table.m_values + (val_dim+1) * offNm;

            //multiply the values of neighbour 2
            for(int idx_filter=0; idx_filter<nr_filters; idx_filter++){
                for (int i = 0; i < val_dim+1; i++){
                    int idx_weight = idx_filter*(filter_extent*(val_dim+1) ) + idx_neigbhour*(val_dim+1) + i;  //filter bank has sizes nr_filters x filter_extent x in_val_dim where in_val_dim is the fastest changing index
                    float weight=filter_bank[idx_weight];
                    valOut[idx_filter] += weight * valNm[i];
                }
            }
        }
        idx_neigbhour++;


    }

    //multiply the values for the center vertex
    for(int idx_filter=0; idx_filter<nr_filters; idx_filter++){
        for (int i = 0; i < val_dim+1; i++){
            int idx_weight = idx_filter*(filter_extent*(val_dim+1) ) + idx_neigbhour*(val_dim+1) + i;  //filter bank has sizes nr_filters x filter_extent x in_val_dim where in_val_dim is the fastest changing index
            float weight=filter_bank[idx_weight];
            valOut[idx_filter] += weight * valMe[i];
            // printf("weight is %f!\n",weight);
            // printf("valOut[idx_filter] is %f!\n",valOut[idx_filter]);
        }
    }

    //copy the value for the homogeneous coordinate
    valOut[nr_filters] = valMe[val_dim];


    // for (int i = 0; i < pos_dim; i++) {
    //     myKey[i] = hash_table.m_keys[idx * pos_dim + i];
    //     np[i] = myKey[i] + 1;
    //     nm[i] = myKey[i] - 1;
    // }
    // np[remainder] -= pos_dim + 1;
    // nm[remainder] += pos_dim + 1;

    // int offNp = hash_table.retrieve(np);
    // int offNm = hash_table.retrieve(nm);


    // //in case neighbours don't exist (lattice edges) offNp and offNm are -1
    // float zeros[val_dim+1]{0};
    // float *valNp = zeros; //or valMe? for edges?
    // float *valNm = zeros;
    // if(offNp >= 0){
    //     valNp = hash_table.m_values + (val_dim+1) * offNp;
    // }
    // if(offNm >= 0){
    //     valNm = hash_table.m_values + (val_dim+1) * offNm;
    // }


    // // printf(" idx %d !\n", idx);
    // for (int i = 0; i < val_dim+1; i++){
    //     valOut[i] = 0.25 * valNp[i] + 0.5 * valMe[i] + 0.25 * valNm[i];
    
    // }

}


template<int pos_dim, int val_dim, int filter_extent>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
depthwise_convolve(int nr_vertices, float* values_out, float* filter_bank, int dilation, HashTableGPU hash_table_query, HashTableGPU hash_table_neighbours, const int query_lvl, const int neighbours_lvl, const bool use_center_vertex_from_lattice_neigbhours, bool flip_neighbours, bool debug_kernel) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a lattice vertex
    if (idx >= nr_vertices) return;
    if (idx >= *hash_table_query.m_nr_filled) return;

    //filter bank is a matrix of size filter_extent x val_dim. We proload it into shared mem
    __shared__ float filter_shared[filter_extent*val_dim];
    if (threadIdx.x == 0 ){
        for (int i = 0; i < filter_extent*val_dim; i++) {
            filter_shared[i]=filter_bank[i];
        }
    }
    __syncthreads();

    float val_out_local[val_dim]{0}; //we store the convolved vertex output here first because this will get stored as a register and then we finally copy it to global memory in the val_out_cur_vertex

    

    // int row_size=filter_extent*val_dim; // each row contains a patch around the current lattice vertex that contains the values of all the neighbours in the filter extent (and the center vertex)
    float *val_out_cur_vertex = values_out + val_dim * idx;


    // find my key (from the hash_table_query) and the keys of my neighbors (from the hash_table_neighbours). The hash tables can actually be the same
    float key_query_float[pos_dim + 1];
    float key_sum=0;
    for (int i = 0; i < pos_dim; i++) {
        key_query_float[i] = hash_table_query.m_keys[idx * pos_dim + i];
        key_sum+=key_query_float[i];
    }
    key_query_float[pos_dim]= -key_sum;


    int lvl_diff=query_lvl-neighbours_lvl; 
    float scale=pow(2.0f, (float)lvl_diff); 
    // printf("scale is %f \n", scale);
    for (int i = 0; i < pos_dim+1; i++) {
        key_query_float[i] = key_query_float[i]*scale; 
    }
    // printf("scaled key at idx %d is %f  %f  %f %f  \n",idx, key_query[0],key_query[1],key_query[2], key_query[3]);

    // if (scale < 1.0){
    //     printf("scaled key at idx %d is %f  %f  %f %f  \n",idx, key_query_float[0],key_query_float[1],key_query_float[2], key_query_float[3]);
    // }


    //if the scale is smaller than 1.0 it means we are in a fine scale which we are trying to embedd in a coarser one. Therefore we are multiplying the keys by 0.5 and then moving in each axis by 0.5. However this division by 2 may still create integer key so when moving by 0.5 in a direction we will end up with fractional key. These keys that are stil integers correspond with vertices from the fine that lie directly on top of the coarse vertex when dividing then by 2. But when we convolve over the coarse vertices these are not taken into account which may be a mistake. Either way for the moment we shall ignore them 
    bool has_all_coords_integer=true;
    if(scale<1.0){
       has_all_coords_integer=are_all_coords_integer(key_query_float, pos_dim+1);
    }


    // bool has_one_coord_integer=is_only_one_coord_integer(key_query_float, pos_dim+1);
    // if (has_one_coord_integer){
    //     printf("scaled key at idx %d is %f  %f  %f %f  \n",idx, key_query_float[0],key_query_float[1],key_query_float[2], key_query_float[3]);
    // }
    // int nr_coords_integer_val=nr_coords_integer(key_query_float, pos_dim+1);
    // //The nr_coords integer is either pos_dim+1 or 0
    // if (scale <1.0 && ( nr_coords_integer_val!=pos_dim+1 && nr_coords_integer_val!=0 ) ){
    //     printf("nr_coords_integer_val is %d \n", nr_coords_integer_val);
    //     printf("scaled key at idx %d is %f  %f  %f %f  \n",idx, key_query_float[0],key_query_float[1],key_query_float[2], key_query_float[3]);
    // }
    




    


    int np[pos_dim + 1];
    int nm[pos_dim + 1];


    if(debug_kernel){
        // printf("val_full_dim is %d\n", val_full_dim);
        // printf("query lvl is %d\n", query_lvl);
        // printf("neighbours lvl is %d\n", neighbours_lvl);
        // printf("dilation is %d\n", dilation);
        // printf("pos_dim is %d\n", pos_dim);
        // printf("scale is %f \n", scale);
    }

    // printf("val_full_dim is %d\n", val_full_dim);


    
    //store the values of this current lattice vertex (the one at the center of the kernel)
    // float zeros[val_dim]{0};
    float* valMe ;
    bool center_vertex_has_valid_value=false; //valid value means a non zero one. It will save us from writing a value of zero which is redundant
    int key_query_int[pos_dim + 1];
    for (int i = 0; i < pos_dim+1; i++) {
        key_query_int[i] = round(key_query_float[i]);
    }
    
    //if we have fractional coords it means we are in a fine lattice which is embeddeed in a coarser one. So the keys were multiplied by 0.5 or something like that. That means that we will not find any center vertex
    if(use_center_vertex_from_lattice_neigbhours && has_all_coords_integer){
        int query_offset = hash_table_neighbours.retrieve(key_query_int); 
        if(query_offset>=0){
            valMe = hash_table_neighbours.m_values + val_dim * query_offset;
            center_vertex_has_valid_value=true;
        }
    }else if (!use_center_vertex_from_lattice_neigbhours){
        valMe= hash_table_query.m_values + val_dim * idx;
        center_vertex_has_valid_value=true;
    }


    bool should_check_neighbours=true;
    if(scale>=1.0){ //we are convolving a lattice of the same scale ,or we are querying from a coarser to a finer one. so we definitelly check the neighbours
        should_check_neighbours=true;
    }else if(scale<1.0 && has_all_coords_integer){ //we are in a fine lattice that has the query as all integer, means that whne we move by 0.5 in every axis we won't have an integer neighbour anymore. therefore a coarser vertex will not be there
        should_check_neighbours=false;
    }else if(scale<1.0 && !has_all_coords_integer){
        should_check_neighbours=true; //We have a fractional key, but when checking the neigbhours we will move by 0.5 and therefore end up with an integer key
    }

    // if(scale<1.0){
    //     printf("should_check_neighbours is %d \n", should_check_neighbours);
    // }



    int nr_immediate_neigbhours=2*(pos_dim+1);
    const int nr_axes=pos_dim+1;
    // int idx_neigbhour=0; //if there are 6 neighbours in total (in the case of pos_dim being 2), this will be in range [0,5]
    // printf("nr_axes is %d!\n",nr_axes);
    int nr_neighbours_found=0;
    float movement_multiplier=1.0;
    if(scale<1.0){ //if the scale is fractional than the movement also has to be fractional in order to ensure we end up with a integer key
        movement_multiplier=scale;
    }
    float np_float[pos_dim + 1];
    float nm_float[pos_dim + 1];
    if( should_check_neighbours ){
        for(int axis=0; axis<nr_axes; axis++){
            //for each axis we have 2 neighbours

            // //chekc first if the neigbhours have integer coords 
            // for (int i = 0; i < pos_dim+1; i++) {
            //     np_float[i] = key_query_float[i] + movement_multiplier*dilation;
            //     nm_float[i] = key_query_float[i] - movement_multiplier*dilation;
            // }
            // np_float[axis] = key_query_float[axis] - movement_multiplier*dilation*pos_dim;
            // nm_float[axis] = key_query_float[axis] + movement_multiplier*dilation*pos_dim;
            // bool np_coords_integer=are_all_coords_integer(np_float, pos_dim+1);
            // bool nm_coords_integer=are_all_coords_integer(nm_float, pos_dim+1);
            // if(!np_coords_integer){
            //     printf("np has no integer coords\n");
            // }

            // //get the integer coords
            // for (int i = 0; i < pos_dim+1; i++) {
            //     np[i] = round(np_float[i]);
            //     nm[i] = round(nm_float[i]);
            // }


            bool np_coords_integer=true;
            bool nm_coords_integer=true;
            //if the pos_dim+1 is even, eg 4, then pos dim is 3 which kinda the usual case we work with. In this case we just get the keys for the neighbours
            if( (pos_dim+1)%2==0 ){
                for (int i = 0; i < pos_dim+1; i++) {
                    np[i] = round(key_query_float[i] + movement_multiplier*dilation);
                    nm[i] = round(key_query_float[i] - movement_multiplier*dilation);
                }
                np[axis] = round(key_query_float[axis] - movement_multiplier*dilation*pos_dim);
                nm[axis] = round(key_query_float[axis] + movement_multiplier*dilation*pos_dim);
            }else{
                //the pos dim+1 is odd which means that the key_query_float after scaling can be something like 0.5, 0.5, 1.0. Now if we move with a movement_multiplied of 0.5 we end up with non integer key. We should double check for that. This doesnt happen in the case when pos_dim+1 is even because then we can always get a coordinate vector that sums to zero without having fractional coordinates.
                //chekc first if the neigbhours have integer coords 
                for (int i = 0; i < pos_dim+1; i++) {
                    np_float[i] = key_query_float[i] + movement_multiplier*dilation;
                    nm_float[i] = key_query_float[i] - movement_multiplier*dilation;
                }
                np_float[axis] = key_query_float[axis] - movement_multiplier*dilation*pos_dim;
                nm_float[axis] = key_query_float[axis] + movement_multiplier*dilation*pos_dim;
                np_coords_integer=are_all_coords_integer(np_float, pos_dim+1);
                nm_coords_integer=are_all_coords_integer(nm_float, pos_dim+1);
                // if(!np_coords_integer){
                //     printf("np has no integer coords\n");
                // }

                //get the integer coords
                for (int i = 0; i < pos_dim+1; i++) {
                    np[i] = round(np_float[i]);
                    nm[i] = round(nm_float[i]);
                }
            }
            

            int offNp =-1;
            int offNm =-1;

            if (np_coords_integer){
                offNp = hash_table_neighbours.retrieve(np);
            }
            if (nm_coords_integer){
                offNm = hash_table_neighbours.retrieve(nm);
            }

            //in case neighbours don't exist (lattice edges) offNp and offNm are -1
            float *valNp; //or valMe? for edges?
            float *valNm;
            //each neigbhour gets multiplied with the weight in the filter bank sequencially from 0 to filter_extent-1 (the last weight is for the center lattice vertex)
            if(offNp >= 0 && np_coords_integer){
                nr_neighbours_found++;
                valNp = hash_table_neighbours.m_values + val_dim * offNp;

                int neighbour_idx=0;
                if(flip_neighbours){ //for the backwards pass we flip the neighbours so that when multiplying with the kernel they get the weights as if the kernel was centered around the neighbour
                    neighbour_idx=1;
                }
                int idx_within_row= val_dim*axis*2 + neighbour_idx*val_dim;  //there are 2 neigbours per axis and each has val_ful_dim values
                //store the values of neighbour 1
                #pragma unroll
                for (int i = 0; i < val_dim; i++){
                    // int row_idx=idx_within_row +i; //we store each neigbhour values one after another. so if we have neighbour with 3 values each they will be in a row stored as n1v1, n1v2, n1v3, n2v1, n2v2 etc
                    // row_out[row_idx] = valNp[i];
                    val_out_local[i] += valNp[i]* filter_shared[idx_within_row + i];
                }
            }else{
                // if(debug_kernel){
                //     printf("not found neighbour np\n");
                // }
            }

            // idx_neigbhour++;

            if(offNm >= 0 && nm_coords_integer){
                nr_neighbours_found++;
                valNm = hash_table_neighbours.m_values + val_dim * offNm;

                int neighbour_idx=1;
                if(flip_neighbours){ //for the backwards pass we flip the neighbours so that when multiplying with the kernel they get the weights as if the kernel was centered around the neighbour
                    neighbour_idx=0;
                }
                int idx_within_row= val_dim*axis*2 + neighbour_idx*val_dim;  //there are 2 neigbours per axis and each has val_ful_dim values
                //store the values of neighbour 2
                #pragma unroll
                for (int i = 0; i < val_dim; i++){
                    // int row_idx=idx_within_row +i; //we store each neigbhour values one after another. so if we have neighbour with 3 values each they will be in a row stored as n1v1, n1v2, n1v3, n2v1, n2v2 etc
                    // row_out[row_idx] = valNm[i];
                    val_out_local[i] += valNm[i]* filter_shared[idx_within_row + i];
                }
            }else{
                // if(debug_kernel){
                //     printf("not found neighbour nm\n");
                // }
            }



        }
    }


    if(debug_kernel){
        // printf("for idx %d of the query keys we have found %d neigbhours in the neighbours keys\n", idx, nr_neighbours_found);
    }

    if(nr_neighbours_found==0){
        // printf("didn't find any neigbhours for key at idx %d, dilation %d with key %f  %f  %f %f  \n",idx, dilation, key_query[0],key_query[1],key_query[2], key_query[3]);
    }

    int idx_within_row= val_dim*nr_immediate_neigbhours; //we skip the values of all the neighbours that we stored in the row and now we are pointing to the position in the row where we can write 
    //store the values of the center vertex
    if(center_vertex_has_valid_value){
        #pragma unroll
        for (int i = 0; i < val_dim; i++){
            // int row_idx=idx_within_row +i; //we store each neigbhour values one after another. so if we have neighbour with 3 values each they will be in a row stored as n1v1, n1v2, n1v3, n2v1, n2v2 etc
            // row_out[row_idx] = valMe[i];
            val_out_local[i] += valMe[i]* filter_shared[idx_within_row + i];
        }
    }

    //copy the val_out_local to the output values of that vertex
    for (int i = 0; i < val_dim; i++){
        val_out_cur_vertex[i]=val_out_local[i];
    }



}

template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
im2row(int nr_vertices, float* im2row_out, int filter_extent, int dilation, HashTableGPU hash_table_query, HashTableGPU hash_table_neighbours, const int query_lvl, const int neighbours_lvl,  bool flip_neighbours, bool debug_kernel) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a lattice vertex
    if (idx >= nr_vertices) return;
    if (idx >= *hash_table_query.m_nr_filled) return;
    

    int row_size=filter_extent*val_dim; // each row contains a patch around the current lattice vertex that contains the values of all the neighbours in the filter extent (and the center vertex)
    float *row_out = im2row_out + row_size * idx;


    // find my key (from the hash_table_query) and the keys of my neighbors (from the hash_table_neighbours). The hash tables can actually be the same
    float key_query_float[pos_dim + 1];
    float key_sum=0;
    for (int i = 0; i < pos_dim; i++) {
        key_query_float[i] = hash_table_query.m_keys[idx * pos_dim + i];
        key_sum+=key_query_float[i];
    }
    key_query_float[pos_dim]= -key_sum;


    int lvl_diff=query_lvl-neighbours_lvl; 
    float scale=pow(2.0f, (float)lvl_diff); 
    // printf("scale is %f \n", scale);
    for (int i = 0; i < pos_dim+1; i++) {
        key_query_float[i] = key_query_float[i]*scale; 
    }
    // printf("scaled key at idx %d is %f  %f  %f %f  \n",idx, key_query[0],key_query[1],key_query[2], key_query[3]);

    // if (scale < 1.0){
    //     printf("scaled key at idx %d is %f  %f  %f %f  \n",idx, key_query_float[0],key_query_float[1],key_query_float[2], key_query_float[3]);
    // }


    //if the scale is smaller than 1.0 it means we are in a fine scale which we are trying to embedd in a coarser one. Therefore we are multiplying the keys by 0.5 and then moving in each axis by 0.5. However this division by 2 may still create integer key so when moving by 0.5 in a direction we will end up with fractional key. These keys that are stil integers correspond with vertices from the fine that lie directly on top of the coarse vertex when dividing then by 2. But when we convolve over the coarse vertices these are not taken into account which may be a mistake. Either way for the moment we shall ignore them 
    bool has_all_coords_integer=true;
    if(scale<1.0){
       has_all_coords_integer=are_all_coords_integer(key_query_float, pos_dim+1);
    }



    int np[pos_dim + 1];
    int nm[pos_dim + 1];


    if(debug_kernel){
        // printf("val_full_dim is %d\n", val_full_dim);
        // printf("query lvl is %d\n", query_lvl);
        // printf("neighbours lvl is %d\n", neighbours_lvl);
        // printf("dilation is %d\n", dilation);
        // printf("pos_dim is %d\n", pos_dim);
        // printf("scale is %f \n", scale);
    }

    
    //store the values of this current lattice vertex (the one at the center of the kernel)
    // float zeros[val_dim]{0};
    float* valMe ;
    bool center_vertex_has_valid_value=false; //valid value means a non zero one. It will save us from writing a value of zero which is redundant
    int key_query_int[pos_dim + 1];
    for (int i = 0; i < pos_dim+1; i++) {
        key_query_int[i] = round(key_query_float[i]);
    }
    
    //if we have fractional coords it means we are in a fine lattice which is embeddeed in a coarser one. So the keys were multiplied by 0.5 or something like that. That means that we will not find any center vertex
    // if(use_center_vertex_from_lattice_neigbhours && has_all_coords_integer){
    if(has_all_coords_integer){
        int query_offset = hash_table_neighbours.retrieve(key_query_int); 
        if(query_offset>=0){
            valMe = hash_table_neighbours.m_values + val_dim * query_offset; //the hash_table_neighbours can actually be pointing to the one of the current lattice
            center_vertex_has_valid_value=true;
        }
    }
    // }else if (!use_center_vertex_from_lattice_neigbhours){
    //     valMe= hash_table_query.m_values + val_dim * idx;
    //     center_vertex_has_valid_value=true;
    // }


    bool should_check_neighbours=true;
    if(scale>=1.0){ //we are convolving a lattice of the same scale ,or we are querying from a coarser to a finer one. so we definitelly check the neighbours
        should_check_neighbours=true;
    }else if(scale<1.0 && has_all_coords_integer){ //we are in a fine lattice that has the query as all integer, means that whne we move by 0.5 in every axis we won't have an integer neighbour anymore. therefore a coarser vertex will not be there
        should_check_neighbours=false;
    }else if(scale<1.0 && !has_all_coords_integer){
        should_check_neighbours=true; //We have a fractional key, but when checking the neigbhours we will move by 0.5 and therefore end up with an integer key
    }



    int nr_immediate_neigbhours=2*(pos_dim+1);
    const int nr_axes=pos_dim+1;
    int nr_neighbours_found=0;
    float movement_multiplier=1.0;
    if(scale<1.0){ //if the scale is fractional than the movement also has to be fractional in order to ensure we end up with a integer key
        movement_multiplier=scale;
    }
    float np_float[pos_dim + 1];
    float nm_float[pos_dim + 1];
    if( should_check_neighbours ){
        for(int axis=0; axis<nr_axes; axis++){
            //for each axis we have 2 neighbours

            bool np_coords_integer=true;
            bool nm_coords_integer=true;
            //if the pos_dim+1 is even, eg 4, then pos dim is 3 which kinda the usual case we work with. In this case we just get the keys for the neighbours
            if( (pos_dim+1)%2==0 ){
                for (int i = 0; i < pos_dim+1; i++) {
                    np[i] = round(key_query_float[i] + movement_multiplier*dilation);
                    nm[i] = round(key_query_float[i] - movement_multiplier*dilation);
                }
                np[axis] = round(key_query_float[axis] - movement_multiplier*dilation*pos_dim);
                nm[axis] = round(key_query_float[axis] + movement_multiplier*dilation*pos_dim);
            }else{
                //the pos dim+1 is odd which means that the key_query_float after scaling can be something like 0.5, 0.5, 1.0. Now if we move with a movement_multiplied of 0.5 we end up with non integer key. We should double check for that. This doesnt happen in the case when pos_dim+1 is even because then we can always get a coordinate vector that sums to zero without having fractional coordinates.
                //chekc first if the neigbhours have integer coords 
                for (int i = 0; i < pos_dim+1; i++) {
                    np_float[i] = key_query_float[i] + movement_multiplier*dilation;
                    nm_float[i] = key_query_float[i] - movement_multiplier*dilation;
                }
                np_float[axis] = key_query_float[axis] - movement_multiplier*dilation*pos_dim;
                nm_float[axis] = key_query_float[axis] + movement_multiplier*dilation*pos_dim;
                np_coords_integer=are_all_coords_integer(np_float, pos_dim+1);
                nm_coords_integer=are_all_coords_integer(nm_float, pos_dim+1);
                // if(!np_coords_integer){
                //     printf("np has no integer coords\n");
                // }

                //get the integer coords
                for (int i = 0; i < pos_dim+1; i++) {
                    np[i] = round(np_float[i]);
                    nm[i] = round(nm_float[i]);
                }
            }
            

            int offNp =-1;
            int offNm =-1;

            if (np_coords_integer){
                offNp = hash_table_neighbours.retrieve(np);
            }
            if (nm_coords_integer){
                offNm = hash_table_neighbours.retrieve(nm);
            }

            //in case neighbours don't exist (lattice edges) offNp and offNm are -1
            float *valNp; //or valMe? for edges?
            float *valNm;
            //each neigbhour gets multiplied with the weight in the filter bank sequencially from 0 to filter_extent-1 (the last weight is for the center lattice vertex)
            if(offNp >= 0 && np_coords_integer){
                nr_neighbours_found++;
                valNp = hash_table_neighbours.m_values + val_dim * offNp;

                int neighbour_idx=0;
                if(flip_neighbours){ //for the backwards pass we flip the neighbours so that when multiplying with the kernel they get the weights as if the kernel was centered around the neighbour
                    neighbour_idx=1;
                }
                int idx_within_row= val_dim*axis*2 + neighbour_idx*val_dim;  //there are 2 neigbours per axis and each has val_ful_dim values
                //store the values of neighbour 1
                #pragma unroll
                for (int i = 0; i < val_dim; i++){
                    int row_idx=idx_within_row +i; //we store each neigbhour values one after another. so if we have neighbour with 3 values each they will be in a row stored as n1v1, n1v2, n1v3, n2v1, n2v2 etc
                    row_out[row_idx] = valNp[i];
                }
            }else{
                // if(debug_kernel){
                //     printf("not found neighbour np\n");
                // }
            }

            // idx_neigbhour++;

            if(offNm >= 0 && nm_coords_integer){
                nr_neighbours_found++;
                valNm = hash_table_neighbours.m_values + val_dim * offNm;

                int neighbour_idx=1;
                if(flip_neighbours){ //for the backwards pass we flip the neighbours so that when multiplying with the kernel they get the weights as if the kernel was centered around the neighbour
                    neighbour_idx=0;
                }
                int idx_within_row= val_dim*axis*2 + neighbour_idx*val_dim;  //there are 2 neigbours per axis and each has val_ful_dim values
                //store the values of neighbour 2
                #pragma unroll
                for (int i = 0; i < val_dim; i++){
                    int row_idx=idx_within_row +i; //we store each neigbhour values one after another. so if we have neighbour with 3 values each they will be in a row stored as n1v1, n1v2, n1v3, n2v1, n2v2 etc
                    row_out[row_idx] = valNm[i];
                }
            }else{
                // if(debug_kernel){
                //     printf("not found neighbour nm\n");
                // }
            }



        }
    }


    if(debug_kernel){
        // printf("for idx %d of the query keys we have found %d neigbhours in the neighbours keys\n", idx, nr_neighbours_found);
    }

    if(nr_neighbours_found==0){
        // printf("didn't find any neigbhours for key at idx %d, dilation %d with key %f  %f  %f %f  \n",idx, dilation, key_query[0],key_query[1],key_query[2], key_query[3]);
    }

    int idx_within_row= val_dim*nr_immediate_neigbhours; //we skip the values of all the neighbours that we stored in the row and now we are pointing to the position in the row where we can write 
    //store the values of the center vertex
    if(center_vertex_has_valid_value){
        #pragma unroll
        for (int i = 0; i < val_dim; i++){
            int row_idx=idx_within_row +i; //we store each neigbhour values one after another. so if we have neighbour with 3 values each they will be in a row stored as n1v1, n1v2, n1v3, n2v1, n2v2 etc
            row_out[row_idx] = valMe[i];
        }
    }



}


template<int pos_dim, int val_full_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
test_row2im(int capacity, float* im2row_in, int filter_extent, int dilation, HashTableGPU hash_table_query, HashTableGPU hash_table_neighbours, const int query_lvl, const int neighbours_lvl, const bool use_center_vertex) {

    // const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a lattice vertex
    if (idx >= capacity)
        return;
    if (idx >= *hash_table_query.m_nr_filled){
        return;
    }


    //to go from lattice rowified to the original lattice we sum over all the places where we duplicated the values when we did a im2row
    // so for example  a lattice like this with 3 vertices each with 2 values
    // # .------.------.
    // # | L1V1 | L1V2 |
    // # :------+------:
    // # | L2V1 | L2V2 |
    // # :------+------:
    // # | L3V1 | L3V2 |
    // # '------'------
    //may get rowified like this 
    // # .------.------.------.------.------.------.------.------.------.------.------.------.-------.------.
    // # | 0    | 0    | L3V1 | L3V2 | 0    | 0    | 0    | 0    | L2V1 | L2V2 | 0    | 0    | L1V1  | L1V2 |
    // # :------+------+------+------+------+------+------+------+------+------+------+------+-------+------:
    // # | 0    | 0    | 0    | 0    | 0    | 0    | L3V1 | L3V2 | 0    | 0    | L1V1 | L1V2 | L2V1  | L2V2 |
    // # :------+------+------+------+------+------+------+------+------+------+------+------+-------+------:
    // # | L1V1 | L1V2 | 0    | 0    | L2V1 | L2V2 | 0    | 0    | 0    | 0    | 0    | 0    | L3V1  | L3V2 |
    // # '------'------'------'------'------'------'------'------'------'------'------'------'-------'------'
    //not to go back to the original values we could just grab the last 2 columns but since the main usage of row2im is agregation of errors for the backwards pass, we will sum the elements that are duplicated. So to get the lattice values at elements (0,0) so the position of L1V1 we will sum in the lattice rowified over the positions (0,13), (1,11) and (2,0) which is all the places where L1V1 got copied. 
    //to get this we calculate for lattice vertex, the neigbhour and we now that those neighbours will be responsible of adding MyValue on their corresponding row. We have to find out on which position on the row of the neighbour MyVlue will be copied into and sum over the val_full_dim values that are there

    int row_size=filter_extent*val_full_dim; // each row contains a patch around the current lattice vertex that contains the values of all the neighbours in the filter extent (and the center vertex)


    // find my key (from the hash_table_query) and the keys of my neighbors (from the hash_table_neighbours). The hash tables can actually be the same
    int key_query[pos_dim + 1];
    for (int i = 0; i < pos_dim; i++) {
        key_query[i] = hash_table_query.m_keys[idx * pos_dim + i];
    }
    //scale my key in case the neighbours is at a finer lvl
    int lvl_diff=query_lvl-neighbours_lvl; //will be strictly positive because query is coarsen and neigbhour is finer
    if(lvl_diff<0){
        printf("At the moment we don't support the query being finer. But maybe we should for the backward pass of the coarsening");
    }
    int scale=1;
    //for soem reason pow doesnt work with nvrtc
    for (int i = 0; i < lvl_diff; i++) {
        scale*=2;
    }
    for (int i = 0; i < pos_dim+1; i++) {
        key_query[i] = key_query[i]*scale; 
    }
    


    int np[pos_dim + 1];
    int nm[pos_dim + 1];

    
    //store the values of this current lattice vertex (the one at the center of the kernel)
    float *val_me_out = hash_table_query.m_values + val_full_dim * idx;

    float zeros[val_full_dim]{0};
    int nr_immediate_neigbhours=2*(pos_dim+1);
    int nr_axes=pos_dim+1;
    int idx_neigbhour=0; //if there are 6 neighbours in total (in the case of pos_dim being 2), this will be in range [0,5]
    // printf("nr_axes is %d!\n",nr_axes);
    for(int axis=0; axis<nr_axes; axis++){
        //for each axis we have 2 neighbours

        for (int i = 0; i < pos_dim+1; i++) {
            np[i] = key_query[i] + dilation;
            nm[i] = key_query[i] - dilation;
        }
        np[axis] = key_query[axis] - dilation*pos_dim;
        nm[axis] = key_query[axis] + dilation*pos_dim;

        int offNp = hash_table_neighbours.retrieve(np);
        int offNm = hash_table_neighbours.retrieve(nm);

        //in case neighbours don't exist (lattice edges) offNp and offNm are -1
        float *valNp = zeros; //or valMe? for edges?
        float *valNm = zeros;


        //neighbour1---
        if(offNp >= 0){
            //a neighbour exits at position offNp which means that the meValue was copied somewhere in the row offNp of lattice rowified
            //to ge the position inside the row:
            //we know that the neigbhour will see the current vertex which lies on the same axis so the axis_idx will be the same, now we only have to decide if it's the positive neigbhour or the negative one
            //Since the neigbhour is my positive one (np) then for the neighbour I am the negative one Nm
            //Therefore the start position within the row where I will start to find my values will be at 
            float *row = im2row_in + row_size * offNp; //this is the row where we will find our value
            int idx_within_row= val_full_dim*axis*2 + 1*val_full_dim;  //we have 2 neighbour per row and each have val_full_dim, and we should skip axis_idx nr of them. The +1*Val_full_dum is because we skip one more value chunk because this vertex will be seen as Nm for the neigbhour

            float* start_of_my_values=row + idx_within_row;
            for (int i = 0; i < val_full_dim; i++){
                if(val_me_out[i] != start_of_my_values[i]){
                    printf("the values don't correspond");
                }
            }

            //in the case of row2im we would have to summ all the start_of_my_values into val_me_out

        }
      

        //neighbour 2
        if(offNm >= 0){

            float *row = im2row_in + row_size * offNm; //this is the row where we will find our value
            int idx_within_row= val_full_dim*axis*2 + 0*val_full_dim;  //we have 2 neighbour per row and each have val_full_dim, and we should skip axis_idx nr of them. The +1*Val_full_dum is because we skip one more value chunk because this vertex will be seen as Nm for the neigbhour

            float* start_of_my_values=row + idx_within_row;
            for (int i = 0; i < val_full_dim; i++){
                if(val_me_out[i] != start_of_my_values[i]){
                    printf("the values don't correspond");
                }
            }
        }

    }


    if(use_center_vertex){
        // printf("testing the center vertex\n");
        //on the current row we also store the values of the vertex 
        float *row = im2row_in + row_size * idx; //this is the row where we will find our value
        int idx_within_row= row_size - val_full_dim; 
        float* start_of_my_values=row + idx_within_row;
        for (int i = 0; i < val_full_dim; i++){
            if(val_me_out[i] != start_of_my_values[i]){
                printf("the values don't correspond");
            }
        }
    }


}



template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
row2im(int capacity, float* im2row_in, int filter_extent, int dilation, HashTableGPU hash_table_query, HashTableGPU hash_table_neighbours, const int query_lvl, const int neighbours_lvl, const bool do_test) {

    // printf("inside row2im use_center_vertex is %d\n", use_center_vertex);
    // printf("capacity is %d",capacity );
    // printf("filled is %d",*hash_table_query.m_nr_filled );

    // const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a lattice vertex
    if (idx >= capacity)
        return;
    if (idx >= *hash_table_query.m_nr_filled){
        return;
    }


    //to go from lattice rowified to the original lattice we sum over all the places where we duplicated the values when we did a im2row
    // so for example  a lattice like this with 3 vertices each with 2 values
    // # .------.------.
    // # | L1V1 | L1V2 |
    // # :------+------:
    // # | L2V1 | L2V2 |
    // # :------+------:
    // # | L3V1 | L3V2 |
    // # '------'------
    //may get rowified like this 
    // # .------.------.------.------.------.------.------.------.------.------.------.------.-------.------.
    // # | 0    | 0    | L3V1 | L3V2 | 0    | 0    | 0    | 0    | L2V1 | L2V2 | 0    | 0    | L1V1  | L1V2 |
    // # :------+------+------+------+------+------+------+------+------+------+------+------+-------+------:
    // # | 0    | 0    | 0    | 0    | 0    | 0    | L3V1 | L3V2 | 0    | 0    | L1V1 | L1V2 | L2V1  | L2V2 |
    // # :------+------+------+------+------+------+------+------+------+------+------+------+-------+------:
    // # | L1V1 | L1V2 | 0    | 0    | L2V1 | L2V2 | 0    | 0    | 0    | 0    | 0    | 0    | L3V1  | L3V2 |
    // # '------'------'------'------'------'------'------'------'------'------'------'------'-------'------'
    //not to go back to the original values we could just grab the last 2 columns but since the main usage of row2im is agregation of errors for the backwards pass, we will sum the elements that are duplicated. So to get the lattice values at elements (0,0) so the position of L1V1 we will sum in the lattice rowified over the positions (0,13), (1,11) and (2,0) which is all the places where L1V1 got copied. 
    //to get this we calculate for lattice vertex, the neigbhour and we now that those neighbours will be responsible of adding MyValue on their corresponding row. We have to find out on which position on the row of the neighbour MyVlue will be copied into and sum over the val_full_dim values that are there



    int row_size=filter_extent*val_full_dim; // each row contains a patch around the current lattice vertex that contains the values of all the neighbours in the filter extent (and the center vertex)


    // find my key (from the hash_table_query) and the keys of my neighbors (from the hash_table_neighbours). The hash tables can actually be the same
    float key_query_float[pos_dim + 1];
    float key_sum=0;
    for (int i = 0; i < pos_dim; i++) {
        key_query_float[i] = hash_table_query.m_keys[idx * pos_dim + i];
        key_sum+=key_query_float[i];
    }
    key_query_float[pos_dim]= -key_sum;


    int lvl_diff=query_lvl-neighbours_lvl; 
    float scale=pow(2.0f, (float)lvl_diff); 
    // printf("scale is %f \n", scale);
    for (int i = 0; i < pos_dim+1; i++) {
        key_query_float[i] = key_query_float[i]*scale; 
    }
    // printf("scaled key at idx %d is %f  %f  %f %f  \n",idx, key_query[0],key_query[1],key_query[2], key_query[3]);


    //if the scale is smaller than 1.0 it means we are in a fine scale which we are trying to embedd in a coarser one. Therefore we are multiplying the keys by 0.5 and then moving in each axis by 0.5. However this division by 2 may still create integer key so when moving by 0.5 in a direction we will end up with fractional key. These keys that are stil integers correspond with vertices from the fine that lie directly on top of the coarse vertex when dividing then by 2. But when we convolve over the coarse vertices these are not taken into account which may be a mistake. Either way for the moment we shall ignore them 
    bool has_all_coords_integer=true;
    if(scale<1.0){
       has_all_coords_integer=are_all_coords_integer(key_query_float, pos_dim+1);
    }


    int np[pos_dim + 1];
    int nm[pos_dim + 1];



    //store the values of this current lattice vertex (the one at the center of the kernel)
    // float zeros[val_full_dim]{0};
    // float* valMe = zeros;
    // bool center_vertex_has_valid_value=false; //valid value means a non zero one. It will save us from writing a value of zero which is redundant
    int key_query_int[pos_dim + 1];
    for (int i = 0; i < pos_dim+1; i++) {
        key_query_int[i] = round(key_query_float[i]);
    }
    
    // //if we have fractional coords it means we are in a fine lattice which is embeddeed in a coarser one. So the keys were multiplied by 0.5 or something like that. That means that we will not find any center vertex
    // if(use_center_vertex_from_lattice_neigbhours && has_all_coords_integer){
    //     int query_offset = hash_table_neighbours.retrieve(key_query_int); 
    //     if(query_offset>=0){
    //         valMe = hash_table_neighbours.m_values + val_full_dim * query_offset;
    //         center_vertex_has_valid_value=true;
    //     }
    // }else if (!use_center_vertex_from_lattice_neigbhours){
    //     valMe= hash_table_query.m_values + val_full_dim * idx;
    //     center_vertex_has_valid_value=true;
    // }



    bool should_check_neighbours=true;
    if(scale>=1.0){ //we are convolving a lattice of the same scale ,or we are querying from a coarser to a finer one. so we definitelly check the neighbours
        should_check_neighbours=true;
    }else if(scale<1.0 && has_all_coords_integer){ //we are in a fine lattice that has the query as all integer, means that whne we move by 0.5 in every axis we won't have an integer neighbour anymore. therefore a coarser vertex will not be there
        should_check_neighbours=false;
    }else if(scale<1.0 && !has_all_coords_integer){
        should_check_neighbours=true; //We have a fractional key, but when checking the neigbhours we will move by 0.5 and therefore end up with an integer key
    }




    
    //store the values of this current lattice vertex (the one at the center of the kernel)
    float *val_me_out = hash_table_query.m_values + val_dim * idx;

    // int nr_immediate_neigbhours=2*(pos_dim+1);
    int nr_axes=pos_dim+1;
    // int idx_neigbhour=0; //if there are 6 neighbours in total (in the case of pos_dim being 2), this will be in range [0,5]
    float movement_multiplier=1.0;
    if(scale<1.0){ //if the scale is fractional than the movement also has to be fractional in order to ensure we end up with a integer key
        movement_multiplier=scale;
    }
    if(should_check_neighbours){
        for(int axis=0; axis<nr_axes; axis++){
            //for each axis we have 2 neighbours

            //for each axis we have 2 neighbours

            for (int i = 0; i < pos_dim+1; i++) {
                np[i] = round(key_query_float[i] + movement_multiplier*dilation);
                nm[i] = round(key_query_float[i] - movement_multiplier*dilation);
            }
            np[axis] = round(key_query_float[axis] - movement_multiplier*dilation*pos_dim);
            nm[axis] = round(key_query_float[axis] + movement_multiplier*dilation*pos_dim);

            int offNp = hash_table_neighbours.retrieve(np);
            int offNm = hash_table_neighbours.retrieve(nm);


            //in case neighbours don't exist (lattice edges) offNp and offNm are -1
            // float *valNp = zeros; //or valMe? for edges?
            // float *valNm = zeros;


            //neighbour1---
            if(offNp >= 0){
                //a neighbour exits at position offNp which means that the meValue was copied somewhere in the row offNp of lattice rowified
                //to ge the position inside the row:
                //we know that the neigbhour will see the current vertex which lies on the same axis so the axis_idx will be the same, now we only have to decide if it's the positive neigbhour or the negative one
                //Since the neigbhour is my positive one (np) then for the neighbour I am the negative one Nm
                //Therefore the start position within the row where I will start to find my values will be at 
                float *row = im2row_in + row_size * offNp; //this is the row where we will find our value
                int idx_within_row= val_dim*axis*2 + 1*val_dim;  //we have 2 neighbour per row and each have val_full_dim, and we should skip axis_idx nr of them. The +1*Val_full_dum is because we skip one more value chunk because this vertex will be seen as Nm for the neigbhour

                float* start_of_my_values=row + idx_within_row;
                for (int i = 0; i < val_dim; i++){
                    if(!do_test){
                        val_me_out[i]+=start_of_my_values[i];
                    }
                    if(do_test){
                        if(val_me_out[i] != start_of_my_values[i]){
                            printf("the values don't correspond. val_me_out is %f and start of my values is %f\n ", val_me_out[i], start_of_my_values[i]);
                        }
                    }
                }

                //in the case of row2im we would have to summ all the start_of_my_values into val_me_out

            }
        

            //neighbour 2
            if(offNm >= 0){

                float *row = im2row_in + row_size * offNm; //this is the row where we will find our value
                int idx_within_row= val_dim*axis*2 + 0*val_dim;  //we have 2 neighbour per row and each have val_full_dim, and we should skip axis_idx nr of them. The +1*Val_full_dum is because we skip one more value chunk because this vertex will be seen as Nm for the neigbhour

                float* start_of_my_values=row + idx_within_row;
                for (int i = 0; i < val_dim; i++){

                    if(!do_test){
                        val_me_out[i]+=start_of_my_values[i];
                    }
                    if(do_test){
                        if(val_me_out[i] != start_of_my_values[i]){
                            printf("the values don't correspond. val_me_out is %f and start of my values is %f\n ", val_me_out[i], start_of_my_values[i]);
                        }
                    }
                }
            }

        }
    }
    

    // printf("inside row2im use_center_vertex is %d\n", use_center_vertex);
    // if(use_center_vertex){
    // printf("checking the center vertex\n");
    //on the current row we also store the values of the vertex 

    //if we have fractional coords it means we are in a fine lattice which is embeddeed in a coarser one. So the keys were multiplied by 0.5 or something like that. That means that we will not find any center vertex
    // if(use_center_vertex_from_lattice_neigbhours && has_all_coords_integer){
    if(has_all_coords_integer){
        int query_offset = hash_table_neighbours.retrieve(key_query_int); 
        if(query_offset>=0){
            float *row = im2row_in + row_size * query_offset; //this is the row where we will find our value
            int idx_within_row= row_size - val_dim; 
            float* start_of_my_values=row + idx_within_row;
            for (int i = 0; i < val_dim; i++){
                if(!do_test){
                    val_me_out[i]+=start_of_my_values[i];
                }
                if(do_test){
                    if(val_me_out[i] != start_of_my_values[i]){
                        printf("the values don't correspond. val_me_out is %f and start of my values is %f\n ", val_me_out[i], start_of_my_values[i]);
                    }
                }
            }
        }
    }
    // }else if (!use_center_vertex_from_lattice_neigbhours){
    //     float *row = im2row_in + row_size * idx; //this is the row where we will find our value
    //     int idx_within_row= row_size - val_dim; 
    //     float* start_of_my_values=row + idx_within_row;
    //     for (int i = 0; i < val_dim; i++){
    //         // val_me_out[i]+=start_of_my_values[i];
    //         if(!do_test){
    //             val_me_out[i]+=start_of_my_values[i];
    //         }
    //         if(do_test){
    //             if(val_me_out[i] != start_of_my_values[i]){
    //                 printf("the values don't correspond. val_me_out is %f and start of my values is %f\n ", val_me_out[i], start_of_my_values[i]);
    //             }
    //         }
    //     }
    // }




}


// //return the fractionary part of a float number https://devtalk.nvidia.com/default/topic/399562/frac-function-in-cuda/
// __device__ float fracf(float x){
//     return x - truncf(x);
//     // return x - floor(x);
// } 

template<int pos_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
coarsen(int capacity, HashTableGPU fine_hash_table, HashTableGPU coarse_hash_table) {

    // finer lattice has a certain lattices at integer coordinates
    // corse_lattice create_coarser_lattice(fine_lattice)
    //     for every key in the fine lattice
    //         key_div=divide by 2 the key
    //         if (the key_div that still have integer coordinates)
    //             hashmap_coarse.insert(key_div)
    //             for every neighbour in the 1 hop
    //              if neighbour exist:
    //                 n_key_div=divide the neighbouring key by 2
    //                 vec_center_neighbour = n_key_div - key_div (vector that goes from the center lattice, which is the one with the integer coordinates to the one that does not)
    //                 (WRONG)new_key= n_key_div+vec_center_neighbour //this will be an integer position which the key at the corser level
    //                 new_key= key_div+vec_center_neighbour*2 //this will be an integer position which the key at the corser level
    //                 hashmap_corse.insert(key)

    // const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a lattice vertex
    if (idx >= capacity)
        return;

    if (idx >= *fine_hash_table.m_nr_filled){
        return;
    }

    bool debug=false;
    if(idx==100){
        // debug=true;
    }
    
    //get each full (m_pos_dim+1) key of the fine_hash_table (in the hash table only the first m_pos_dim digits are stored because we know they have to sum up to 1)
    int fine_key[pos_dim + 1];
    int coord_sum=0;
    for (int i = 0; i < pos_dim; i++) {
        fine_key[i] = fine_hash_table.m_keys[idx * pos_dim + i];
        coord_sum+=fine_key[i];
    }
    fine_key[pos_dim] = -coord_sum;

    if(debug){
        printf("fine_key is %d,%d,%d,%d\n",fine_key[0],fine_key[1],fine_key[2],fine_key[3] );
    }

    //divide by 2 the finer key (so as to embedd it into the coarser one) and check if the division creates a key which has integer coordinates
    int fine_key_div[pos_dim + 1];
    bool has_integer_coords=true;
    for (int i = 0; i < pos_dim+1; i++) {
        // float digit_div_float=((float)fine_key[i])/2.0;
        float digit_div_float=((float)fine_key[i])/2.0;
        fine_key_div[i] = round(digit_div_float);
        // if(debug){
        //     printf("divit_div_float at position %d is %f\n",i, digit_div_float );
        // }

        float integer_part=0.0;
        float decimal_part=0.0;
        decimal_part=modff(digit_div_float, &integer_part);
        decimal_part=fabs(decimal_part);
        // printf("decimal part is %f \n", decimal_part);
        if( decimal_part>0.1){
            has_integer_coords=false;
        }
        // fine_key_div[i] = (int)digit_div_float;
        // fine_key_div[i] = digit_div_float;
    }

    // if(debug){
        // printf("fine_keys_div is %d,%d,%d,%d\n",fine_key_div[0],fine_key_div[1],fine_key_div[2],fine_key_div[3] );
    // }
    // printf(" fine_key is %d,%d,%d,%d  it has integer coordinates %d \n", fine_key[0],fine_key[1],fine_key[2],fine_key[3] , has_integer_coords  );
    // printf(" fine_key is %d,%d,%d,%d fine_keys_div is %d,%d,%d,%d\n", fine_key[0],fine_key[1],fine_key[2],fine_key[3] ,  fine_key_div[0],fine_key_div[1],fine_key_div[2],fine_key_div[3] );



    if(has_integer_coords){
    // if(true){
        // printf(" fine_key is %d,%d,%d,%d fine_keys_div is %d,%d,%d,%d\n", fine_key[0],fine_key[1],fine_key[2],fine_key[3] ,  fine_key_div[0],fine_key_div[1],fine_key_div[2],fine_key_div[3] );
        coarse_hash_table.insert(fine_key_div); 
        // coarse_hash_table.insert(fine_key); 


        // printf("has integer coords \n");
        // for every neighbour in the 1 hop
        int np[pos_dim + 1];
        int nm[pos_dim + 1];
        // float np_div[pos_dim + 1];
        // float nm_div[pos_dim + 1];
        // float vec_center_neighbour[pos_dim + 1];
        int nr_axes=pos_dim+1;
        for(int axis=0; axis<nr_axes; axis++){

            if(debug){
                printf("axis is %d\n",axis );
            }

            //for each axis we have 2 neighbours
            for (int i = 0; i < pos_dim+1; i++) {
                np[i] = fine_key[i] + 1.0;
                nm[i] = fine_key[i] - 1.0;
            }
            np[axis] = fine_key[axis] - 1.0*pos_dim;
            nm[axis] = fine_key[axis] + 1.0*pos_dim;

            if(debug){
                printf("np is %d,%d,%d,%d\n",np[0],np[1],np[2],np[3] );
            }

            // //divide the neighbouring key by 2 (with floating coords)
            // for (int i = 0; i < pos_dim+1; i++) {
            //     np_div[i]=((float)np[i])/2.0;
            //     nm_div[i]=((float)nm[i])/2.0;
            // }
            // if(debug){
            //     printf("np_div is %f,%f,%f,%f\n",np_div[0],np_div[1],np_div[2],np_div[3] );
            // }

            int scale_modif_debug=1.0;
            //NEIGHBOUR 1
            int offNp = fine_hash_table.retrieve(np);
            if(offNp>=0){
                // // vec_center_neighbour = n_key_div - key_div (vector that goes from the center lattice, which is the one with the integer coordinates to the one that does not)
                // for (int i = 0; i < pos_dim+1; i++) {
                //     vec_center_neighbour[i]= np_div[i] - (float)fine_key_div[i];
                // }
                // if(debug){
                //     printf("vec_center_neighbour %f,%f,%f,%f\n",vec_center_neighbour[0],vec_center_neighbour[1],vec_center_neighbour[2],vec_center_neighbour[3] );
                // }
                // //calculate the new key at the coarse lvl. new_key= n_key_div+vec_center_neighbour /
                // for (int i = 0; i < pos_dim+1; i++) {
                //     // np[i]= round(np_div[i] + vec_center_neighbour[i]); //round in case any numerical errors happened
                //     np[i]= round( fine_key_div[i] + vec_center_neighbour[i]*2 ); //round in case any numerical errors happened
                // }

                // //another way of calculatin the new keyu at the coarse level
                // for (int i = 0; i < pos_dim+1; i++) {
                //     int movement=1.0;
                //     if(i==axis){
                //         movement = -pos_dim;
                //     }
                //     np[i]= fine_key_div[i] + movement; //round in case any numerical errors happened
                // }

                //attempt 3
                for (int i = 0; i < pos_dim+1; i++) {
                    np[i] = fine_key_div[i] + scale_modif_debug*1.0;
                }
                np[axis] = fine_key_div[axis] - scale_modif_debug*1.0*pos_dim;


                //the dumbest way to check why it does weird thing is to just insert all the neighbours directl, the lattice vertices should still be lower because we only take the neighbours of the vertices that have integer coords after division
                
                if(debug){
                    printf("inserting %d,%d,%d,%d\n",np[0],np[1],np[2],np[3] );
                }
                coarse_hash_table.insert(np); 
            }


            //NEIGHBOUR 2
            int offNm = fine_hash_table.retrieve(nm);
            if(offNm>=0){
                // //same for the other neigbhour
                // for (int i = 0; i < pos_dim+1; i++) {
                //     vec_center_neighbour[i]= nm_div[i] - (float)fine_key_div[i];
                // }
                // //calculate the new key at the coarse lvl. new_key= n_key_div+vec_center_neighbour /
                // for (int i = 0; i < pos_dim+1; i++) {
                //     // nm[i]= round(nm_div[i] + vec_center_neighbour[i]); //round in case any numerical errors happened
                //     nm[i]= round( fine_key_div[i] + vec_center_neighbour[i]*2 ); //round in case any numerical errors happened
                // }

                // //another way of calculatin the new keyu at the coarse level
                // for (int i = 0; i < pos_dim+1; i++) {
                //     int movement=-1.0;
                //     if(i==axis){
                //         movement = pos_dim;
                //     }
                //     nm[i]= fine_key_div[i] + movement; //round in case any numerical errors happened
                // }

                //attempt 3
                for (int i = 0; i < pos_dim+1; i++) {
                    nm[i] = fine_key_div[i] - scale_modif_debug*1.0;
                }
                nm[axis] = fine_key_div[axis] + scale_modif_debug*1.0*pos_dim;

                // printf("inserting %d,%d,%d,%d\n",np[0],np[1],np[2],np[3] );
                coarse_hash_table.insert(nm); 
            }


        }


    }
   

}



template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice(const int n, float *values, int* splatting_indices, float* splatting_weights , HashTableGPU hash_table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;

    float value[val_dim]{0};
    float weight = 0;

    for (int i = 0; i <= pos_dim; i++) {
        // MatrixEntry r = matrix[idx * (pos_dim + 1) + i];

        int splatting_idx=splatting_indices[idx * (pos_dim + 1) + i];
        int splatting_weight=splatting_weights[idx * (pos_dim + 1) + i];

        float *val = hash_table.m_values + splatting_idx * (val_dim+1);
        for (int j = 0; j < val_dim; j++) {
            value[j] += splatting_weight * val[j];
        }
        weight += splatting_weight * val[val_dim];
    }

    weight = 1.0 / weight;
    for (int j = 0; j < val_dim; j++){
        values[idx * val_dim  + j] = value[j] * weight;
        // printf("values out %f \n", value[j] * weight);
    }
}



template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_with_precomputation(const float* positions,  float* values, const int nr_positions, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }



    //here we accumulate the values and the homogeneous term
    float val_hom[val_dim]{0};

    for (int remainder = 0; remainder <= pos_dim; remainder++) {

        int idx_val=splatting_indices[idx * (pos_dim + 1) + remainder];
        float weight= splatting_weights[idx * (pos_dim + 1) + remainder];

        float *val = const_cast<float *>(hash_table.m_values + idx_val * val_dim );

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        if(idx_val!=-1){
            for (int i = 0; i < val_dim ; i++){
                val_hom[i]+= val[i]* weight;
                // printf("val[i]  %f \n", val[i] );
                // printf("barycentric  %f \n", barycentric[remainder] );
            }
        }
    
    }


    //do not divicde by the homogeneous coordinate, rather just store the value as it is because we will afterwards need the homogeneous coordinate for the backwards passs
    for (int i = 0; i < val_dim; i++){
            values[idx*val_dim + i]= val_hom[i] ;
    }



}


template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_no_precomputation(const float* positions,  float* values, const int nr_positions, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }


    
    float elevated[pos_dim + 1];
    const float *position = positions + idx * pos_dim;
    elevate<pos_dim>(elevated, position);
    int rem0[pos_dim + 1];
    int rank[pos_dim + 1];


    // //TODO the scale factor can be precomputed
    // float scaleFactor[pos_dim];
    // float invStdDev = (pos_dim + 1) * sqrt(2.0f / 3);
    // for (int i = 0; i < pos_dim; i++) {
    //     scaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
    // }

    // // embed position vector into the hyperplane
    // // first rotate position into the (pd+1)-dimensional hyperplane
    // // sm contains the sum of 1..n of our feature vector
    // float sm = 0;
    // for (int i = pos_dim; i > 0; i--) {
    //     float cf = position[i - 1] * scaleFactor[i - 1];
    //     // float cf = position[i - 1] ;
    //     elevated[i] = sm - i * cf;
    //     sm += cf;
    // }
    // elevated[0] = sm;


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pos_dim; i++)
        rank[i] = 0;
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //here we accumulate the values and the homogeneous term
    float val_hom[val_dim]{0};

    int key[pos_dim];
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        int idx_val=hash_table.retrieve(key);
        float *val = const_cast<float *>(hash_table.m_values + idx_val * val_dim );
        // printf("idx_val  %d \n", idx_val );

        //store also the splatting indices and weight so that they can be used for the backwards pass
        if(idx_val>=0){
            splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
            splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
        }else{
            // printf("Slicing around a lattice vertex that is not yet created at positions idx %d \n", idx);
        }

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        if(idx_val!=-1){
            for (int i = 0; i < val_dim ; i++){
                val_hom[i]+= val[i]* barycentric[remainder];
                // printf("val[i]  %f \n", val[i] );
                // printf("barycentric  %f \n", barycentric[remainder] );
            }
        }
    
    }

    //divide by the homogeneous coord but only if the val_full_dim>1 because if it's 1 then the val_dim is 0 so we are left with nothing
    // if(do_normalization){
    // for (int i = 0; i < val_dim; i++){
    //     float weight=val_hom[val_dim];
    //     if(weight!=0.0){ //to avoid divisionz by 0
    //         values[idx*val_dim + i]= val_hom[i] / weight;
    //     }else{ //the weight is 0 which means we landed in a simplex that is not allocated. The value will just be 0 then
    //         values[idx*val_dim + i]= 0.0;
    //     }
    // }


    //do not divicde by the homogeneous coordinate, rather just store the value as it is because we will afterwards need the homogeneous coordinate for the backwards passs
    for (int i = 0; i < val_dim; i++){
            values[idx*val_dim + i]= val_hom[i] ;
    }



}


template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
gather_no_precomputation(const float* positions,  float* gathered_values, const int nr_positions, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }

    bool debug=false;
    if (idx==0){
        debug=true;
    }


    
    float elevated[pos_dim + 1];
    const float *position = positions + idx * pos_dim;
    elevate<pos_dim>(elevated, position);
    int rem0[pos_dim + 1];
    int rank[pos_dim + 1];


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pos_dim; i++)
        rank[i] = 0;
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //here we accumulate the values and the homogeneous term
    // float ga[val_full_dim]{0};
    int row_size_gathered=(pos_dim+1)*(val_dim+1);
    float* gathered_row = gathered_values + idx * row_size_gathered;

    int key[pos_dim];
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        int idx_val=hash_table.retrieve(key);
        float *val = const_cast<float *>(hash_table.m_values + idx_val * val_dim );
        // printf("idx_val  %d \n", idx_val );

        //store also the splatting indices and weight so that they can be used for the backwards pass
        if(idx_val>=0 && barycentric[remainder]>0.00001){ //we ignore the vertices that are too far awa when we slice a vertex that is on top of another one we will get some vertices with barycentric coordines zero and which one we get is arbitrary
            splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
            splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
        }else{
            // printf("Slicing around a lattice vertex that is not yet created at positions idx %d \n", idx);
        }

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        if(idx_val!=-1 && barycentric[remainder]>0.00001){
            int idx_in_row=remainder*( val_dim + 1 );
            for (int i = 0; i < val_dim ; i++){
                if(debug){
                    // printf("copying val %d into gathered row at positions %d \n", i, idx_in_row +i);
                }
                // gathered_row[idx_in_row + i] = val[i];
                gathered_row[idx_in_row + i] = val[i]*barycentric[remainder];
            }
            if(debug){
                // printf("copying barycentric into gathered row at positions %d \n", idx_in_row+val_full_dim);
            }
            gathered_row[idx_in_row+val_full_dim]=barycentric[remainder];
        }
    
    }



}



template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
gather_with_precomputation(const float* positions,  float* gathered_values, const int nr_positions, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }


    //here we accumulate the values and the homogeneous term
    // float ga[val_full_dim]{0};
    int row_size_gathered=(pos_dim+1)*(val_dim+1);
    float* gathered_row = gathered_values + idx * row_size_gathered;

    //int key[pos_dim];
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + remainder];
        if(splatting_idx>=0){
            float weight = splatting_weights[ idx * (pos_dim + 1) + remainder];
            float *val = const_cast<float *>(hash_table.m_values + splatting_idx * val_dim );

            //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
            int idx_in_row=remainder*( val_dim + 1 );
            for (int i = 0; i < val_dim ; i++){
                gathered_row[idx_in_row + i] = val[i]*weight;
            }
            gathered_row[idx_in_row+val_dim]=weight;

        }



       


    }



}

template<int pos_dim, int val_full_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
gather_elevated_no_precomputation(const int* keys,  float* gathered_values, const int nr_vertices, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table_to_gather_from, const int lattice_to_gather_from_lvl, const int elevated_verts_lvl) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_vertices){ //don't go out of bounds
        return;
    }

    float elevated[pos_dim + 1];
    const int *key_elevated_vert = keys + idx * pos_dim;
    //get the elevated key which is just the full m_pos_dim+1 key
    int key_sum=0;
    for (int i = 0; i < pos_dim; i++) {
        elevated[i]=key_elevated_vert[i];
        key_sum+=key_elevated_vert[i];
    }
    elevated[pos_dim] = -key_sum;

    //in case the elevated verts and the hash table to slice from are at different lattice levels, we would need to scale them
    int lvl_diff=elevated_verts_lvl-lattice_to_gather_from_lvl; 
    float scale=pow(2.0f, (float)lvl_diff); 
    // printf("scale is %f \n", scale);
    for (int i = 0; i < pos_dim+1; i++) {
        elevated[i] = elevated[i]*scale; 
    }


    // elevate<pos_dim>(elevated, position);
    int rem0[pos_dim + 1];
    int rank[pos_dim + 1];

    


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pos_dim; i++)
        rank[i] = 0;
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //here we accumulate the values and the homogeneous term
    // float ga[val_full_dim]{0};
    int row_size_gathered=(pos_dim+1)*(val_full_dim+1);
    float* gathered_row = gathered_values + idx * row_size_gathered;

    int key[pos_dim];
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        int idx_val=hash_table_to_gather_from.retrieve(key);
        float *val = const_cast<float *>(hash_table_to_gather_from.m_values + idx_val * val_full_dim );
        // printf("idx_val  %d \n", idx_val );

        //store also the splatting indices and weight so that they can be used for the backwards pass
        if(idx_val>=0 && barycentric[remainder]>0.00001){ //we ignore the vertices that are too far awa when we slice a vertex that is on top of another one we will get some vertices with barycentric coordines zero and which one we get is arbitrary
            splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
            splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
        }else{
            // printf("Slicing around a lattice vertex that is not yet created at positions idx %d \n", idx);
        }

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        if(idx_val!=-1 && barycentric[remainder]>0.00001){
            int idx_in_row=remainder*( val_full_dim + 1 );
            for (int i = 0; i < val_full_dim ; i++){
                // gathered_row[idx_in_row + i] = val[i];
                gathered_row[idx_in_row + i] = val[i]*barycentric[remainder];
            }
            gathered_row[idx_in_row+val_full_dim]=barycentric[remainder];
        }
    
    }



}

template<int pos_dim, int val_full_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_elevated_verts(float* values, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table_to_slice_from, HashTableGPU hash_table_elevated_verts, const int lattice_to_slice_from_lvl, const int elevated_verts_lvl) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if (idx >= *hash_table_elevated_verts.m_nr_filled){
        return;
    }


    
    float elevated[pos_dim + 1];
    const int *key_elevated_vert = hash_table_elevated_verts.m_keys + idx * pos_dim;
    //get the elevated key which is just the full m_pos_dim+1 key
    int key_sum=0;
    for (int i = 0; i < pos_dim; i++) {
        elevated[i]=key_elevated_vert[i];
        key_sum+=key_elevated_vert[i];
    }
    elevated[pos_dim] = -key_sum;

    //in case the elevated verts and the hash table to slice from are at different lattice levels, we would need to scale them
    int lvl_diff=elevated_verts_lvl-lattice_to_slice_from_lvl; 
    float scale=pow(2.0f, (float)lvl_diff); 
    // printf("scale is %f \n", scale);
    for (int i = 0; i < pos_dim+1; i++) {
        elevated[i] = elevated[i]*scale; 
    }


    // elevate<pos_dim>(elevated, position);
    int rem0[pos_dim + 1];
    int rank[pos_dim + 1];



    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pos_dim; i++)
        rank[i] = 0;
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //here we accumulate the values and the homogeneous term
    float val_hom[val_full_dim]{0};

    int key[pos_dim];
    int nr_vertices_allocated=0;
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        int idx_val=hash_table_to_slice_from.retrieve(key);
        float *val = const_cast<float *>(hash_table_to_slice_from.m_values + idx_val * val_full_dim );
        // printf("idx_val  %d \n", idx_val );

        //store also the splatting indices and weight so that they can be used for the backwards pass
        if(idx_val>=0 && barycentric[remainder]>0.00001){ //we ignore the vertices that are too far awa when we slice a vertex that is on top of another one we will get some vertices with barycentric coordines zero and which one we get is arbitrary
            splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
            splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
            nr_vertices_allocated++;
        }else{
            // printf("Slicing around a lattice vertex that is not yet created at positions idx %d \n", idx);
        }

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        if(idx_val!=-1 && barycentric[remainder]>0.00001){
            for (int i = 0; i < val_full_dim ; i++){
                val_hom[i]+= val[i]* barycentric[remainder];
                // printf("val[i]  %f \n", val[i] );
                // printf("barycentric  %f \n", barycentric[remainder] );
            }
        }
    
    }

    if(nr_vertices_allocated==0){
        printf("Slicing around a positions which has no vertices around it at idx %d \n", idx);
    }

    //divide by the homogeneous coord but only if the val_full_dim>1 because if it's 1 then the val_dim is 0 so we are left with nothing
    // if(do_normalization){
    // for (int i = 0; i < val_dim; i++){
    //     float weight=val_hom[val_dim];
    //     if(weight!=0.0){ //to avoid divisionz by 0
    //         values[idx*val_dim + i]= val_hom[i] / weight;
    //     }else{ //the weight is 0 which means we landed in a simplex that is not allocated. The value will just be 0 then
    //         values[idx*val_dim + i]= 0.0;
    //     }
    // }


    //do not divicde by the homogeneous coordinate, rather just store the value as it is because we will afterwards need the homogeneous coordinate for the backwards passs
    for (int i = 0; i < val_full_dim; i++){
        values[idx*val_full_dim + i]= val_hom[i] ;
    }



}


template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_classify_no_precomputation(const float* positions,  float* class_logits, const float* delta_weights, const float* linear_clasify_weight, const float* linear_clasify_bias, const int nr_classes, const int nr_positions, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }


    
    float elevated[pos_dim + 1];
    const float *position = positions + idx * pos_dim;
    elevate<pos_dim>(elevated, position);
    int rem0[pos_dim + 1];
    int rank[pos_dim + 1];


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pos_dim; i++)
        rank[i] = 0;
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //here we accumulate the values and the homogeneous term
    float val_hom[val_dim]{0};

    const float* delta_weights_row=delta_weights+idx*(pos_dim+1); //delta_weights has shape nr_positions x (pos_dim+1)

    int key[pos_dim];
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        int idx_val=hash_table.retrieve(key);
        float *val = const_cast<float *>(hash_table.m_values + idx_val * val_dim );
        // printf("idx_val  %d \n", idx_val );

        //store also the splatting indices and weight so that they can be used for the backwards pass
        if(idx_val>=0){
            splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
            splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
        }else{
            // printf("Slicing around a lattice vertex that is not yet created at positions idx %d \n", idx);
        }

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        if(idx_val!=-1){
            for (int i = 0; i < val_dim ; i++){
                val_hom[i]+= val[i]* (barycentric[remainder]+delta_weights_row[remainder]);
                // val_hom[i]+= val[i]* (delta_weights_row[remainder]);
                // printf("val[i]  %f \n", val[i] );
                // printf("barycentric  %f \n", barycentric[remainder] );
                // printf("delta weight is   %f \n", delta_weights_row[remainder] );
            }
        }
    
    }


    //now the value need to pass through a linear layer
    float* logits_out_for_cur_position=class_logits+idx*nr_classes;//class_logits has shape nr_positions x nr_classes

    for (int c = 0; c < nr_classes; c++) {
        const float* weight_for_class= linear_clasify_weight+ c*val_dim;
        for (int val_idx = 0; val_idx < val_dim; val_idx++) {
            //WARNING linear clasify weight has shape nr_classes x val_Full_dim. So in the tranposed way that we would expect if it was just a mtrix multiply
            // if (c==0 && val_idx==0){
                // printf("logits_out_for_cur_position  %f \n", logits_out_for_cur_position[0] );
            // }
            logits_out_for_cur_position[c]+=weight_for_class[val_idx]*val_hom[val_idx];
        }
        logits_out_for_cur_position[c]+=linear_clasify_bias[c];
    }

  

}




template<int pos_dim, int val_dim, int nr_classes>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_classify_with_precomputation(const float* positions,  float* class_logits, const float* delta_weights, const float* linear_clasify_weight, const float* linear_clasify_bias, const int nr_positions, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }

    // return;





    //here we accumulate the values and the homogeneous term
    float val_hom[val_dim]{0};

    const float* delta_weights_row=delta_weights+idx*(pos_dim+1); //delta_weights has shape nr_positions x (pos_dim+1)

    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + remainder];
        if(splatting_idx>=0){
            float weight = splatting_weights[ idx * (pos_dim + 1) + remainder];
            float *val = const_cast<float *>(hash_table.m_values + splatting_idx * val_dim );

            //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
            for (int i = 0; i < val_dim ; i++){
                val_hom[i]+= val[i]* (weight+delta_weights_row[remainder]);
            }

        }

  
    
    }

    // return;

    //now the value need to pass through a linear layer
    float* logits_out_for_cur_position=class_logits+idx*nr_classes;//class_logits has shape nr_positions x nr_classes


    //load the weights of the linear layers into shared mem 
    __shared__ float linear_weights_shared[nr_classes*val_dim];
    if (threadIdx.x == 0 ){
        for (int i = 0; i < nr_classes*val_dim; i++) {
            linear_weights_shared[i]=linear_clasify_weight[i];
        }
    }
    __syncthreads();


    for (int c = 0; c < nr_classes; c++) {
        // const float* weight_for_class= linear_clasify_weight+ c*val_full_dim;
        const float* weight_for_class= linear_weights_shared+ c*val_dim;
        for (int val_idx = 0; val_idx < val_dim; val_idx++) {
            //WARNING linear clasify weight has shape nr_classes x val_Full_dim. So in the tranposed way that we would expect if it was just a mtrix multiply
            // if (c==0 && val_idx==0){
                // printf("logits_out_for_cur_position  %f \n", logits_out_for_cur_position[0] );
            // }
            logits_out_for_cur_position[c]+=weight_for_class[val_idx]*val_hom[val_idx];
        }
        logits_out_for_cur_position[c]+=linear_clasify_bias[c];
    }

  

}


template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_backwards_with_precomputation(const int nr_positions, float* sliced_values_hom, float* grad_sliced_values, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim



    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    //each positions will splat onto pos_dim+1 vertices
    float *grad_sliced_val = grad_sliced_values + idx * val_dim;
    float *sliced_value_hom = sliced_values_hom + idx * ( val_dim +1);
    float one_div_valHom = 1.0/sliced_value_hom[val_dim];
    float one_div_valHom_2 = 1.0/(sliced_value_hom[val_dim] * sliced_value_hom[val_dim]);
    //compute the gradient of the homogeneous coordinate (check the leather notebook for more info)
    float grad_hom_val=0.0;
    for (int j = 0; j < val_dim; j++) {
        grad_hom_val+=grad_sliced_val[j]*sliced_value_hom[j];
    }
    grad_hom_val*=one_div_valHom_2;


    for(int color=0; color<pos_dim+1; color++){
        // int index_into_m_entries=round(splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0]);
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];
        if(splatting_idx>=0){

            // float weight = splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 1];
            float weight = splatting_weights[ idx * (pos_dim + 1) + color];
            float *valOut = hash_table.m_values + splatting_idx * (val_dim+1);
            // float *val_forward_vertex = values_forward_vertices + splatting_idx * (val_dim+1);

            //get v3h which is the sliced value in homogeneous coordinates


            //acumulate the values
            // float sum_values_homogeneous=0;
            for (int j = 0; j < val_dim; j++) {
                float grad_local=weight/sliced_value_hom[val_dim]; //the gradient of the sliced value (and normalized) wrt to the lattice vertex 
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
                    // #warning CUDA ARCH IS FINE
                    atomicAdd(valOut +j, grad_sliced_val[j] * grad_local);
                    // atomicAdd(valOut +j, grad_sliced_val[j]*weight);
                #else 
                    #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
                #endif
            }

            // //homogeneous coordinate grad
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
                // #warning CUDA ARCH IS FINE
                atomicAdd(valOut +val_dim, -weight * grad_hom_val);
            #else 
                #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
            #endif
        
        }

    }




}



template<int pos_dim, int val_full_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_backwards_with_precomputation_no_homogeneous(const int nr_positions, float* grad_sliced_values, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {

    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim



    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    //each positions will splat onto pos_dim+1 vertices
    float *grad_sliced_val = grad_sliced_values + idx * val_full_dim;
    // printf("grad_sliced_val_is %f val_full_dim is %d\n",*grad_sliced_val, val_full_dim);

    //we will need to splat this gradient to 4 vertices, instead of reading this gradient 4 times from memory, we just read it once in shared mem and then read it from there 
    //we load BLOCK_SIZE vertices so a matrix of rows=BLOCK_SIZE and cols=val_dim
    // __shared__ float grad_sliced_val_shared[BLOCK_SIZE*val_full_dim];
    // int block_nr=blockIdx.x;
    // if (threadIdx.x == 0 ){
    //     for (int r = 0; r < BLOCK_SIZE; r++) {
    //         for (int c = 0; c < val_full_dim; c++) {
    //             int row_to_copy_from=max(  r+block_nr*BLOCK_SIZE, nr_positions-1);
    //             grad_sliced_val_shared[c+r*val_full_dim]= grad_sliced_values[ c+  row_to_copy_from*val_full_dim ];
    //         }
    //     }
    // }
    // __syncthreads();

    //attempt 2 load into local memory which could be global but the optimizer might be smart and put it into register 
    float grad_sliced_val_cur_pos[val_full_dim];
    for (int j = 0; j < val_full_dim; j++) {
        grad_sliced_val_cur_pos[j]=grad_sliced_val[j];
    }

    
    for(int color=0; color<pos_dim+1; color++){
        // int index_into_m_entries=round(splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0]);
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];
        if(splatting_idx>=0){

            // float weight = splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 1];
            float weight = splatting_weights[ idx * (pos_dim + 1) + color];
            float *valOut = hash_table.m_values + splatting_idx * val_full_dim;
            // float *val_forward_vertex = values_forward_vertices + splatting_idx * (val_dim+1);

            //get v3h which is the sliced value in homogeneous coordinates


            //acumulate the values
            // float sum_values_homogeneous=0;
            for (int j = 0; j < val_full_dim; j++) {
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
                    // printf("accumulating weighted gradient %f where gradient is %f and weight is %f  \n", grad_sliced_val[j]*weight, grad_sliced_val[j], weight );
                    // #warning CUDA ARCH IS FINE
                    // atomicAdd(valOut +j, grad_sliced_val[j]*weight);
                    atomicAdd(valOut +j, grad_sliced_val_cur_pos[j]*weight);
                    // atomicAdd(valOut +j, grad_sliced_val_shared[j+threadIdx.x*val_full_dim]*weight);
                #else 
                    #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
                #endif
            }

        
        }else{
            // printf("splatting idx is not valid for idx %d\n", idx );
        }


    }









}




template<int pos_dim, int val_dim, int nr_classes>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_classify_backwards_with_precomputation(const int nr_positions, float* grad_class_logits, float* initial_values, int* splatting_indices, float* splatting_weights, float* delta_weights, float* linear_clasify_weight, float* linear_clasify_bias, float* grad_lattice_values, float* grad_delta_weights, float* grad_linear_clasify_weight, float* grad_linear_clasify_bias,  HashTableGPU hash_table) {


    //initial_Values refers to the values that the lattice had in the forward pass. it has size nr_vertices x val_full_dim
    //grad_class_logits is the gradient of the loss with respect to the sliced out values which has size nr_positions x nr_classes

    // printf("wtf \n");

    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }



    //load the weights of the linear layers into shared mem 
    // printf("trying to allocate shared memory of size %d \n", nr_classes*val_full_dim);
    __shared__ float linear_weights_shared[nr_classes*val_dim];
    if (threadIdx.x == 0 ){
        for (int i = 0; i < nr_classes*val_dim; i++) {
            linear_weights_shared[i]=linear_clasify_weight[i];
        }
    }
    __syncthreads();


    //each positions will splat onto pos_dim+1 vertices
    float *grad_class_logits_cur_position = grad_class_logits + idx * nr_classes;


    //GRAD LATTICE VERTICES (need to so some atomic add over all the vertices from whic we sliced)
    for(int color=0; color<pos_dim+1; color++){
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];
        if(splatting_idx>=0){

            float splat_weight = splatting_weights[ idx * (pos_dim + 1) + color];
            float splat_delta_weight=delta_weights[ idx * (pos_dim + 1) + color];
            float *grad_lattice_vertex_out = grad_lattice_values + splatting_idx * val_dim;


            for (int v = 0; v < val_dim; v++) {
                float grad=0.0;
                for (int c = 0; c < nr_classes; c++) {
                    // //debug
                    // if( linear_clasify_weight[v+c*val_full_dim]!=linear_weights_shared[v+c*val_full_dim] ){
                    //     printf("The global value is not the same as the shared one, global is %f and shared is %f \n",linear_clasify_weight[v+c*val_full_dim],linear_weights_shared[v+c*val_full_dim]  );
                    // }
                    grad+=grad_class_logits_cur_position[c]*linear_weights_shared[v+c*val_dim]*(splat_weight+splat_delta_weight);
                }
                atomicAdd(grad_lattice_vertex_out+v, grad);
            }
        
        }

    }


    //GRAD LINEAR CLASIFY WEIGHT (need to do some attomic add over all the clasify weights which are shape nr_classes x val_full_dim )
    //we need the sliced value ( sliced using as weight both the splatting weight and the delta weight)
    float sliced_value[val_dim]{0};
    for(int color=0; color<pos_dim+1; color++){
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];
        if(splatting_idx>=0){

            float splat_weight = splatting_weights[ idx * (pos_dim + 1) + color];
            float splat_delta_weight=delta_weights[ idx * (pos_dim + 1) + color];
            float* vertex_value=initial_values+splatting_idx*val_dim;

            for (int v = 0; v < val_dim; v++) {
                sliced_value[v]+=vertex_value[v]*(splat_weight + splat_delta_weight);
                // sliced_value[v]+=vertex_value[v]*(splat_delta_weight);
            }
        
        }
    }
    //Now we accumulate the gradient into the LINEAR CLASIFY WEIGHT 
    for (int c = 0; c < nr_classes; c++) {
        for (int v = 0; v < val_dim; v++) {
            atomicAdd(grad_linear_clasify_weight+v +c*val_dim, sliced_value[v]*grad_class_logits_cur_position[c]);
        }
    }


    //GRAD LINEAR CLASIFY BIAS (need to do some atomic add over the nr_classes biases)
    for (int c = 0; c < nr_classes; c++) {
        atomicAdd(grad_linear_clasify_bias + c , grad_class_logits_cur_position[c]);
    }



    //GRAD DELTA WEIGHTS (need to do some atomic add over the pos_dim+1 delta weights that we have for this positions)
    for (int color = 0; color < pos_dim+1; color++) {
        float grad=0.0;
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];
        if(splatting_idx>=0){
            float* vertex_value=initial_values+splatting_idx*val_dim;
            for (int c = 0; c < nr_classes; c++) {
                float grad_output_wrt_dw=0.0;
                //need to calculate the gradient of the output o1 with respect to the delta weight
                for (int v = 0; v < val_dim; v++) {
                    // //debug
                    // if( linear_clasify_weight[v+c*val_full_dim]!=linear_weights_shared[v+c*val_full_dim] ){
                    //     printf("The global value is not the same as the shared one, global is %f and shared is %f ",linear_clasify_weight[v+c*val_full_dim],linear_weights_shared[v+c*val_full_dim]  );
                    // }
                    grad_output_wrt_dw+=vertex_value[v]*linear_weights_shared[v +c*val_dim];
                }
                grad+=grad_output_wrt_dw*grad_class_logits_cur_position[c];
            }
        }
        atomicAdd(grad_delta_weights + color + idx*(pos_dim+1) , grad);
    }




}




template<int pos_dim, int val_full_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
gather_backwards_with_precomputation(const int nr_positions, float* grad_sliced_values, int* splatting_indices, float* splatting_weights,  HashTableGPU hash_table) {


    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    // bool debug=false;
    // if(idx==0){
    //     debug=true;
    // }

    //each positions will splat onto pos_dim+1 vertices
    int row_size_grad_sliced= (pos_dim+1)*(val_full_dim+1);
    float *grad_sliced_val = grad_sliced_values + idx * row_size_grad_sliced;
    
    for(int color=0; color<pos_dim+1; color++){
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];
        if(splatting_idx>=0){

            float weight = splatting_weights[ idx * (pos_dim + 1) + color];
            float *valOut = hash_table.m_values + splatting_idx * val_full_dim;


            //scatter the values
            int idx_in_row=color*(val_full_dim+1);
            for (int j = 0; j < val_full_dim; j++) {
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
                    atomicAdd(valOut +j, grad_sliced_val[j + idx_in_row]*weight );
                    // atomicAdd(valOut +j, grad_sliced_val[j + idx_in_row] );
                #else 
                    #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
                #endif
            }

        
        }else{
            // printf("splatting idx is not valid for idx %d\n", idx );
        }


    }


}


#endif

