#include "lattice_net/Lattice.cuh"

//c++
#include <string>

#include "EasyPytorch/UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it
#include "EasyCuda/UtilsCuda.h"
#include "string_utils.h"

//my stuff
#include "lattice_net/HashTable.cuh"
// #include "surfel_renderer/lattice/kernels/HashTableGPU.cuh"
#include "lattice_net/kernels/LatticeGPU.cuh"

//jitify
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LAUNCH 1

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;
//Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
#define ENABLE_CUDA_PROFILING 1
#include "Profiler.h" 

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

//jitify
using jitify::reflection::type_of;

// #define BLOCK_SIZE 128 //TODO no actually need for it. It can be a parameter. And the one kernel that needs to read this inside it's code can just use BLOCKdim.x
// #define BLOCK_SIZE 64 //TODO no actually need for it. It can be a parameter. And the one kernel that needs to read this inside it's code can just use BLOCKdim.x


using torch::Tensor;
using namespace easy_pbr::utils;






//CPU code that calls the kernels
Lattice::Lattice(const std::string config_file):
    m_impl( new LatticeGPU() ),
    m_hash_table( new HashTable() ),
    m_pos_dim(-1),
    m_val_dim(-1),
    m_val_full_dim(-1),
    m_lvl(1)
    {

    init_params(config_file);
    VLOG(3) << "Creating lattice";

}

Lattice::Lattice(const std::string config_file, const std::string name):
    m_impl( new LatticeGPU() ),
    m_hash_table( new HashTable() ),
    m_pos_dim(-1),
    m_val_dim(-1),
    m_val_full_dim(-1),
    m_name(name)
    {

    init_params(config_file);

    // //random states
    // m_nr_states=1000000;
    // cudaMalloc ( &m_devStates, m_nr_states*sizeof( curandState ) );
    // // setup seeds
    // m_impl->setup_seeds(m_devStates, m_nr_states);

    VLOG(3) << "Creating lattice: " <<name;

}

Lattice::Lattice(Lattice* other):
    m_impl( new LatticeGPU() ),
    m_hash_table( new HashTable() ),
    m_pos_dim(-1),
    m_val_dim(-1),
    m_val_full_dim(-1),
    m_lvl(1)
    {
        m_lvl=other->m_lvl;
        m_pos_dim=other->m_pos_dim;
        m_val_dim=other->m_val_dim;
        m_val_full_dim=other->m_val_full_dim;
        m_hash_table_capacity=other->m_hash_table_capacity;
        m_sigmas=other->m_sigmas;
        m_sigmas_tensor=other->m_sigmas_tensor.clone();
        m_splatting_indices_tensor=other->m_splatting_indices_tensor;
        m_splatting_weights_tensor=other->m_splatting_weights_tensor;
        m_lattice_rowified=other->m_lattice_rowified;
        m_positions=other->m_positions;
        //hashtable
        m_hash_table->m_capacity=other->m_hash_table_capacity;
        m_hash_table->m_pos_dim=other->m_pos_dim;
        // m_hash_table->m_val_dim=other->m_val_dim;
        //hashtable tensors shallow copy (just a pointer assignemtn so they use the same data in memory)
        m_hash_table->m_keys_tensor=other->m_hash_table->m_keys_tensor;
        m_hash_table->m_values_tensor=other->m_hash_table->m_values_tensor;
        m_hash_table->m_entries_tensor=other->m_hash_table->m_entries_tensor;
        m_hash_table->m_nr_filled_tensor=other->m_hash_table->m_nr_filled_tensor.clone();
        m_hash_table->update_impl();
        // m_devStates=other->m_devStates;

}

Lattice::~Lattice(){
    // LOG(WARNING) << "Deleting lattice: " << m_name;
}

void Lattice::init_params(const std::string config_file){
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config lattice_config=cfg["lattice_gpu"];
    m_hash_table_capacity = lattice_config["hash_table_capacity"];

    int nr_sigmas=lattice_config["nr_sigmas"]; //nr of is sigma values we have. Each one affecting a different number of dimensions of the positions
    for (int i=0; i < nr_sigmas; i++) {
        std::string param_name="sigma_"+std::to_string(i);
        std::string sigma_val_and_extent = (std::string)lattice_config[param_name];
        std::vector<std::string> tokenized = split(sigma_val_and_extent, " ");
        CHECK(tokenized.size()==2) << "For each sigma we must define its value and the extent(nr of dimensions it affects) in space separated string. So the nr of tokens split string should have would be 1. However the nr of tokens we have is" << tokenized.size();
        std::pair<float, int> sigma_params = std::make_pair<float,int> (  std::stof(tokenized[0]), std::stof(tokenized[1]) );
        m_sigmas_val_and_extent.push_back(sigma_params);
    }
    set_sigmas(m_sigmas_val_and_extent);


}

void Lattice::set_sigmas(std::initializer_list<  std::pair<float, int> > sigmas_list){
    m_sigmas.clear();
    for(auto sigma_pair : sigmas_list){
        float sigma=sigma_pair.first; //value of the sigma
        int nr_dim=sigma_pair.second; //how many dimensions are affected by this sigma
        for(int i=0; i < nr_dim; i++){
            m_sigmas.push_back(sigma);
        }
    }

    m_sigmas_tensor=vec2tensor(m_sigmas);
}

void Lattice::set_sigmas(std::vector<  std::pair<float, int> > sigmas_list){
    m_sigmas.clear();
    for(auto sigma_pair : sigmas_list){
        float sigma=sigma_pair.first; //value of the sigma
        int nr_dim=sigma_pair.second; //how many dimensions are affected by this sigma
        for(int i=0; i < nr_dim; i++){
            m_sigmas.push_back(sigma);
        }
    }

    m_sigmas_tensor=vec2tensor(m_sigmas);
}

void Lattice::set_and_check_input(torch::Tensor& positions_raw, torch::Tensor& values){
    //check input
    CHECK(positions_raw.size(1)==values.size(1)) << "Sizes of positions and values should match. Meaning that that there should be a value for each position";
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    CHECK(values.scalar_type()==at::kFloat) << "values should be of type float";
    CHECK(values.dim()==3) << "values should have dim 3 correspondin to NHW. However it has sizes" << values.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    m_val_dim=values.size(2);
    m_val_full_dim=m_val_dim+1;
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
}



void Lattice::begin_splat(){
    m_hash_table->clear(); 

    if(m_tmp_blurred_values_tensor.defined()){
        m_tmp_blurred_values_tensor.fill_(0);
    }
    // if(m_sliced_values_hom_tensor.defined()){
    //     m_sliced_values_hom_tensor.fill_(0);
    // }
    // if(m_lattice_rowified.defined()){
    //     m_lattice_rowified.fill_(0);
    // }
    // if(m_distributed_tensor.defined()){
    //     m_distributed_tensor.fill_(0);
    // }
    // if(m_splatting_indices_tensor.defined()){
    //     m_splatting_indices_tensor.fill_(-1);
    // }
    // if(m_splatting_weights_tensor.defined()){
    //     m_splatting_weights_tensor.fill_(-1);
    // }

    m_impl->begin_splat(); //can be commented out ,it just calls a kernel function for debugging

    // dim3 grid(1);
    // dim3 block(1);
    // m_test_jitify_program.kernel("kernel_hello")
    //             .instantiate()
    //             .configure(grid, block)
    //             .launch();
}

// //is we use the previous begin splat, it will clear everything including the keys, nr fileld and everything else. However is we do this during the backwards pass of a slice then we end up with malformed lattices which have no filled vertes
// void Lattice::begin_splat_modify_only_values(){
//     VLOG(1) << "begin splat modify only values. Filling up with zeros values tensor which has shape" << m_hash_table->m_values_tensor.sizes();
//     m_hash_table->m_values_tensor.fill_(0);
// }



void Lattice::splat_standalone(torch::Tensor& positions_raw, torch::Tensor& values, const bool with_homogeneous_coord){
    set_and_check_input(positions_raw, values);
    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);
    m_val_dim=values.size(2);
    m_val_full_dim=m_val_dim+1;


    //if it's not initialized to the correct values we intialize the hashtable
    if( !m_hash_table->m_keys_tensor.defined() ){
        // m_first_time=false;
        // m_hash_table= HashTable(m_hash_table_capacity, m_pos_dim, m_val_dim);
        m_hash_table->init(m_hash_table_capacity, m_pos_dim, m_val_dim);
        m_hash_table->to(torch::kCUDA);
       
        // m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }).to(torch::kInt32);
        // m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) });
        // m_splatting_indices_tensor=m_splatting_indices_tensor.to("cuda");
        // m_splatting_weights_tensor=m_splatting_weights_tensor.to("cuda");
    }else{
        //allocate a vector of matrices big enough for the current nr of positions
        // cudaFree(m_matrix);
        // cudaMalloc((void**)&m_matrix, sizeof(MatrixEntry)*nr_positions* (m_pos_dim+1) );
    }

    if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
        m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    // m_splatting_indices_and_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1), 2 });
    m_splatting_indices_tensor.fill_(-1);
    m_splatting_weights_tensor.fill_(-1);


    //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    values=values.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    // m_new_values_tensor=m_new_values_tensor.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma");

    TIME_START("splat");
    m_impl->splat_standalone(positions.data<float>(), values.data<float>(), nr_positions, m_pos_dim, m_val_dim, m_val_full_dim,
                            m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), with_homogeneous_coord, *(m_hash_table->m_impl) );

    // m_impl->splat_standalone(positions.data<float>(), values.data<float>(), nr_positions, m_pos_dim, m_val_dim, 
                            // m_splatting_indices_and_weights_tensor.data<float>(), *(m_hash_table.m_impl) );
    TIME_END("splat");

    VLOG(3) << "after splatting nr_verts is " << nr_lattice_vertices();
  
}


void Lattice::just_create_verts(torch::Tensor& positions_raw, const bool with_homogeneous_coord){
    // set_and_check_input(positions_raw, values);
    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);

    //if it's not initialized to the correct values we intialize the hashtable
    if( !m_hash_table->m_keys_tensor.defined() ){
        // m_first_time=false;
        // m_hash_table= HashTable(m_hash_table_capacity, m_pos_dim, m_val_dim);
        m_hash_table->init(m_hash_table_capacity, m_pos_dim, m_val_dim);
        m_hash_table->to(torch::kCUDA);
       
        // m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }).to(torch::kInt32);
        // m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) });
        // m_splatting_indices_tensor=m_splatting_indices_tensor.to("cuda");
        // m_splatting_weights_tensor=m_splatting_weights_tensor.to("cuda");
    }else{
        //allocate a vector of matrices big enough for the current nr of positions
        // cudaFree(m_matrix);
        // cudaMalloc((void**)&m_matrix, sizeof(MatrixEntry)*nr_positions* (m_pos_dim+1) );
    }

    if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
        m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    // m_splatting_indices_and_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1), 2 });
    // m_splatting_indices_tensor.fill_(-1);
    // m_splatting_weights_tensor.fill_(-1);


    //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    // m_new_values_tensor=m_new_values_tensor.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma");

    TIME_START("just_create_verts");
    m_impl->just_create_verts(positions.data<float>(), nr_positions, m_pos_dim, m_val_dim, m_val_full_dim,
                            m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), with_homogeneous_coord, *(m_hash_table->m_impl) );

    // m_impl->splat_standalone(positions.data<float>(), values.data<float>(), nr_positions, m_pos_dim, m_val_dim, 
                            // m_splatting_indices_and_weights_tensor.data<float>(), *(m_hash_table.m_impl) );
    TIME_END("just_create_verts");

    VLOG(3) << "after just_create_verts nr_verts is " << nr_lattice_vertices();
  
}


void Lattice::distribute(torch::Tensor& positions_raw, torch::Tensor& values){
    set_and_check_input(positions_raw, values);
    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);
    m_val_dim=values.size(2);
    m_val_full_dim=m_val_dim+1;


    // if( !m_distributed_tensor.defined() || m_distributed_tensor.size(0)!=nr_positions ){
    //     m_distributed_tensor = torch::zeros({ nr_positions , m_pos_dim+1 , (m_pos_dim+1) + (m_val_dim+1)  });
    //     m_distributed_tensor = m_distributed_tensor.to("cuda");
    // }
    if( !m_distributed_tensor.defined() || m_distributed_tensor.size(0)!=nr_positions*(m_pos_dim+1) ){
        m_distributed_tensor = torch::zeros({ nr_positions *(m_pos_dim+1) , m_pos_dim + m_val_dim +1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_distributed_tensor.fill_(0);
    }


    //if it's not initialized to the correct values we intialize the hashtable
    // if(m_hash_table->m_pos_dim!=m_pos_dim || m_hash_table->m_val_dim!=m_val_dim){
    if(!m_hash_table->m_keys_tensor.defined()){
        // m_first_time=false;
        // m_hash_table= HashTable(m_hash_table_capacity, m_pos_dim, m_val_dim);
        m_hash_table->init(m_hash_table_capacity, m_pos_dim, m_val_dim);
        m_hash_table->to(torch::kCUDA);

    }
    if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
        m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1)}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0)  );
        m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }

    // m_splatting_indices_and_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1), 2 });
    m_splatting_indices_tensor.fill_(-1);
    m_splatting_weights_tensor.fill_(-1);

    m_hash_table->clear();


    //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    values=values.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma");

    TIME_START("distribute");
    m_impl->distribute(positions.data<float>(), values.data<float>(), m_distributed_tensor.data<float>(), nr_positions, m_pos_dim, m_val_dim, 
                            m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), *(m_hash_table->m_impl) );
    TIME_END("distribute");

    // VLOG(1) << "m_distributed_tensor is " << m_distributed_tensor;

    VLOG(3) << "after distributing nr_verts is " << nr_lattice_vertices();
  
}

        // void create_splatting_mask(bool* mask,  const int* splatting_indices, const int* nr_points_per_simplex, const int nr_positions, const int pos_dim, curandState* globalState ){
Tensor Lattice::create_splatting_mask(const torch::Tensor& nr_points_per_simplex, const int nr_positions, const int max_nr_points){
    // if(nr_positions*(m_pos_dim+1)>m_nr_states){
        // LOG(FATAL) << "We don't have enough random states to for each thread. Increase the nr of states";
    // }

    Tensor mask = torch::zeros({nr_positions*(m_pos_dim+1)}, torch::dtype(torch::kBool).device(torch::kCUDA, 0)  );

    // m_impl->create_splatting_mask(mask.data<bool>(), m_splatting_indices_tensor.data<int>(), nr_points_per_simplex.data<int>(), max_nr_points, nr_positions, m_pos_dim, m_devStates); 
    m_impl->create_splatting_mask(mask.data<bool>(), m_splatting_indices_tensor.data<int>(), nr_points_per_simplex.data<int>(), max_nr_points, nr_positions, m_pos_dim); 

    return mask;
}




std::shared_ptr<Lattice> Lattice::blur_standalone(){

    std::shared_ptr<Lattice> blurred_lattice=create(this); //create a lattice with no config but takes the config from this one
    blurred_lattice->m_name="blurred_lattice";
   
    blurred_lattice->m_hash_table->m_values_tensor=m_hash_table->m_values_tensor.clone();
    blurred_lattice->m_tmp_blurred_values_tensor =m_hash_table->m_values_tensor.clone();
    blurred_lattice->m_hash_table->m_values_tensor=blurred_lattice->m_hash_table->m_values_tensor.to("cuda");
    blurred_lattice->m_tmp_blurred_values_tensor=blurred_lattice->m_tmp_blurred_values_tensor.to("cuda");
    // VLOG(1) << "updating impl";
    blurred_lattice->m_hash_table->update_impl(); //updating the hash table pointer to point to the newly clones values tensor


    TIME_START("blur");
    for (int remainder=0; remainder >= 0 && remainder <= m_pos_dim; remainder++) {

        if(remainder==0){
            //if it's the first time we do a blur along an axis we blur the values from the current hashtable and store them in the blurred_lattice hash_table_values
            m_impl->blur_standalone(m_hash_table_capacity, m_pos_dim, m_val_full_dim, 
                blurred_lattice->m_hash_table->m_values_tensor.data<float>(), remainder, *(m_hash_table->m_impl) );
            // m_impl->blur_standalone(m_hash_table_capacity, m_pos_dim, m_val_dim, blurred_lattice->m_tmp_blurred_values_tensor, remainder, *(m_hash_table.m_impl) );
            m_hash_table->update_impl(); //when swapping we are changing ptr, but this need to be propagated to the cuda implementation too
        }else{
            //if the m_tmp_blurred_values_tensor is not defined or not the correct size we allocate it
            // if()

            //if its the second or above time we do a blur, we blur from the values of the blur lattice and store in the m_tmp_blurred_values_tensor and then we swap. In the end we remove the m_tmp_blurred_values_tensor
            blurred_lattice->m_impl->blur_standalone(m_hash_table_capacity, m_pos_dim, m_val_full_dim, 
                blurred_lattice->m_tmp_blurred_values_tensor.data<float>(), remainder, *(blurred_lattice->m_hash_table->m_impl) );

            std::swap(blurred_lattice->m_hash_table->m_values_tensor, blurred_lattice->m_tmp_blurred_values_tensor);
            blurred_lattice->m_hash_table->update_impl(); //when swapping we are changing ptr, but this need to be propagated to the cuda implementation too
        }
    }
    TIME_END("blur");
    CUDA_CHECK_ERROR();

    //TODO finished blurring, we remove the m_tmp_blurred_values_tensor

    return blurred_lattice;

}


std::shared_ptr<Lattice> Lattice::convolve_im2row_standalone(torch::Tensor& filter_bank, const int dilation, const bool with_homogeneous_coord, std::shared_ptr<Lattice> lattice_neighbours, const bool use_center_vertex_from_lattice_neigbhours, const bool flip_neighbours){


    CHECK(filter_bank.defined()) << "Filter bank is undefined";
    CHECK(filter_bank.dim()==2) << "Filter bank should have dimension 2, corresponding with (filter_extent * val_dim+1) x nr_filters.  However it has dimension: " << filter_bank.dim();
    // CHECK(filter_bank.size(0)== 2*(m_pos_dim+1)+1 ) <<"Filter extent should cover nr of vertices corresponding to a 1 hop neighborhood. Bigger neighbourhoods are not yet implemented. That means it should be 2*(m_pos_dim+1)+1 which would be" << 2*(m_pos_dim+1)+1 << "however the filter_bank.size(1) is " << filter_bank.size(1);
    // CHECK(filter_bank.size(2) == m_val_dim+1) << "Filters should convolve over all the values of this lattice so the m_val_dim+1 which is " << m_val_dim+1 << "which is " << "should be equal to filter_bank.size(2) which is " << filter_bank.size(2);

    int nr_filters=filter_bank.size(1) ;
    int filter_extent=filter_bank.size(0) / m_val_full_dim;
    // VLOG(1) << "filter_bank sizes is" << filter_bank.sizes();
    CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent;

    //this lattice should be coarser (so a higher lvl) or at least at the same lvl as the lattice neigbhours (which is a finer lvl therefore the lattice_neigbhours.m_lvl is lower)
    CHECK(m_lvl-lattice_neighbours->m_lvl<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;
    
    VLOG(4) <<"starting convolved im2row_standlaone. The current lattice has nr_vertices_lattices" << nr_lattice_vertices();
    CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
    int nr_vertices=nr_lattice_vertices();
    // VLOG(1)

    std::shared_ptr<Lattice> convolved_lattice=create(this); //create a lattice with no config but takes the config from this one
    convolved_lattice->m_name="convolved_lattice";
    int cur_hash_table_size=m_hash_table->m_values_tensor.size(0);
    // VLOG(1) << "cloning values tensor which has size" << cur_hash_table_size;
    // VLOG(1) << "cloning values tensor which has size" << cur_hash_table_size;
    if(with_homogeneous_coord){
        convolved_lattice->m_hash_table->m_values_tensor=torch::zeros({m_hash_table_capacity, nr_filters+1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ) ; // +1 because of homogeneous coordinates
    }else{
        //no need to allocate because it will be directly set to be the whatever comes from the matrix mutliplicaiton between lattice_rowified and filter bank
    }
    // convolved_lattice->m_hash_table->m_values_tensor=convolved_lattice->m_hash_table->m_values_tensor.to("cuda");

    //m_val_dim and m_val_full_dim are equal now
    if(with_homogeneous_coord){
        convolved_lattice->m_val_dim=nr_filters;
        convolved_lattice->m_val_full_dim=nr_filters+1;
    }else{
        convolved_lattice->m_val_dim=nr_filters;
        convolved_lattice->m_val_full_dim=nr_filters;
    }
    convolved_lattice->m_hash_table->update_impl(); //updating the hash table pointer to point to the newly clones values tensor

    //kernel bank is of size nr_filers x filter_extent x in_val_dim
    filter_bank=filter_bank.to("cuda");

    //fill im2row TODO precompute it in the lattice

    TIME_START("create_lattice_rowified");
    // VLOG(1) << "checking if lattice rowified has size: nr_vertices" << nr_vertices << " filter_extent " << filter_extent << " m_val_full_dim " << m_val_full_dim;
    if( !m_lattice_rowified.defined() || m_lattice_rowified.size(0)!=nr_vertices || m_lattice_rowified.size(1)!=filter_extent*m_val_full_dim  ){
        // VLOG(1) << "Creating a lattice rowified with size: nr_vertices" << nr_vertices << " filter_extent " << filter_extent << " m_val_full_dim " << m_val_full_dim;
        m_lattice_rowified=torch::zeros({nr_vertices, filter_extent*m_val_full_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_lattice_rowified.fill_(0);
    }
    TIME_END("create_lattice_rowified");

    TIME_START("convolve_im2row");
    TIME_START("im2row");
    bool debug_kernel=false;
    if(m_lvl==2){
        // debug_kernel=true;
    }

    // VLOG(1) << "calling im2row with lattice neighbours which have vlaues of norm " << lattice_neighbours->m_hash_table->m_values_tensor.norm();
    VLOG(4) <<"calling im2row with m_val_full_dim of " << m_val_full_dim;
    m_impl->im2row(nr_vertices, m_pos_dim, m_val_full_dim, dilation, m_lattice_rowified.data<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, use_center_vertex_from_lattice_neigbhours, flip_neighbours, debug_kernel);

    // m_impl->test_row2im(m_hash_table_capacity, m_pos_dim, m_val_full_dim, dilation, m_lattice_rowified.data<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, use_center_vertex);
    TIME_END("im2row");

    // VLOG(1) <<"lattice rowified is \n" << m_lattice_rowified;
    // Tensor lattice_rowified_unsqueezed=m_lattice_rowified.unsqueeze(0);
    // EigenMatrixXfRowMajor lattice_rowified_eigen=tensor2eigen(lattice_rowified_unsqueezed);
    // VLOG(1) <<"lattice rowified is \n" << lattice_rowified_eigen;

    // VLOG(1) << "im2row should have at least some non zero value pero row. The rowsise sum of lattice_rowified is " << m_lattice_rowified.sum(1);


    //multiply each patch with the filter bank
    Tensor convolved= m_lattice_rowified.mm(filter_bank);
    // VLOG(1) << "finished multiplication";
    // VLOG(1) << "current values has shape" << m_hash_table->m_values_tensor.sizes();
    // VLOG(1) << "convolved_hash_table.values has shape" << convolved_lattice->m_hash_table->m_values_tensor.sizes();
    // VLOG(1) << "convolved has shape" << convolved.sizes();
    if(with_homogeneous_coord){
        // store both the convolved output and the homogeneous coordinate from the previous lattice
        convolved_lattice->m_hash_table->m_values_tensor.slice(1, 0, nr_filters) = convolved; //along dimension 0 (corresponding to the nr of columns) get from the column 0 the column nr_filters
        convolved_lattice->m_hash_table->m_values_tensor.slice(1, nr_filters, nr_filters+1) = m_hash_table->m_values_tensor.slice(1, m_val_dim, m_val_full_dim); //TODO the clone is just in case but it shouldnt teoretically be needed
    }else{
        convolved_lattice->m_hash_table->m_values_tensor=convolved;
    }
    convolved_lattice->m_hash_table->update_impl(); //very important

    TIME_END("convolve_im2row");

    VLOG(4) << "convolved lattice has nr filled " << convolved_lattice->nr_lattice_vertices();
    CHECK(convolved_lattice->nr_lattice_vertices()!=0) << "Why does this convolved lattice has zero nr_filled?";

    // VLOG(1) << "this lattice has lattice rowified of norm " <<m_lattice_rowified.norm();

    //FOR DEBUG assign the lattice rowified also the the convolve lattice so that we can query it afterwards and debug why there are vertices that don't have any neighbours
    // convolved_lattice->m_lattice_rowified=m_lattice_rowified.clone(); //IMPORTANT at the moment. Do not comment out
    convolved_lattice->m_lattice_rowified=m_lattice_rowified;

    return convolved_lattice;

}

torch::Tensor Lattice::im2row(std::shared_ptr<Lattice> lattice_neighbours, const int filter_extent, const int dilation, const bool use_center_vertex_from_lattice_neigbhours, const bool flip_neighbours){


    CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent;

    CHECK(m_lvl-lattice_neighbours->m_lvl<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level with respect to the neighbours. Or if they are at the same level then nothing needs to be done";
    
    VLOG(3) <<"starting convolved im2row_standlaone. The current lattice has nr_vertices_lattices" << nr_lattice_vertices();
    CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
    int nr_vertices=nr_lattice_vertices();


    TIME_START("create_lattice_rowified");
    if( !m_lattice_rowified.defined() || m_lattice_rowified.size(0)!=nr_vertices || m_lattice_rowified.size(1)!=filter_extent*m_val_full_dim  ){
        m_lattice_rowified=torch::zeros({nr_vertices, filter_extent*m_val_full_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_lattice_rowified.fill_(0);
    }
    TIME_END("create_lattice_rowified");

    bool debug_kernel=false;
    if(m_lvl==2){
        // debug_kernel=true;
    }

    VLOG(3) <<"calling im2row with m_val_full_dim of " << m_val_full_dim;
    m_impl->im2row(nr_vertices, m_pos_dim, m_val_full_dim, dilation, m_lattice_rowified.data<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, use_center_vertex_from_lattice_neigbhours, flip_neighbours, debug_kernel);

    return m_lattice_rowified;

}

torch::Tensor Lattice::row2im(const torch::Tensor& lattice_rowified,  const int dilation, const int filter_extent, const int nr_filters, std::shared_ptr<Lattice> lattice_neighbours, const bool use_center_vertex_from_lattice_neigbhours, const bool do_test){

    int nr_vertices=nr_lattice_vertices();
    if(!do_test){
        if(m_hash_table->m_values_tensor.size(0)!=nr_vertices || m_hash_table->m_values_tensor.size(1)==nr_filters){
            m_hash_table->m_values_tensor=torch::zeros({nr_vertices, nr_filters}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); 
        }else{
            m_hash_table->m_values_tensor.fill_(0);
        }
    }

    CHECK(nr_lattice_vertices()!=0) <<"Something went wrong because have zero lattice vertices";


    m_val_full_dim=nr_filters;
    m_hash_table->update_impl();

    m_impl->row2im(m_hash_table_capacity, m_pos_dim, m_val_full_dim, dilation, lattice_rowified.data<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, use_center_vertex_from_lattice_neigbhours, do_test);

    return m_hash_table->m_values_tensor;
}


std::shared_ptr<Lattice> Lattice::create_coarse_verts(){

    std::shared_ptr<Lattice> coarse_lattice=create(this); //create a lattice with no config but takes the config from this one
    coarse_lattice->m_name="coarse_lattice";
    coarse_lattice->m_lvl=m_lvl+1;
    coarse_lattice->m_sigmas_tensor=m_sigmas_tensor.clone()*2.0; //the sigma for the coarser one is double. This is done so if we slice at this lattice we scale the positions with the correct sigma
    for(int i=0; i<m_sigmas.size(); i++){
        coarse_lattice->m_sigmas[i]=m_sigmas[i]*2.0;
    } 
    // coarse_lattice->m_splatting_indices_tensor=torch::zeros({nr_positions*(m_pos_dim+1)}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0)  ); //we need them for when we slice the finer vertices form this coarse lattice
    // coarse_lattice->m_splatting_weights_tensor=torch::zeros({nr_positions*(m_pos_dim+1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1,1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
    coarse_lattice->m_hash_table->m_keys_tensor=torch::zeros({m_hash_table_capacity, m_pos_dim}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->m_entries_tensor=torch::zeros({m_hash_table_capacity}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) ) ;
    coarse_lattice->m_hash_table->m_nr_filled_tensor=torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->clear();
    // coarse_lattice->m_splatting_indices_tensor.fill_(-1);
    // coarse_lattice->m_splatting_weights_tensor.fill_(-1);
    coarse_lattice->m_hash_table->update_impl();

    TIME_START("coarsen");
    m_impl->coarsen(m_hash_table_capacity, m_pos_dim, *(m_hash_table->m_impl), *(coarse_lattice->m_hash_table->m_impl)  );
    TIME_END("coarsen");

    int nr_vertices=coarse_lattice->nr_lattice_vertices();
    VLOG(3) << "after coarsening nr_verts of the coarse lattice is " << nr_vertices;

    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({nr_vertices, m_val_full_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  ); //we create exactly the values required for he vertices that were allocated
    coarse_lattice->m_hash_table->update_impl();

    return coarse_lattice;

}


std::shared_ptr<Lattice> Lattice::create_coarse_verts_naive(torch::Tensor& positions_raw){

    std::shared_ptr<Lattice> coarse_lattice=create(this); //create a lattice with no config but takes the config from this one
    // VLOG(1) << "new coarse lattice has splatting incies" << m_splatting_indices_tensor;
    coarse_lattice->m_name="coarse_lattice";
    coarse_lattice->m_lvl=m_lvl+1;
    coarse_lattice->m_sigmas_tensor=m_sigmas_tensor.clone()*2.0; //the sigma for the coarser one is double. This is done so if we slice at this lattice we scale the positions with the correct sigma
    coarse_lattice->m_sigmas=m_sigmas;
    for(int i=0; i<m_sigmas.size(); i++){
        coarse_lattice->m_sigmas[i]=m_sigmas[i]*2.0;
    } 
    // coarse_lattice->m_splatting_indices_tensor=m_splatting_indices_tensor.clone(); //we need them for when we slice the finer vertices form this coarse lattice
    // coarse_lattice->m_splatting_weights_tensor=m_splatting_weights_tensor.clone();
    // coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1,1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
    // coarse_lattice->m_hash_table->m_keys_tensor=m_hash_table->m_keys_tensor.clone();
    // coarse_lattice->m_hash_table->m_entries_tensor=m_hash_table->m_entries_tensor.clone();
    // coarse_lattice->m_hash_table->m_nr_filled_tensor=m_hash_table->m_nr_filled_tensor.clone();
    // coarse_lattice->m_splatting_indices_tensor=torch::zeros({nr_positions*(m_pos_dim+1)}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0)  ); //we need them for when we slice the finer vertices form this coarse lattice
    // coarse_lattice->m_splatting_weights_tensor=torch::zeros({nr_positions*(m_pos_dim+1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1,1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
    coarse_lattice->m_hash_table->m_keys_tensor=torch::zeros({m_hash_table_capacity, m_pos_dim}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->m_entries_tensor=torch::zeros({m_hash_table_capacity}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) ) ;
    coarse_lattice->m_hash_table->m_nr_filled_tensor=torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->clear();

    //just by creating the lattice from this one we  copy the splatting indices and tensors because we will either way slice at the finest lattice at the end

    coarse_lattice->m_hash_table->update_impl();

    // coarse_lattice->m_splatting_indices_tensor.fill_(-1);
    // coarse_lattice->m_splatting_weights_tensor.fill_(-1);

    // TIME_START("coarsen");
    // m_impl->coarsen(m_hash_table_capacity, m_pos_dim, *(m_hash_table->m_impl), *(coarse_lattice->m_hash_table->m_impl)  );
    // TIME_END("coarsen");




    coarse_lattice->begin_splat();
    coarse_lattice->m_hash_table->update_impl();

    coarse_lattice->just_create_verts(positions_raw, /*with_homogeneous_coord*/ false);


    //NO NEED  to create vertices because they will be create when the latice rowified gets multiplied with the filter
    // int nr_vertices=coarse_lattice->nr_lattice_vertices();
    // VLOG(1) << "after coarsening nr_verts of the coarse lattice is " << nr_vertices;
    // coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({nr_vertices, m_val_full_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ) ; //we create exactly the values required for he vertices that were allocated
    // coarse_lattice->m_val_full_dim=m_val_full_dim;
    // coarse_lattice->m_hash_table->update_impl();

    return coarse_lattice;

}


torch::Tensor Lattice::slice_standalone_no_precomputation(torch::Tensor& positions_raw, const bool with_homogeneous_coord){

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";
    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);


     //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma")

    //initialize the output values to zero 
    if( !m_sliced_values_hom_tensor.defined() || m_sliced_values_hom_tensor.size(0)!= 1 || m_sliced_values_hom_tensor.size(1)!=nr_positions || m_sliced_values_hom_tensor.size(2)!=m_val_full_dim){
        m_sliced_values_hom_tensor=torch::zeros({1, nr_positions, m_val_full_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_sliced_values_hom_tensor.fill_(0);
    }

    //recalculate the splatting indices and weight for the backward pass of the slice
    if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
        m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    m_splatting_indices_tensor.fill_(-1);
    m_splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    TIME_START("slice");
    m_impl->slice_standalone_no_precomputation( positions.data<float>(), m_sliced_values_hom_tensor.data<float>(), m_pos_dim, m_val_full_dim,  nr_positions, m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), *(m_hash_table->m_impl) );
    TIME_END("slice");

    if(with_homogeneous_coord){
        //divide by the homogeneous coordinate (m_sliced_values_tensor has size N x hashtable capacity x (val+1) )
        Tensor sliced_values_tensor=m_sliced_values_hom_tensor.slice(2, 0, m_val_full_dim-1).clone() / m_sliced_values_hom_tensor.slice(2, m_val_full_dim-1, m_val_full_dim); 
        // VLOG(1) << "m_sliced_values_hom_tensor sizes: " << sliced_values_tensor.sizes();
        // VLOG(1) << "sliced_values_tensor sizes: "<< sliced_values_tensor.sizes();
        return sliced_values_tensor;
    }else{
        return m_sliced_values_hom_tensor.clone(); // I clone it just in case because I know this will be used also for the backwards pass
    }

    // display_c10_cuda_mem_stat(0);

    //return the output values
    // m_output_values_tensor.to("cpu");
    // return  m_output_values_tensor;


}


torch::Tensor Lattice::gather_standalone_no_precomputation(torch::Tensor& positions_raw){

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";
    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);


     //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    VLOG(3) << "gather standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma")

    //initialize the output values to zero 
    int row_size_gathered=(m_pos_dim+1)*(m_val_full_dim+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
    if( !m_gathered_values_tensor.defined() || m_gathered_values_tensor.size(0)!= 1 || m_gathered_values_tensor.size(1)!=nr_positions || m_gathered_values_tensor.size(2)!=row_size_gathered){
        m_gathered_values_tensor=torch::zeros({1, nr_positions, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_gathered_values_tensor.fill_(0);
    }

    //recalculate the splatting indices and weight for the backward pass of the slice
    if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
        m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    m_splatting_indices_tensor.fill_(-1);
    m_splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    TIME_START("gather");
    m_impl->gather_standalone_no_precomputation( positions.data<float>(), m_gathered_values_tensor.data<float>(), m_pos_dim, m_val_full_dim,  nr_positions, m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), *(m_hash_table->m_impl) );
    TIME_END("gather");

    return m_gathered_values_tensor;

}

torch::Tensor Lattice::gather_elevated_standalone_no_precomputation(const std::shared_ptr<Lattice> lattice_to_gather_from){

    torch::Tensor keys=m_hash_table->m_keys_tensor;
    int nr_vertices=nr_lattice_vertices();
    m_pos_dim=keys.size(1);
    m_val_full_dim=lattice_to_gather_from->m_val_full_dim;

    CHECK(keys.scalar_type()==torch::kInt32) << "keys at which we gather should be of type int";
    CHECK(keys.dim()==2) << "keys should have dim 2 correspondin to capacity x pos_dim. However it has sizes" << keys.sizes();
    //set position and check that the sigmas were set correctly
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";


    //initialize the output values to zero 
    int row_size_gathered=(lattice_to_gather_from->m_pos_dim+1)*(lattice_to_gather_from->m_val_full_dim+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
    if( !m_gathered_values_tensor.defined() || m_gathered_values_tensor.size(0)!= 1 || m_gathered_values_tensor.size(1)!=nr_vertices || m_gathered_values_tensor.size(2)!=row_size_gathered){
        m_gathered_values_tensor=torch::zeros({1, nr_vertices, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_gathered_values_tensor.fill_(0);
    }

    //recalculate the splatting indices and weight for the backward pass of the slice
    if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_vertices*(m_pos_dim+1)  ){
        m_splatting_indices_tensor = torch::zeros({nr_vertices*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        m_splatting_weights_tensor = torch::zeros({nr_vertices*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    m_splatting_indices_tensor.fill_(-1);
    m_splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    TIME_START("gather");
    m_impl->gather_elevated_standalone_no_precomputation( keys.data<int>(), m_gathered_values_tensor.data<float>(), m_pos_dim, lattice_to_gather_from->m_val_full_dim,  nr_vertices, m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), *(lattice_to_gather_from->m_hash_table->m_impl), lattice_to_gather_from->m_lvl, m_lvl );
    TIME_END("gather");

    return m_gathered_values_tensor;

}


torch::Tensor Lattice::gather_standalone_with_precomputation(torch::Tensor& positions_raw){

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";
    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);


     //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    VLOG(3) << "gather standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma")

    //initialize the output values to zero 
    int row_size_gathered=(m_pos_dim+1)*(m_val_full_dim+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
    if( !m_gathered_values_tensor.defined() || m_gathered_values_tensor.size(0)!= 1 || m_gathered_values_tensor.size(1)!=nr_positions || m_gathered_values_tensor.size(2)!=row_size_gathered){
        m_gathered_values_tensor=torch::zeros({1, nr_positions, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_gathered_values_tensor.fill_(0);
    }

    //assume we have already splatting weight and indices
    if( !m_splatting_indices_tensor.defined() || !m_splatting_weights_tensor.defined()  || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1) ||  m_splatting_weights_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
        LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(m_pos_dim+1) << "but indices has size " << m_splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << m_splatting_weights_tensor.sizes();
    //     m_splatting_indices_tensor = torch::zeros({nr_vertices*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    //     m_splatting_weights_tensor = torch::zeros({nr_vertices*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    // if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
    //     m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    //     m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    // }
    // m_splatting_indices_tensor.fill_(-1);
    // m_splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    TIME_START("gather");
    m_impl->gather_standalone_with_precomputation( positions.data<float>(), m_gathered_values_tensor.data<float>(), m_pos_dim, m_val_full_dim,  nr_positions, m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), *(m_hash_table->m_impl) );
    TIME_END("gather");

    return m_gathered_values_tensor;

}

std::shared_ptr<Lattice> Lattice::slice_elevated_verts(const std::shared_ptr<Lattice> lattice_to_slice_from){
    // hash_table_to_slice_from
        // void slice_elevated_verts(const int hash_table_capacity, float* sliced_values, const int pos_dim, const int val_full_dim, int* splatting_indices, float* splatting_weights, const HashTableGPU& hash_table_to_slice_from, const HashTableGPU& hash_table_elevated_verts, const int lattice_to_slice_from_lvl, const int elevated_vert_lvl){

    int nr_elevated_verts=nr_lattice_vertices();
    int pos_dim_slice_from=lattice_to_slice_from->m_pos_dim;
    int val_full_dim_slice_from=lattice_to_slice_from->m_val_full_dim;
    int val_dim_slice_from=lattice_to_slice_from->m_val_dim;


    std::shared_ptr<Lattice> sliced_lattice=create(this); //create a lattice with no config but takes the config from this one
    sliced_lattice->m_name="sliced_lattice";
    sliced_lattice->m_pos_dim=pos_dim_slice_from;
    sliced_lattice->m_val_full_dim=val_full_dim_slice_from;
    sliced_lattice->m_val_dim=val_dim_slice_from;


    //create the values in which we will store the sliced values
    if( !sliced_lattice->m_hash_table->m_values_tensor.defined() || sliced_lattice->m_hash_table->m_values_tensor.size(0)!= nr_elevated_verts || sliced_lattice->m_hash_table->m_values_tensor.size(1)!=val_full_dim_slice_from){
        sliced_lattice->m_hash_table->m_values_tensor=torch::zeros({nr_elevated_verts, val_full_dim_slice_from}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        sliced_lattice->m_hash_table->m_values_tensor.fill_(0);
    }

    //recalculate the splatting indices and weight for the backward pass of the slice
    if( !sliced_lattice->m_splatting_indices_tensor.defined() || sliced_lattice->m_splatting_indices_tensor.size(0)!=nr_elevated_verts*(pos_dim_slice_from+1)  ){
        sliced_lattice->m_splatting_indices_tensor = torch::zeros({nr_elevated_verts*(pos_dim_slice_from+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        sliced_lattice->m_splatting_weights_tensor = torch::zeros({nr_elevated_verts*(pos_dim_slice_from+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    sliced_lattice->m_splatting_indices_tensor.fill_(-1);
    sliced_lattice->m_splatting_weights_tensor.fill_(-1);
    sliced_lattice->m_hash_table->update_impl();


    TIME_START("slice_elev");
    VLOG(3)<< "calling sliced_elevated verts with sliced_lattice values tensor "<< sliced_lattice->m_hash_table->m_values_tensor.sizes() << "the val_full dim is " << val_full_dim_slice_from << "nr filled of " << m_hash_table->m_nr_filled_tensor;
    m_impl->slice_elevated_verts(m_hash_table_capacity, sliced_lattice->m_hash_table->m_values_tensor.data<float>(), pos_dim_slice_from, val_full_dim_slice_from, 
                                sliced_lattice->m_splatting_indices_tensor.data<int>(), sliced_lattice->m_splatting_weights_tensor.data<float>(),
                                *(lattice_to_slice_from->m_hash_table->m_impl), *(sliced_lattice->m_hash_table->m_impl) ,
                                lattice_to_slice_from->m_lvl, sliced_lattice->m_lvl);
    TIME_END("slice_elev");

    return sliced_lattice;
    
}

torch::Tensor Lattice::slice_classify_no_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes){

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    int nr_positions=positions_raw.size(1);
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";


     //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    delta_weights=delta_weights.to("cuda");
    linear_clasify_weight=linear_clasify_weight.to("cuda");
    linear_clasify_bias=linear_clasify_bias.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma")

    //we store here the class logits directly
    if( !m_sliced_values_hom_tensor.defined() || m_sliced_values_hom_tensor.size(0)!= 1 || m_sliced_values_hom_tensor.size(1)!=nr_positions || m_sliced_values_hom_tensor.size(2)!=nr_classes){
        m_sliced_values_hom_tensor=torch::zeros({1, nr_positions, nr_classes}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_sliced_values_hom_tensor.fill_(0);
    }


    //recalculate the splatting indices and weight for the backward pass of the slice
    if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
        m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    m_splatting_indices_tensor.fill_(-1);
    m_splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    TIME_START("slice_classify");
    m_impl->slice_classify_no_precomputation( positions.data<float>(), 
                                              m_sliced_values_hom_tensor.data<float>(), 
                                              delta_weights.data<float>(), 
                                              linear_clasify_weight.data<float>(), 
                                              linear_clasify_bias.data<float>(), 
                                              nr_classes,
                                              m_pos_dim, 
                                              m_val_full_dim,  
                                              nr_positions, 
                                              m_splatting_indices_tensor.data<int>(), 
                                              m_splatting_weights_tensor.data<float>(), 
                                              *(m_hash_table->m_impl) );
    TIME_END("slice_classify");

    return m_sliced_values_hom_tensor;

}


torch::Tensor Lattice::slice_classify_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes){

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    int nr_positions=positions_raw.size(1);
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";


     //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    delta_weights=delta_weights.to("cuda");
    linear_clasify_weight=linear_clasify_weight.to("cuda");
    linear_clasify_bias=linear_clasify_bias.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma")

    //we store here the class logits directly
    if( !m_sliced_values_hom_tensor.defined() || m_sliced_values_hom_tensor.size(0)!= 1 || m_sliced_values_hom_tensor.size(1)!=nr_positions || m_sliced_values_hom_tensor.size(2)!=nr_classes){
        m_sliced_values_hom_tensor=torch::zeros({1, nr_positions, nr_classes}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }else{
        m_sliced_values_hom_tensor.fill_(0);
    }


    // //recalculate the splatting indices and weight for the backward pass of the slice
    // if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
    //     m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    //     m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    // }
    // m_splatting_indices_tensor.fill_(-1);
    // m_splatting_weights_tensor.fill_(-1);

    //assume we have already splatting weight and indices
    if( !m_splatting_indices_tensor.defined() || !m_splatting_weights_tensor.defined()  || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1) ||  m_splatting_weights_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
        LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(m_pos_dim+1) << "but indices has size " << m_splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << m_splatting_weights_tensor.sizes();
    //     m_splatting_indices_tensor = torch::zeros({nr_vertices*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    //     m_splatting_weights_tensor = torch::zeros({nr_vertices*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }
    m_hash_table->update_impl();


    TIME_START("slice_classify_cuda");
    m_impl->slice_classify_with_precomputation( positions.data<float>(), 
                                              m_sliced_values_hom_tensor.data<float>(), 
                                              delta_weights.data<float>(), 
                                              linear_clasify_weight.data<float>(), 
                                              linear_clasify_bias.data<float>(), 
                                              nr_classes,
                                              m_pos_dim, 
                                              m_val_full_dim,  
                                              nr_positions, 
                                              m_splatting_indices_tensor.data<int>(), 
                                              m_splatting_weights_tensor.data<float>(), 
                                              *(m_hash_table->m_impl) );
    TIME_END("slice_classify_cuda");

    return m_sliced_values_hom_tensor;

}





void Lattice::slice_backwards_standalone_with_precomputation(torch::Tensor& positions_raw, const torch::Tensor& sliced_values_hom, const Tensor& grad_sliced_values){

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";
    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);



    TIME_START("slice_back");
    m_impl->slice_backwards_standalone_with_precomputation( sliced_values_hom.data<float>(), grad_sliced_values.data<float>(), m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), m_pos_dim, m_val_dim, nr_positions, *(m_hash_table->m_impl) );
    TIME_END("slice_back");

    // return  m_output_values_tensor;


}


void Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous(torch::Tensor& positions_raw, const Tensor& grad_sliced_values){

    // VLOG(1) <<"slice_backwards_standalone_with_precomputation_no_homogeneous got positions" << positions_raw;
    // VLOG(1) <<"slice_backwards_standalone_with_precomputation_no_homogeneous got grad_sliced_values" << grad_sliced_values;

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";
    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);
    CHECK(grad_sliced_values.dim()==3) <<"grad_sliced_values should be nr_batches x nr_positions x m_val_full_dim, so it should have 3 dimensions. However it has "<< grad_sliced_values.dim();

    // m_hash_table->m_values_tensor=torch::zeros({m_hash_table_capacity, grad_sliced_values.size(2)});
    // m_hash_table->m_values_tensor=torch::zeros({m_hash_table_capacity, grad_sliced_values.size(2)}).to("cuda");
    // m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), grad_sliced_values.size(2)},  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // m_hash_table->update_impl();

    //no need to reallocate the values, we just need to check that they have the correct size
    if(m_hash_table->m_values_tensor.size(0) != nr_lattice_vertices() || m_hash_table->m_values_tensor.size(1)!=grad_sliced_values.size(2) ){
        // LOG(WARNING) << "Reallocating the values tensor which might be quite slow.";
        m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), grad_sliced_values.size(2)},  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }else{
        m_hash_table->m_values_tensor.fill_(0);
    }
    m_hash_table->update_impl();



    TIME_START("slice_back");
    m_impl->slice_backwards_standalone_with_precomputation_no_homogeneous(grad_sliced_values.data<float>(), m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), m_pos_dim, m_val_full_dim, nr_positions, *(m_hash_table->m_impl) );
    TIME_END("slice_back");

    // VLOG(1) << "slice_backwards_standalone_with_precomputation_no_homogeneous at the finale we have  values vector of "<<m_hash_table->m_values_tensor;

    // return  m_output_values_tensor;


}

void Lattice::slice_backwards_elevated_verts_with_precomputation(const std::shared_ptr<Lattice> lattice_sliced_from, const Tensor& grad_sliced_values, const int nr_verts_to_slice_from){

    CHECK(grad_sliced_values.dim()==2) <<"grad_sliced_values should be nr_positions x m_val_full_dim, so it should have 2 dimensions. However it has "<< grad_sliced_values.dim();

    m_val_full_dim=grad_sliced_values.size(1);
    int nr_positions=nr_lattice_vertices();

    // m_hash_table->m_values_tensor=torch::zeros({m_hash_table_capacity, grad_sliced_values.size(2)});
    // m_hash_table->m_values_tensor=torch::zeros({m_hash_table_capacity, grad_sliced_values.size(1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    //no need to reallocate the values, we just need to check that they have the correct size
    if(m_hash_table->m_values_tensor.size(0) != nr_lattice_vertices() || m_hash_table->m_values_tensor.size(1)!=nr_verts_to_slice_from ){
        LOG(WARNING) << "Reallocating the values tensor which might be quite slow.";
        lattice_sliced_from->m_hash_table->m_values_tensor=torch::zeros({nr_verts_to_slice_from, grad_sliced_values.size(1)},  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }else{
        lattice_sliced_from->m_hash_table->m_values_tensor.fill_(0);
    }
    lattice_sliced_from->m_hash_table->update_impl();



    TIME_START("slice_back");
    m_impl->slice_backwards_standalone_with_precomputation_no_homogeneous(grad_sliced_values.data<float>(), m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), m_pos_dim, m_val_full_dim, nr_positions, *(lattice_sliced_from->m_hash_table->m_impl) );
    TIME_END("slice_back");

    // return  m_output_values_tensor;


}

void Lattice::slice_classify_backwards_with_precomputation(const torch::Tensor& grad_class_logits, torch::Tensor& positions_raw, torch::Tensor& initial_values, torch::Tensor& delta_weights, torch::Tensor&  linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& grad_lattice_values, torch::Tensor& grad_delta_weights, torch::Tensor& grad_linear_clasify_weight, torch::Tensor& grad_linear_clasify_bias){

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    m_pos_dim=positions_raw.size(2);
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";
    int nr_positions=positions_raw.size(1);
    CHECK(grad_class_logits.dim()==3) <<"grad_class_logits should be nr_batches x nr_positions x nr_classes, so it should have 3 dimensions. However it has "<< grad_class_logits.dim();
    m_val_full_dim=initial_values.size(1);
    m_val_dim=initial_values.size(1);


    // //no need to reallocate the values, we just need to check that they have the correct size
    // if(m_hash_table->m_values_tensor.size(0) != nr_lattice_vertices() || m_hash_table->m_values_tensor.size(1)!=m_val_full_dim ){
    //     LOG(WARNING) << "Reallocating the values tensor which might be quite slow.";
    //     m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), m_val_full_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // }else{
    //     m_hash_table->m_values_tensor.fill_(0);
    // }
    // m_hash_table->update_impl();



    TIME_START("slice_clasify_back");
    m_impl->slice_classify_backwards_with_precomputation(grad_class_logits.data<float>(), initial_values.data<float>(),  m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), m_pos_dim, m_val_full_dim, nr_positions,
    delta_weights.data<float>(), linear_clasify_weight.data<float>(), linear_clasify_bias.data<float>(), nr_classes, grad_lattice_values.data<float>(), grad_delta_weights.data<float>(), grad_linear_clasify_weight.data<float>(),grad_linear_clasify_bias.data<float>(),
     *(m_hash_table->m_impl) );
    TIME_END("slice_clasify_back");

}

void Lattice::gather_backwards_standalone_with_precomputation(const torch::Tensor& positions_raw, const Tensor& grad_sliced_values){

    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);
    m_val_full_dim=grad_sliced_values.size(2)/(m_pos_dim+1)-1; //we will acumulate the gradient into the value tensor. And it should have the same val_dim as the values that were in the lattice_we_gathered from

    // set_and_check_input(positions_raw, values);
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==3) << "positions should have dim 3 correspondin to NHW. However it has sizes" << positions_raw.sizes();
    //set position and check that the sigmas were set correctly
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set.";
    CHECK(grad_sliced_values.dim()==3) <<"grad_sliced_values should be nr_batches x nr_positions x ((m_val_full_dim+1)*(m_pos_dim+1)), so it should have 3 dimensions. However it has "<< grad_sliced_values.dim();



    
    // m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), grad_sliced_values.size(2)},  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // m_hash_table->update_impl();

    //no need to reallocate the values, we just need to check that they have the correct size
    if(m_hash_table->m_values_tensor.size(0) != nr_lattice_vertices() || m_hash_table->m_values_tensor.size(1)!=m_val_full_dim ){
        // LOG(WARNING) << "Reallocating the values tensor which might be quite slow.";
        m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), m_val_full_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }else{
        m_hash_table->m_values_tensor.fill_(0);
    }
    m_hash_table->update_impl();



    // TIME_START("gather_back");
    m_impl->gather_backwards_standalone_with_precomputation(grad_sliced_values.data<float>(), m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), m_pos_dim, m_val_full_dim, nr_positions, *(m_hash_table->m_impl) );
    // TIME_END("gather_back");


}

void Lattice::gather_backwards_elevated_standalone_with_precomputation(const std::shared_ptr<Lattice> lattice_gathered_from, const Tensor& grad_sliced_values){

    torch::Tensor keys=m_hash_table->m_keys_tensor;
    int nr_vertices=nr_lattice_vertices();
    int nr_vertices_in_lattice_gathered_from=lattice_gathered_from->nr_lattice_vertices();
    // m_pos_dim=keys.size(1);
    // m_val_full_dim=grad_sliced_values.size(2)/(m_pos_dim+1)-1; //we will acumulate the gradient into the value tensor. And it should have the same val_dim as the values that were in the lattice_we_gathered from

    CHECK(keys.scalar_type()==torch::kInt32) << "keys at which we gather should be of type int";
    CHECK(keys.dim()==2) << "keys should have dim 2 correspondin to capacity x pos_dim. However it has sizes" << keys.sizes();
    //set position and check that the sigmas were set correctly
    CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
    // CHECK(m_val_dim!=-1) << "m_val_dim is -1. We have to splat something first so that the m_val_dim gets set."
    CHECK(grad_sliced_values.dim()==3) <<"grad_sliced_values should be nr_batches x nr_vertices(this lattice) x ((m_val_full_dim+1)*(m_pos_dim+1)), so it should have 3 dimensions. However it has "<< grad_sliced_values.dim();



    //no need to reallocate the values, we just need to check that they have the correct size
    if(lattice_gathered_from->m_hash_table->m_values_tensor.size(0) != nr_vertices_in_lattice_gathered_from || lattice_gathered_from->m_hash_table->m_values_tensor.size(1)!=m_val_full_dim ){
        LOG(WARNING) << "Reallocating the values tensor which might be quite slow.";
        lattice_gathered_from->m_hash_table->m_values_tensor=torch::zeros({nr_vertices_in_lattice_gathered_from, m_val_full_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }else{
        lattice_gathered_from->m_hash_table->m_values_tensor.fill_(0);
    }
    lattice_gathered_from->m_hash_table->update_impl();



    // TIME_START("gather_back");
    m_impl->gather_backwards_standalone_with_precomputation(grad_sliced_values.data<float>(), m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), m_pos_dim, m_val_full_dim, nr_vertices, *(lattice_gathered_from->m_hash_table->m_impl) );
    // TIME_END("gather_back");


}



// torch::Tensor filter_rotate(const torch::Tensor filter_bank, const int filter_extent, const bool use_center_vertex){
//     //filter bank has size (filter_extent x val_full_dim) x nr_filters

//     int val_full_dim=filter_bank.size(0)/filter_extent;

//     Tensor filter_for_neighbours;
//     if(use_center_vertex){
//         filter_for_neighbours=filter_bank.slice(0, 0, filter_bank.size(0)-val_full_dim); //get the rows that correspond only to the neibhours because only those we will rotate
//     }else{
//         filter_for_neighbours=filter_bank;
//     }

// }


std::vector<Tensor> Lattice::to_tensors(){
    //TODO add also the m_splatting indices and weights

    // tensors=  values, keys, entries, nr_filled, sigmas, splatting_indices, splatting_weights
    std::vector<Tensor> tensors;
    // tensors.push_back(m_hash_table.m_values_tensor);
    // tensors.push_back(m_hash_table.m_keys_tensor);
    // tensors.push_back(m_hash_table.m_entries_tensor);
    // tensors.push_back(m_hash_table.m_nr_filled_tensor);
    // tensors.push_back(m_sigmas_tensor);
    // tensors.push_back(m_splatting_indices_tensor);
    // tensors.push_back(m_splatting_weights_tensor);
    return tensors;
}

void Lattice::from_tensors(const std::vector<torch::Tensor>& tensors){
    //set all the tensors in the right place
    // m_hash_table.m_values_tensor=tensors[0];
    // m_hash_table.m_keys_tensor=tensors[1];
    // m_hash_table.m_entries_tensor=tensors[2];
    // m_hash_table.m_nr_filled_tensor=tensors[3];
    // m_sigmas_tensor=tensors[4];
    // m_splatting_indices_tensor=tensors[5];
    // m_splatting_weights_tensor=tensors[6];

    // //set pos dim and val dim and hash_table_capacity
    // m_pos_dim=m_hash_table.m_keys_tensor.size(1);
    // m_val_dim=m_hash_table.m_values_tensor.size(1)-1;
    // m_hash_table_capacity=m_hash_table.m_keys_tensor.size(0);
    // m_hash_table.m_pos_dim=m_pos_dim; 
    // m_hash_table.m_val_dim=m_val_dim; 
    // m_hash_table.m_capacity=m_hash_table_capacity;
    // m_hash_table.update_impl();

}


std::shared_ptr<Lattice> Lattice::clone_lattice(){
    std::shared_ptr<Lattice> new_lattice=create(this); //create a lattice with no config but takes the config from this one
    return new_lattice;
}

//retuns the keys of the lattice as vertices. We cannot retunr a mesh because nvcc complains about compiling the MeshCore with betterenum
Eigen::MatrixXd Lattice::keys_to_verts(){
    CHECK(m_pos_dim==2) << "In order to show the keys as a mesh the pos_dim has to be 2 because only then the keys will be in 3D space and not in something bigger";

    Tensor keys=m_hash_table->m_keys_tensor.clone();
    keys=keys.unsqueeze(0);
    keys=keys.to(at::kFloat);
    EigenMatrixXfRowMajor keys_eigen_2D=tensor2eigen(keys);
    CHECK(keys_eigen_2D.cols()==2) << "The keys should be 2D keys because storing the full 3D one would be redundant as the key digits sum up to zero";


    //those keys only store the 2 dimensional part, we need to recreate the full m_pos_dim+1 key
    Eigen::MatrixXd V; 
    V.resize(keys_eigen_2D.rows(), 3);
    Eigen::VectorXf summed = keys_eigen_2D.rowwise().sum();
    for (int i=0; i < keys_eigen_2D.rows(); i++) {
        V(i,0)=keys_eigen_2D(i,0);
        V(i,1)=keys_eigen_2D(i,1);
        V(i,2)=-summed(i);
    }

    return V;
}

Eigen::MatrixXd Lattice::elevate(torch::Tensor& positions_raw){

    int nr_positions=positions_raw.size(1);
    m_pos_dim=positions_raw.size(2);

    //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma");

    Tensor elevated=torch::zeros({nr_positions,m_pos_dim+1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    elevated.fill_(0);

    TIME_START("elevate");
    m_impl->elevate(positions.data<float>(),  m_pos_dim, nr_positions, elevated.data<float>());
    TIME_END("elevate");

    elevated=elevated.unsqueeze(0);
    EigenMatrixXfRowMajor elevated_eigen_rowmajor=tensor2eigen(elevated);
    Eigen::MatrixXd elevated_eigen;
    elevated_eigen=elevated_eigen_rowmajor.cast<double>();
    return elevated_eigen;
}

Eigen::MatrixXd Lattice::deelevate(const torch::Tensor& keys){

    //get keys as eigen matrix
    Tensor keys_valid=keys.slice(0, 0, nr_lattice_vertices()).clone();
    EigenMatrixXfRowMajor keys_eigen_row_major=tensor2eigen(keys_valid.to(at::kFloat).unsqueeze(0));
    Eigen::MatrixXd keys_eigen(keys_eigen_row_major.rows(), m_pos_dim+1); //reconstruct the full key
    for (int i=0; i < keys_eigen.rows(); i++) {
        float sum=0;
        for (int j=0; j < m_pos_dim; j++) {
            keys_eigen(i,j)=keys_eigen_row_major(i,j);
            sum+=keys_eigen_row_major(i,j);
        }
        keys_eigen(i,m_pos_dim)=sum;
    }


    //create E matrix 
    Eigen::MatrixXd E=create_E_matrix(m_pos_dim);
    //inverse it 
    Eigen::MatrixXd E_inv=E.completeOrthogonalDecomposition().pseudoInverse();
    //multiply by inverse
    //scale by inv stddev 
    float invStdDev = (m_pos_dim + 1) * sqrt(2.0f / 3);
    //scale my sigmas
    Eigen::MatrixXd deelevated_vertices(keys_eigen_row_major.rows(),3);
    for (int i=0; i < keys_eigen_row_major.rows(); i++) {
        Eigen::VectorXd key=keys_eigen.row(i);
        Eigen::VectorXd vertex_deelevated= (E_inv*key).array()*invStdDev;
        for (int j=0; j < m_sigmas.size(); j++) {
            vertex_deelevated(j)=vertex_deelevated(j)*m_sigmas[j];
        }
        deelevated_vertices.row(i)=vertex_deelevated;
    }


    return deelevated_vertices;    
}

Eigen::MatrixXd Lattice::color_no_neighbours(){
    CHECK(m_lattice_rowified.size(0)==nr_lattice_vertices()) << "the lattice rowified should have rows for each vertex lattice. However we have a lattice rowified of size " << m_lattice_rowified.sizes() << " and nr of vertices is " << nr_lattice_vertices();

    VLOG(1) << "color_no_neihbours: Lattice rowified has size" << m_lattice_rowified.sizes();
    VLOG(1) << "color_no_neihbours: nr_lattice_vertices is " << nr_lattice_vertices();
    EigenMatrixXfRowMajor rowified_row_major=tensor2eigen(m_lattice_rowified.to(at::kFloat).unsqueeze(0));
    Eigen::MatrixXd C(nr_lattice_vertices(), 3);
    C.setZero();
    for (int i=0; i < rowified_row_major.rows(); i++) {
        float sum=0;
        for (int j=0; j < rowified_row_major.cols(); j++) {
            sum+=rowified_row_major(i,j);
        }
        if(sum==0){
            VLOG(1) << "setting row to red at idx " << i;
            C.row(i) << 1.0, 0.0, 0.0;
        }
    }

    return C;
}
Eigen::MatrixXd Lattice::create_E_matrix(const int pos_dim){

    //page 30 of Andrew Adams thesis
    Eigen::MatrixXf E_left(pos_dim+1, pos_dim );
    Eigen::MatrixXf E_right(pos_dim, pos_dim );
    E_left.setZero();
    E_right.setZero();
    //E left is has at the bottom a square matrix which has an upper triangular part of ones. Afterwards the whole E_left gets appended another row on top of all ones
    E_left.bottomRows(pos_dim).triangularView<Eigen::Upper>().setOnes();
    //the diagonal of the bottom square is linearly incresing from [-1, -m_pos_dim]
    E_left.bottomRows(pos_dim).diagonal().setLinSpaced(pos_dim,1,pos_dim);
    E_left.bottomRows(pos_dim).diagonal()= -E_left.bottomRows(pos_dim).diagonal();
    //E_left has the first row all set to ones
    E_left.row(0).setOnes();
    // VLOG(1) << "E left is \n" << E_left;
    //E right is just a diagonal matrix with entried in the diag set to 1/sqrt((d+1)(d+2)). Take into account that the d in the paper starts at 1 and we start at 0 so we add a +1 to diag_idx
    for(int diag_idx=0; diag_idx<pos_dim; diag_idx++){
        E_right(diag_idx, diag_idx) =  1.0 / (sqrt((diag_idx + 1) * (diag_idx + 2))) ;
    }
    // VLOG(1) << "E right is \n" << E_right;

    //rotate into H_d
    Eigen::MatrixXf E = E_left*E_right;

    return E.cast<double>();
}

void Lattice::increase_sigmas(const float stepsize){
        // m_sigmas.clear();
    for(int i=0; i<m_sigmas.size(); i++){
        m_sigmas[i]+=stepsize;
    }

    m_sigmas_tensor=vec2tensor(m_sigmas);
}













// void Lattice::splat_standalone(torch::Tensor& positions_raw, torch::Tensor& values){
//     set_and_check_input(positions_raw, values);
//     int nr_positions=positions_raw.size(1);
//     m_pos_dim=positions_raw.size(2);
//     m_val_dim=values.size(2);


//     if(m_first_time){
//         m_first_time=false;
//         m_hash_table= HashTable(m_hash_table_capacity, m_pos_dim, m_val_dim);
//         m_hash_table.to(torch::kCUDA);
//         // m_hash_table = m_hash_table.cuda();
//         // cudaMalloc((void**)&m_matrix, sizeof(MatrixEntry)*nr_positions* (m_pos_dim+1) );
//         // cudaMalloc((void**)&m_new_values, sizeof(float)*m_hash_table_capacity*(m_val_dim+1) );
//         m_new_values_tensor = torch::zeros({m_hash_table_capacity, m_val_dim+1});
//         // cudaMemset((void*)m_new_values, 0, sizeof(float)*m_hash_table_capacity*(m_val_dim+1) );
//     }else{
//         //allocate a vector of matrices big enough for the current nr of positions
//         // cudaFree(m_matrix);
//         // cudaMalloc((void**)&m_matrix, sizeof(MatrixEntry)*nr_positions* (m_pos_dim+1) );
//     }
//     m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }).to(torch::kInt32);
//     m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) });
//     m_splatting_indices_tensor.fill_(-1);
//     m_splatting_weights_tensor.fill_(-1);


//     //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     values=values.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     m_new_values_tensor=m_new_values_tensor.to("cuda");
//     m_splatting_indices_tensor=m_splatting_indices_tensor.to("cuda");
//     m_splatting_weights_tensor=m_splatting_weights_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma");


//     // do it with jitify
//     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     int cleanBlockSize = 128;
//     dim3 cleanBlocks((nr_positions - 1) / cleanBlockSize + 1, 2 * (m_pos_dim + 1), 1);


//     TIME_START("splat");
//     m_lattice_program.kernel("kernel_splat")
//                 .instantiate(m_pos_dim, m_val_dim)
//                 .configure(blocks, blockSize)
//                 .launch( positions.data<float>(), values.data<float>(), nr_positions, m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(),  *(m_hash_table.m_impl) );
//     TIME_END("splat");
//     CUDA_CHECK_ERROR();

//     //the nr of m_filled can be a lot bigger than the actual nr of lattice vertices because of the fact that the keys in the hashtable can have duplicates. While cleaning the hashtable we retreive all the keys and we check their index in the m_keys. The maximum index will be the new m_filled
//     // cudaMemset((void*)m_hash_table.m_filled, 0, sizeof(int)*1);

//     TIME_START("clean_hash");
//     m_lattice_program.kernel("cleanHashTable")
//                 .instantiate(m_pos_dim)
//                 .configure(cleanBlocks, cleanBlockSize)
//                 .launch( m_hash_table_capacity  , *(m_hash_table.m_impl) );
//     TIME_END("clean_hash");
//     CUDA_CHECK_ERROR();

//     TIME_START("splat_cache");
//     blocks.y = m_pos_dim + 1;
//     m_lattice_program.kernel("splatCache")
//                 .instantiate(m_pos_dim, m_val_dim)
//                 .configure(blocks, blockSize)
//                 .launch( nr_positions, values.data<float>(), m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), *(m_hash_table.m_impl) );
//     TIME_END("splat_cache");
//     CUDA_CHECK_ERROR();

//     VLOG(1) << "after splatting nr_verts is " << nr_lattice_vertices();
// }

// void LatticeGPU::blur_standalone(){

//      //to cuda
//     TIME_START("upload_cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");


//     // do it with jitify
//     int cleanBlockSize = 512;
//     dim3 cleanBlocks((m_hash_table_capacity - 1) / cleanBlockSize + 1, 2 * (m_pos_dim + 1), 1);
//     // dim3 cleanBlocks((960000 - 1) / cleanBlockSize + 1, 2 * (m_pos_dim + 1), 1);


//     TIME_START("blur");
//     for (int remainder=0; remainder >= 0 && remainder <= m_pos_dim; remainder++) {
//         m_lattice_program.kernel("blur")
//                 .instantiate(m_pos_dim, m_val_dim)
//                 .configure(cleanBlocks, cleanBlockSize)
//                 // .launch( m_hash_table_capacity, m_new_values_tensor.data<float>(), m_splatting_indices_tensor.data<int>(), m_splatting_weights_tensor.data<float>(), remainder, *(m_hash_table.m_impl) );
//                 .launch( m_hash_table_capacity, m_new_values_tensor.data<float>(), remainder, *(m_hash_table.m_impl) );
//         std::swap(m_hash_table.m_values_tensor, m_new_values_tensor);
//         m_hash_table.update_impl(); //when swapping we are changing ptr, but this need to be propagated to the cuda implementation too
//     }
//     TIME_END("blur");
//     CUDA_CHECK_ERROR();

// }

// torch::Tensor LatticeGPU::slice_standalone_no_precomputation(torch::Tensor& positions_raw, torch::Tensor& values){

//     set_and_check_input(positions_raw, values);
//     int nr_positions=positions_raw.size(1);
//     m_pos_dim=positions_raw.size(2);
//     m_val_dim=values.size(2);

//      //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     values=values.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma")

//     //initialize the output values to zero 
//     m_output_values_tensor=torch::zeros_like(values);
//     m_output_values_tensor=m_output_values_tensor.to("cuda");

//     // do it with jitify
//     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     int cleanBlockSize = 128;
//     dim3 cleanBlocks((nr_positions - 1) / cleanBlockSize + 1, 2 * (m_pos_dim + 1), 1);


//     TIME_START("slice");
//     blockSize.y = 1;
//     m_lattice_program.kernel("slice_no_precomputation")
//                 .instantiate(m_pos_dim, m_val_dim)
//                 .configure(blocks, blockSize)
//                 .launch( positions.data<float>(), m_output_values_tensor.data<float>(), nr_positions, *(m_hash_table.m_impl) );
//     TIME_END("slice");
//     CUDA_CHECK_ERROR();


//     //return the output values
//     m_output_values_tensor.to("cpu");
//     return  m_output_values_tensor;


// }

// void LatticeGPU::convolve_standalone(Tensor& kernel_bank){

//     //to cuda
//     TIME_START("upload_cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     kernel_bank=kernel_bank.to("cuda");
//     TIME_END("upload_cuda");

//     //create a new lattice to give as output, the lattice will have the same key matrix IF AND ONLY IF the convolution has no strides or antyhing like that, meaning that the size of the input and the ouput are the same


//     // // do it with jitify
//     // int cleanBlockSize = 512;
//     // dim3 cleanBlocks((m_hash_table_capacity - 1) / cleanBlockSize + 1, 2 * (m_pos_dim + 1), 1);
//     // // dim3 cleanBlocks((960000 - 1) / cleanBlockSize + 1, 2 * (m_pos_dim + 1), 1);


//     // TIME_START("blur");
//     // for (int remainder=0; remainder >= 0 && remainder <= m_pos_dim; remainder++) {
//     //     m_lattice_program.kernel("blur")
//     //             .instantiate(m_pos_dim, m_val_dim)
//     //             .configure(cleanBlocks, cleanBlockSize)
//     //             .launch( m_hash_table_capacity, m_new_values, m_matrix, remainder, m_hash_table );
//     //             // .launch( 960000, m_new_values, m_matrix, remainder, m_hash_table );
//     //     std::swap(m_hash_table.m_values, m_new_values);
//     // }
//     // TIME_END("blur");
//     // CUDA_CHECK_ERROR();

// }


int Lattice::nr_lattice_vertices(){
    // cudaDeviceSynchronize();
    // int nr_verts[1];
    // cudaMemcpy(nr_verts, m_hash_table.m_filled, sizeof(int), cudaMemcpyDeviceToHost);    
    // return nr_verts[0];

    // torch::Tensor nr_filled_cpu=m_hash_table->m_nr_filled_tensor.clone();
    // nr_filled_cpu.to("cpu");
    // return nr_filled_cpu.item<int>();

    //attempt 2 at making it faster
    m_impl->wait_to_create_vertices(); //we synchronize the event and wait until whatever kernel was launched to create vertices has also finished
    int nr_verts=0;
    cudaMemcpy ( &nr_verts,  m_hash_table->m_nr_filled_tensor.data<int>(), sizeof(int), cudaMemcpyDeviceToHost );
    // cudaMemcpy ( &nr_verts,  m_hash_table->m_impl->m_nr_filled, sizeof(int), cudaMemcpyDeviceToHost );
    // VLOG(1) << "nr of verts is " << m_hash_table->m_nr_filled_tensor;
    CHECK(nr_verts>=0) << "nr vertices cannot be negative. However it is ", nr_verts;
    CHECK(nr_verts<1e+8) << "nr vertices cannot be that high. However it is ", nr_verts;
    return nr_verts;
}

void Lattice::set_nr_lattice_vertices(const int nr_verts){
    // cudaDeviceSynchronize();
    // int nr_verts[1];
    // cudaMemcpy(nr_verts, m_hash_table.m_filled, sizeof(int), cudaMemcpyDeviceToHost);    
    // return nr_verts[0];

    // torch::Tensor nr_filled_cpu=m_hash_table->m_nr_filled_tensor.clone();
    // nr_filled_cpu.to("cpu");
    // return nr_filled_cpu.item<int>();

    //attempt 2 at making it faster
    m_impl->wait_to_create_vertices(); //we synchronize the event and wait until whatever kernel was launched to create vertices has also finished
    // int nr_verts=0;
    // cudaMemcpy ( (void*)&nr_verts,  m_hash_table->m_nr_filled_tensor.data<int>(), sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( m_hash_table->m_nr_filled_tensor.data<int>(), &nr_verts, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy ( &nr_verts,  m_hash_table->m_impl->m_nr_filled, sizeof(int), cudaMemcpyDeviceToHost );
    // VLOG(1) << "nr of verts is " << m_hash_table->m_nr_filled_tensor;
    CHECK(nr_verts>=0) << "nr vertices cannot be negative. However it is ", nr_verts;
    CHECK(nr_verts<1e+8) << "nr vertices cannot be that high. However it is ", nr_verts;
    // return nr_verts;
}


int Lattice::get_filter_extent(const int neighborhood_size) {
    // return std::pow(neighborhood_size + 1, pos_dim + 1) - std::pow(neighborhood_size, pos_dim + 1);
    // I don't understant the above code lifter from bilateralnn, For neighbourhood of 1 it gives very weird values, a lot larget ahn what I expected

    //from adams thesis which he gives the nr of only neighbood size of 1 
    CHECK(neighborhood_size==1) << "At the moment we only have implemented a filter with a neighbourhood size of 1. I haven't yet written the more general formula for more neighbourshood size";
    CHECK(m_pos_dim!=-1) << "m pos dim is not set. It is -1";

    // if(use_center_vertex){
        return 2*(m_pos_dim+1) + 1; //because we have 2 neighbour for each axis and we have pos_dim+1 axes. Also a +1 for the center vertex
    // }else{
        // return 2*(m_pos_dim+1);
    // }

}

int Lattice::val_dim(){
    return m_val_dim;
}
int Lattice::val_full_dim(){
    return m_val_full_dim;
}
int Lattice::pos_dim(){
    return m_pos_dim;
}
int Lattice::capacity(){
    return m_hash_table_capacity;
}
torch::Tensor Lattice::sigmas_tensor(){
    return m_sigmas_tensor;
}

//setters
void Lattice::set_val_dim(const int val_dim){
    m_val_dim=val_dim;
}
void Lattice::set_val_full_dim(const int val_full_dim){
    m_val_full_dim=val_full_dim;
}
void Lattice::set_sigma(const float sigma){
    int nr_sigmas=m_sigmas_val_and_extent.size();
    CHECK(nr_sigmas==1) << "We are summing we have onyl one sigma. This method is intended to affect only one and not two sigmas independently";

    for(int i=0; i<m_sigmas.size(); i++){
        m_sigmas[i]=sigma;
    }

    m_sigmas_tensor=vec2tensor(m_sigmas);

}


// Tensor Lattice::values(){
//     return m_hash_table->m_values_tensor;
// }
// Tensor Lattice::sliced_values_hom(){
//     return m_sliced_values_hom_tensor;
// }
// Tensor Lattice::lattice_rowified(){
//     return m_lattice_rowified;
// }



