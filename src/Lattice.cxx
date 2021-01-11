#include "lattice_net/Lattice.h"

//c++
#include <string>

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it
#include "EasyCuda/UtilsCuda.h"
#include "string_utils.h"

//my stuff
#include "lattice_net/HashTable.cuh"
// #include "lattice_net/kernels/LatticeGPU.cuh"

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



using torch::Tensor;
using namespace radu::utils;




//CPU code that calls the kernels
Lattice::Lattice(const std::string config_file):
    // m_impl( new LatticeGPU() ),
    // m_pos_dim(-1),
    // m_val_dim(-1),
    m_lvl(1)
    {

    init_params(config_file);
    VLOG(3) << "Creating lattice";

}

Lattice::Lattice(const std::string config_file, const std::string name):
    // m_impl( new LatticeGPU() ),
    // m_pos_dim(-1),
    // m_val_dim(-1),
    m_name(name),
    m_lvl(1)
    {

    init_params(config_file);


    VLOG(3) << "Creating lattice: " <<name;

}

Lattice::Lattice(Lattice* other)
    // m_impl( new LatticeGPU() ),
    // m_pos_dim(-1),
    // m_val_dim(-1),
    // m_lvl(1)
    {
        m_lvl=other->m_lvl;
        // m_pos_dim=other->m_pos_dim;
        // m_val_dim=other->m_val_dim;
        // m_hash_table_capacity=other->m_hash_table_capacity;
        m_sigmas=other->m_sigmas;
        m_sigmas_tensor=other->m_sigmas_tensor.clone(); //deep copy
        // m_splatting_indices_tensor=other->m_splatting_indices_tensor; //shallow copy
        // m_splatting_weights_tensor=other->m_splatting_weights_tensor; //shallow copy
        // m_lattice_rowified=other->m_lattice_rowified; //shallow copy
        m_positions=other->m_positions; //shallow copy
        //hashtable
        // m_hash_table->m_capacity=other->m_hash_table_capacity; 
        // m_hash_table->m_pos_dim=other->m_pos_dim;
        //hashtable tensors shallow copy (just a pointer assignemtn so they use the same data in memory)
        m_hash_table=std::make_shared<HashTable> (other->hash_table()->m_capacity);
        m_hash_table->m_keys_tensor=other->m_hash_table->m_keys_tensor;
        m_hash_table->m_values_tensor=other->m_hash_table->m_values_tensor;
        m_hash_table->m_entries_tensor=other->m_hash_table->m_entries_tensor;
        m_hash_table->m_nr_filled_tensor=other->m_hash_table->m_nr_filled_tensor.clone(); //deep copy for this one as the new lattice may have different number of vertices
        m_hash_table->update_impl();

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
    int hash_table_capacity = lattice_config["hash_table_capacity"];
    m_hash_table=std::make_shared<HashTable> (hash_table_capacity );

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

void Lattice::check_input(torch::Tensor& positions_raw, torch::Tensor& values){
    //check input
    CHECK(positions_raw.size(0)==values.size(0)) << "Sizes of positions and values should match. Meaning that that there should be a value for each position. Positions_raw has sizes "<<positions_raw.sizes() << " and the values has size " << values.sizes();
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==2) << "positions should have dim 2 correspondin to HW. However it has sizes" << positions_raw.sizes();
    CHECK(values.scalar_type()==at::kFloat) << "values should be of type float";
    CHECK(values.dim()==2) << "values should have dim 2 correspondin to HW. However it has sizes" << values.sizes();
    //set position and check that the sigmas were set correctly
    // m_pos_dim=positions_raw.size(1);
    // m_val_dim=values.size(1);
    // CHECK(m_sigmas.size()==m_pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<m_pos_dim;
}




//getters
int Lattice::val_dim(){
    return m_hash_table->val_dim();
}
int Lattice::pos_dim(){
    return m_hash_table->pos_dim();
}
int Lattice::capacity(){
    return m_hash_table->capacity();
}
std::string Lattice::name(){
    return m_name;
}
int Lattice::nr_lattice_vertices(){
  
    // m_impl->wait_to_create_vertices(); //we synchronize the event and wait until whatever kernel was launched to create vertices has also finished
    cudaEventSynchronize(m_event_nr_vertices_lattice_changed);  //we synchronize the event and wait until whatever kernel was launched to create vertices has also finished
    int nr_verts=0;
    cudaMemcpy ( &nr_verts,  m_hash_table->m_nr_filled_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost );
    CHECK(nr_verts>=0) << "nr vertices cannot be negative. However it is " << nr_verts;
    CHECK(nr_verts<1e+8) << "nr vertices cannot be that high. However it is " << nr_verts;
    return nr_verts;
}
int Lattice::get_filter_extent(const int neighborhood_size) {
    CHECK(neighborhood_size==1) << "At the moment we only have implemented a filter with a neighbourhood size of 1. I haven't yet written the more general formula for more neighbourshood size";
    CHECK(pos_dim()!=-1) << "m pos dim is not set. It is -1";

    return 2*(pos_dim()+1) + 1; //because we have 2 neighbour for each axis and we have pos_dim+1 axes. Also a +1 for the center vertex
}
torch::Tensor Lattice::sigmas_tensor(){
    return m_sigmas_tensor;
}
torch::Tensor Lattice::positions(){
    return m_positions;
}
std::shared_ptr<HashTable> Lattice::hash_table(){
    return m_hash_table;
}



//setters
void Lattice::set_sigma(const float sigma){
    int nr_sigmas=m_sigmas_val_and_extent.size();
    CHECK(nr_sigmas==1) << "We are summing we have onyl one sigma. This method is intended to affect only one and not two sigmas independently";

    for(size_t i=0; i<m_sigmas.size(); i++){
        m_sigmas[i]=sigma;
    }

    m_sigmas_tensor=vec2tensor(m_sigmas);
}
void Lattice::set_name(const std::string name){
    m_name=name;
}



