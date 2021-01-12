#include "lattice_net/Lattice.cuh"

//c++
#include <string>

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it
#include "EasyCuda/UtilsCuda.h"
#include "string_utils.h"

//my stuff
#include "lattice_net/HashTable.cuh"
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



using torch::Tensor;
using namespace radu::utils;




//CPU code that calls the kernels
Lattice::Lattice(const std::string config_file):
    m_impl( new LatticeGPU() ),
    m_lvl(1)
    {

    init_params(config_file);
    VLOG(3) << "Creating lattice";

}

Lattice::Lattice(const std::string config_file, const std::string name):
    m_impl( new LatticeGPU() ),
    m_name(name),
    m_lvl(1)
    {

    init_params(config_file);


    VLOG(3) << "Creating lattice: " <<name;

}

Lattice::Lattice(Lattice* other):
    m_impl( new LatticeGPU() ),
    m_lvl(1)
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
        m_hash_table=std::make_shared<HashTable>(other->hash_table()->capacity() );
        // m_hash_table->m_capacity=other->m_hash_table_capacity; 
        // m_hash_table->m_pos_dim=other->m_pos_dim;
        //hashtable tensors shallow copy (just a pointer assignemtn so they use the same data in memory)
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
    m_hash_table=std::make_shared<HashTable>(hash_table_capacity);

    int nr_sigmas=lattice_config["nr_sigmas"]; //nr of is sigma values we have. Each one affecting a different number of dimensions of the positions
    for (int i=0; i < nr_sigmas; i++) {
        std::string param_name="sigma_"+std::to_string(i);
        std::string sigma_val_and_extent = (std::string)lattice_config[param_name];
        std::vector<std::string> tokenized = radu::utils::split(sigma_val_and_extent, " ");
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

void Lattice::check_positions(const torch::Tensor& positions_raw){
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==2) << "positions should have dim 2 correspondin to HW. However it has sizes" << positions_raw.sizes();
    int pos_dim=positions_raw.size(1);
    CHECK(m_sigmas.size()==pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<pos_dim;

}
void Lattice::check_values(const torch::Tensor& values){
    CHECK(values.scalar_type()==at::kFloat) << "values should be of type float";
    CHECK(values.dim()==2) << "values should have dim 2 correspondin to HW. However it has sizes" << values.sizes();
}
void Lattice::check_positions_and_values(const torch::Tensor& positions_raw, const torch::Tensor& values){
    //check input
    CHECK(positions_raw.size(0)==values.size(0)) << "Sizes of positions and values should match. Meaning that that there should be a value for each position. Positions_raw has sizes "<<positions_raw.sizes() << " and the values has size " << values.sizes();
    check_positions(positions_raw);
    check_values(positions_raw);
}



void Lattice::begin_splat(){
    m_hash_table->clear(); 
}


std::tuple<torch::Tensor, torch::Tensor> Lattice::splat_standalone(torch::Tensor& positions_raw, torch::Tensor& values ){
    check_positions_and_values(positions_raw, values);
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    int val_dim=values.size(1);

    m_positions=positions_raw; //raw positions which created this lattice


    //if it's not initialized to the correct values we intialize the hashtable
    if( !m_hash_table->m_keys_tensor.defined() ){
        m_hash_table->init(pos_dim, val_dim);
        m_hash_table->to(torch::kCUDA);
    }

    // if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);


    //to cuda
    TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    values=values.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    TIME_END("upload_cuda");

    TIME_START("scale_by_sigma");
    Tensor positions=positions_raw/m_sigmas_tensor;
    TIME_END("scale_by_sigma");

    TIME_START("splat");
    m_impl->splat_standalone(positions.data_ptr<float>(), values.data_ptr<float>(), nr_positions, pos_dim, val_dim, 
                            splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(),  *(m_hash_table->m_impl) );

    
    TIME_END("splat");

    // VLOG(3) << "after splatting nr_verts is " << nr_lattice_vertices();
    auto ret = std::make_tuple (splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;
  
}


std::tuple<torch::Tensor, torch::Tensor> Lattice::just_create_verts(torch::Tensor& positions_raw ){
    check_positions(positions_raw);
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);

    //if it's not initialized to the correct values we intialize the hashtable
    if( !m_hash_table->m_keys_tensor.defined() ){
        m_hash_table->init(pos_dim, 1 );
        m_hash_table->to(torch::kCUDA);
    }

    //we DO NOT initialize the indices and weights to -1 because we call this function only for creating vertices, and usually in that case we just propagate the indices and weights from the previous lattice.
    // if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
    //     m_splatting_indices_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    //     m_splatting_weights_tensor = torch::zeros({nr_positions*(m_pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    // }
    // m_splatting_indices_tensor.fill_(-1);
    // m_splatting_weights_tensor.fill_(-1);

    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);


    //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    Tensor positions=positions_raw/m_sigmas_tensor;

    m_impl->just_create_verts(positions.data_ptr<float>(), nr_positions, this->pos_dim(), this->val_dim(), 
                            splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );



    // VLOG(3) << "after just_create_verts nr_verts is " << nr_lattice_vertices();

    auto ret = std::make_tuple (splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;
  
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::distribute(torch::Tensor& positions_raw, torch::Tensor& values){
    check_positions_and_values(positions_raw, values);
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    int val_dim=values.size(1);

    m_positions=positions_raw; //raw positions which created this lattice


    
    //if it's not initialized to the correct values we intialize the hashtable
    if(!m_hash_table->m_keys_tensor.defined()){
        m_hash_table->init(pos_dim, val_dim);
        m_hash_table->to(torch::kCUDA);
    }


    Tensor distributed_tensor = torch::zeros({ nr_positions *(pos_dim+1) , pos_dim + val_dim +1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);
   
    
    m_hash_table->clear();


    //to cuda
    positions_raw=positions_raw.to("cuda");
    values=values.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    Tensor positions=positions_raw/m_sigmas_tensor;

    m_impl->distribute(positions.data_ptr<float>(), values.data_ptr<float>(), distributed_tensor.data_ptr<float>(), nr_positions, pos_dim, val_dim, 
                            splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );


    VLOG(3) << "after distributing nr_verts is " << nr_lattice_vertices();

    auto ret = std::make_tuple (distributed_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;
  
}

// Tensor Lattice::create_splatting_mask(const torch::Tensor& nr_points_per_simplex, const int nr_positions, const int max_nr_points){

//     Tensor mask = torch::zeros({nr_positions*(m_pos_dim+1)}, torch::dtype(torch::kBool).device(torch::kCUDA, 0)  );

//     m_impl->create_splatting_mask(mask.data_ptr<bool>(), m_splatting_indices_tensor.data_ptr<int>(), nr_points_per_simplex.data_ptr<int>(), max_nr_points, nr_positions, m_pos_dim); 

//     return mask;
// }




std::shared_ptr<Lattice> Lattice::convolve_im2row_standalone(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours,  const bool flip_neighbours){

    if (!lattice_neighbours){
        lattice_neighbours=shared_from_this();
    }

    CHECK(filter_bank.defined()) << "Filter bank is undefined";
    CHECK(filter_bank.dim()==2) << "Filter bank should have dimension 2, corresponding with (filter_extent * val_dim) x nr_filters.  However it has dimension: " << filter_bank.dim();

    int nr_filters=filter_bank.size(1) ;
    int filter_extent=filter_bank.size(0) / lattice_neighbours->val_dim();
    CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent << "val dim is " << lattice_neighbours->val_dim();

    //this lattice should be coarser (so a higher lvl) or finer(lower lvl) or at least at the same lvl as the lattice neigbhours. But the differnce should be at most 1 level
    CHECK(std::abs(m_lvl-lattice_neighbours->m_lvl)<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level or finer by 1 lvl with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;
    
    // VLOG(4) <<"starting convolved im2row_standlaone. The current lattice has nr_vertices_lattices" << nr_lattice_vertices();
    CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
    int nr_vertices=nr_lattice_vertices();
    int cur_values_size=m_hash_table->m_values_tensor.size(0);
    // CHECK(nr_vertices==cur_values_size) << "the nr of lattice vertices should be the same as the values tensor has rows. However the nr lattice vertices is " << nr_vertices << " and values has nr of rows " << cur_values_size;

    std::shared_ptr<Lattice> convolved_lattice=create(this); //create a lattice with no config but takes the config from this one
    convolved_lattice->m_name="convolved_lattice";


    filter_bank=filter_bank.to("cuda");


    Tensor lattice_rowified=torch::zeros({nr_vertices, filter_extent* lattice_neighbours->val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    
    m_impl->im2row(nr_vertices, this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, flip_neighbours, false);
    


    //multiply each patch with the filter bank
    Tensor convolved= lattice_rowified.mm(filter_bank);
   
    convolved_lattice->m_hash_table->set_values(convolved);
    // convolved_lattice->m_val_dim=nr_filters;
    // convolved_lattice->m_hash_table->update_impl(); //very important
    // convolved_lattice->m_lattice_rowified=m_lattice_rowified;

    // VLOG(4) << "convolved lattice has nr filled " << convolved_lattice->nr_lattice_vertices();
    CHECK(convolved_lattice->nr_lattice_vertices()!=0) << "Why does this convolved lattice has zero nr_filled?";

    return convolved_lattice;

}

// std::shared_ptr<Lattice> Lattice::depthwise_convolve(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours, const bool use_center_vertex_from_lattice_neigbhours, const bool flip_neighbours){


//     CHECK(filter_bank.defined()) << "Filter bank is undefined";
//     CHECK(filter_bank.dim()==2) << "Filter bank should have dimension 2, corresponding with filter_extent x val_dim.  However it has dimension: " << filter_bank.dim();
//     // CHECK(filter_bank.size(0)== 2*(m_pos_dim+1)+1 ) <<"Filter extent should cover nr of vertices corresponding to a 1 hop neighborhood. Bigger neighbourhoods are not yet implemented. That means it should be 2*(m_pos_dim+1)+1 which would be" << 2*(m_pos_dim+1)+1 << "however the filter_bank.size(1) is " << filter_bank.size(1);
//     // CHECK(filter_bank.size(2) == m_val_dim+1) << "Filters should convolve over all the values of this lattice so the m_val_dim+1 which is " << m_val_dim+1 << "which is " << "should be equal to filter_bank.size(2) which is " << filter_bank.size(2);

//     int filter_extent=filter_bank.size(0);
//     // VLOG(1) << "filter_bank sizes is" << filter_bank.sizes();
//     // VLOG(1) << "val full dim is " << m_val_full_dim;
//     CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent;

//     //this lattice should be coarser (so a higher lvl) or at least at the same lvl as the lattice neigbhours (which is a finer lvl therefore the lattice_neigbhours.m_lvl is lower)
//     CHECK(m_lvl-lattice_neighbours->m_lvl<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;
    
//     VLOG(4) <<"starting convolved im2row_standlaone. The current lattice has nr_vertices_lattices" << nr_lattice_vertices();
//     CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
//     int nr_vertices=nr_lattice_vertices();
//     // VLOG(1)

//     std::shared_ptr<Lattice> convolved_lattice=create(this); //create a lattice with no config but takes the config from this one
//     convolved_lattice->m_name="convolved_lattice";
//     int cur_hash_table_size=m_hash_table->m_values_tensor.size(0);
//     // VLOG(1) << "cloning values tensor which has size" << cur_hash_table_size;
//     // VLOG(1) << "cloning values tensor which has size" << cur_hash_table_size;
//     // if(with_homogeneous_coord){
//         // convolved_lattice->m_hash_table->m_values_tensor=torch::zeros({m_hash_table_capacity, nr_filters+1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ) ; // +1 because of homogeneous coordinates
//     // }else{
//         //no need to allocate because it will be directly set to be the whatever comes from the matrix mutliplicaiton between lattice_rowified and filter bank
//     // }
//     // convolved_lattice->m_hash_table->m_values_tensor=convolved_lattice->m_hash_table->m_values_tensor.to("cuda");
//     convolved_lattice->m_hash_table->m_values_tensor=torch::zeros({nr_vertices, m_val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ) ;

//     //m_val_dim and m_val_full_dim are equal now
//     convolved_lattice->m_val_dim=m_val_dim;
//     convolved_lattice->m_hash_table->update_impl(); //updating the hash table pointer to point to the newly clones values tensor

//     //kernel bank is of size nr_filers x filter_extent x in_val_dim
//     filter_bank=filter_bank.to("cuda");

//     //fill im2row TODO precompute it in the lattice

//     // TIME_START("create_lattice_rowified");
//     // VLOG(1) << "checking if lattice rowified has size: nr_vertices" << nr_vertices << " filter_extent " << filter_extent << " m_val_full_dim " << m_val_full_dim;
//     // if( !m_lattice_rowified.defined() || m_lattice_rowified.size(0)!=nr_vertices || m_lattice_rowified.size(1)!=filter_extent*m_val_dim  ){
//     //     // VLOG(1) << "Creating a lattice rowified with size: nr_vertices" << nr_vertices << " filter_extent " << filter_extent << " m_val_full_dim " << m_val_full_dim;
//     //     m_lattice_rowified=torch::zeros({nr_vertices, filter_extent*m_val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     // }else{
//     //     m_lattice_rowified.fill_(0);
//     // }
//     // TIME_END("create_lattice_rowified");

//     // TIME_START("convolve_im2row");
//     // TIME_START("im2row");
//     bool debug_kernel=false;
//     // if(m_lvl==2){
//     //     // debug_kernel=true;
//     // }

//     // VLOG(1) << "calling im2row with lattice neighbours which have vlaues of norm " << lattice_neighbours->m_hash_table->m_values_tensor.norm();
//     // VLOG(4) <<"calling im2row with m_val_full_dim of " << m_val_full_dim;
//     // TIME_START("depthwise_convolve")
//     m_impl->depthwise_convolve(nr_vertices, m_pos_dim, m_val_dim, filter_bank.data_ptr<float>(), dilation, convolved_lattice->m_hash_table->m_values_tensor.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, use_center_vertex_from_lattice_neigbhours, flip_neighbours, debug_kernel);
//     // TIME_END("depthwise_convolve")

//     // m_impl->test_row2im(m_hash_table_capacity, m_pos_dim, m_val_full_dim, dilation, m_lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, use_center_vertex);
//     // TIME_END("im2row");

//     // VLOG(1) <<"lattice rowified is \n" << m_lattice_rowified;
//     // Tensor lattice_rowified_unsqueezed=m_lattice_rowified.unsqueeze(0);
//     // EigenMatrixXfRowMajor lattice_rowified_eigen=tensor2eigen(lattice_rowified_unsqueezed);
//     // VLOG(1) <<"lattice rowified is \n" << lattice_rowified_eigen;

//     // VLOG(1) << "im2row should have at least some non zero value pero row. The rowsise sum of lattice_rowified is " << m_lattice_rowified.sum(1);


//     //multiply each patch with the filter bank
//     // Tensor convolved= m_lattice_rowified.mm(filter_bank);
//     // VLOG(1) << "finished multiplication";
//     // VLOG(1) << "current values has shape" << m_hash_table->m_values_tensor.sizes();
//     // VLOG(1) << "convolved_hash_table.values has shape" << convolved_lattice->m_hash_table->m_values_tensor.sizes();
//     // VLOG(1) << "convolved has shape" << convolved.sizes();

//     // convolved_lattice->m_hash_table->m_values_tensor=convolved;
//     convolved_lattice->m_hash_table->update_impl(); //very important

//     // TIME_END("convolve_im2row");

//     VLOG(4) << "convolved lattice has nr filled " << convolved_lattice->nr_lattice_vertices();
//     CHECK(convolved_lattice->nr_lattice_vertices()!=0) << "Why does this convolved lattice has zero nr_filled?";

//     // VLOG(1) << "this lattice has lattice rowified of norm " <<m_lattice_rowified.norm();

//     //FOR DEBUG assign the lattice rowified also the the convolve lattice so that we can query it afterwards and debug why there are vertices that don't have any neighbours
//     // convolved_lattice->m_lattice_rowified=m_lattice_rowified.clone(); //IMPORTANT at the moment. Do not comment out
//     convolved_lattice->m_lattice_rowified=m_lattice_rowified;

//     return convolved_lattice;

// }

torch::Tensor Lattice::im2row(std::shared_ptr<Lattice> lattice_neighbours, const int filter_extent, const int dilation, const bool flip_neighbours){

    if (!lattice_neighbours){
        lattice_neighbours=shared_from_this();
    }

    CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent;


    
    //this lattice should be coarser (so a higher lvl) or finer(lower lvl) or at least at the same lvl as the lattice neigbhours. But the differnce should be at most 1 level
    CHECK(std::abs(m_lvl-lattice_neighbours->m_lvl)<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level or finer by 1 lvl with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;

    // VLOG(3) <<"starting convolved im2row_standlaone. The current lattice has nr_vertices_lattices" << nr_lattice_vertices();
    CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
    int nr_vertices=nr_lattice_vertices();
    int cur_values_size=m_hash_table->m_values_tensor.size(0);
    // CHECK(nr_vertices==cur_values_size) << "the nr of lattice vertices should be the same as the values tensor has rows. However the nr lattice vertices is " << nr_vertices << " and values has nr of rows " << cur_values_size;


    // TIME_START("create_lattice_rowified");
    // if( !m_lattice_rowified.defined() || m_lattice_rowified.size(0)!=nr_vertices || m_lattice_rowified.size(1)!=filter_extent*m_val_dim  ){
    Tensor lattice_rowified=torch::zeros({nr_vertices, filter_extent* lattice_neighbours->val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    // }else{
        // m_lattice_rowified.fill_(0);
    // }


    m_impl->im2row(nr_vertices, this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, flip_neighbours, false);

    return lattice_rowified;

}

torch::Tensor Lattice::row2im(const torch::Tensor& lattice_rowified,  const int dilation, const int filter_extent, const int nr_filters, std::shared_ptr<Lattice> lattice_neighbours){

    if (!lattice_neighbours){
        lattice_neighbours=shared_from_this();
    }

    int nr_vertices=nr_lattice_vertices();
    m_hash_table->m_values_tensor=torch::zeros({nr_vertices, nr_filters}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); 
    m_hash_table->update_impl();

    CHECK(nr_lattice_vertices()!=0) <<"Something went wrong because have zero lattice vertices";


    // m_val_dim=nr_filters;

    m_impl->row2im(m_hash_table->capacity(), this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, /*do_test*/false);

    return m_hash_table->m_values_tensor;
}


std::shared_ptr<Lattice> Lattice::create_coarse_verts(){

    int capacity=m_hash_table->capacity();
    int val_dim=m_hash_table->val_dim();
    int pos_dim=m_hash_table->pos_dim();

    std::shared_ptr<Lattice> coarse_lattice=create(this); //create a lattice with no config but takes the config from this one
    coarse_lattice->m_name="coarse_lattice";
    coarse_lattice->m_lvl=m_lvl+1;
    coarse_lattice->m_sigmas_tensor=m_sigmas_tensor.clone()*2.0; //the sigma for the coarser one is double. This is done so if we slice at this lattice we scale the positions with the correct sigma
    for(size_t i=0; i<m_sigmas.size(); i++){
        coarse_lattice->m_sigmas[i]=m_sigmas[i]*2.0;
    } 
    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1, val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
    coarse_lattice->m_hash_table->m_keys_tensor=torch::zeros({capacity, pos_dim}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->m_entries_tensor=torch::zeros({capacity}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) ) ;
    coarse_lattice->m_hash_table->m_nr_filled_tensor=torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->clear();
    coarse_lattice->m_hash_table->update_impl();

    TIME_START("coarsen");
    m_impl->coarsen(capacity, pos_dim, *(m_hash_table->m_impl), *(coarse_lattice->m_hash_table->m_impl)  );
    TIME_END("coarsen");

    int nr_vertices=coarse_lattice->nr_lattice_vertices();
    VLOG(3) << "after coarsening nr_verts of the coarse lattice is " << nr_vertices;

    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({nr_vertices, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  ); //we create exactly the values required for he vertices that were allocated
    coarse_lattice->m_hash_table->update_impl();

    return coarse_lattice;

}


std::shared_ptr<Lattice> Lattice::create_coarse_verts_naive(torch::Tensor& positions_raw){

    int capacity=m_hash_table->capacity();
    int val_dim=m_hash_table->val_dim();
    int pos_dim=m_hash_table->pos_dim();


    std::shared_ptr<Lattice> coarse_lattice=create(this); //create a lattice with no config but takes the config from this one
    coarse_lattice->m_name="coarse_lattice";
    coarse_lattice->m_lvl=m_lvl+1;
    coarse_lattice->m_sigmas_tensor=m_sigmas_tensor.clone()*2.0; //the sigma for the coarser one is double. This is done so if we slice at this lattice we scale the positions with the correct sigma
    coarse_lattice->m_sigmas=m_sigmas;
    for(size_t i=0; i<m_sigmas.size(); i++){
        coarse_lattice->m_sigmas[i]=m_sigmas[i]*2.0;
    } 
    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
    coarse_lattice->m_hash_table->m_keys_tensor=torch::zeros({capacity, pos_dim}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->m_entries_tensor=torch::zeros({capacity}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) ) ;
    coarse_lattice->m_hash_table->m_nr_filled_tensor=torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->clear();
    coarse_lattice->m_hash_table->update_impl();


    coarse_lattice->begin_splat();
    coarse_lattice->m_hash_table->update_impl();

    coarse_lattice->just_create_verts(positions_raw);


    return coarse_lattice;

}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::slice_standalone_no_precomputation(torch::Tensor& positions_raw){

    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


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
    Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    //recalculate the splatting indices and weight for the backward pass of the slice
    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);

    m_hash_table->update_impl();


    TIME_START("slice");
    m_impl->slice_standalone_no_precomputation( positions.data_ptr<float>(), sliced_values_hom_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
    TIME_END("slice");


    auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;

    // return sliced_values_hom_tensor.clone()


}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::gather_standalone_no_precomputation(torch::Tensor& positions_raw){

    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


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
    int row_size_gathered=(pos_dim+1)*(val_dim()+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
    Tensor gathered_values_tensor=torch::zeros({nr_positions, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    //recalculate the splatting indices and weight for the backward pass of the gather
    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    TIME_START("gather");
    m_impl->gather_standalone_no_precomputation( positions.data_ptr<float>(), gathered_values_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
    TIME_END("gather");

    auto ret = std::make_tuple (gathered_values_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;
    // return gathered_values_tensor;

}


torch::Tensor Lattice::gather_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


     //to cuda
    // TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    // TIME_END("upload_cuda");

    // TIME_START("scale_by_sigma");
    VLOG(3) << "gather standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;
    // TIME_END("scale_by_sigma")

    //initialize the output values to zero 
    int row_size_gathered=(this->pos_dim()+1)*(this->val_dim()+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
    Tensor gathered_values_tensor=torch::zeros({nr_positions, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    //assume we have already splatting weight and indices
    if( !splatting_indices_tensor.defined() || !splatting_weights_tensor.defined()  || splatting_indices_tensor.size(0)!=nr_positions*(this->pos_dim()+1) ||  splatting_weights_tensor.size(0)!=nr_positions*(this->pos_dim()+1)  ){
        LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(this->pos_dim()+1) << "but indices has size " << splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << splatting_weights_tensor.sizes();
    }
    m_hash_table->update_impl();


    // TIME_START("gather");
    m_impl->gather_standalone_with_precomputation( positions.data_ptr<float>(), gathered_values_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
    // TIME_END("gather");

    return gathered_values_tensor;

}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  Lattice::slice_classify_no_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes){


    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


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
    Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, nr_classes}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


    //recalculate the splatting indices and weight for the backward pass of the slice
    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    TIME_START("slice_classify");
    m_impl->slice_classify_no_precomputation( positions.data_ptr<float>(), 
                                              sliced_values_hom_tensor.data_ptr<float>(), 
                                              delta_weights.data_ptr<float>(), 
                                              linear_clasify_weight.data_ptr<float>(), 
                                              linear_clasify_bias.data_ptr<float>(), 
                                              nr_classes,
                                              this->pos_dim(), 
                                              this->val_dim(),  
                                              nr_positions, 
                                              splatting_indices_tensor.data_ptr<int>(), 
                                              splatting_weights_tensor.data_ptr<float>(), 
                                              *(m_hash_table->m_impl) );
    TIME_END("slice_classify");

    auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;
    // return sliced_values_hom_tensor;

}


torch::Tensor Lattice::slice_classify_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


     //to cuda
    // TIME_START("upload_cuda");
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    delta_weights=delta_weights.to("cuda");
    linear_clasify_weight=linear_clasify_weight.to("cuda");
    linear_clasify_bias=linear_clasify_bias.to("cuda");
    // TIME_END("upload_cuda");

    // TIME_START("scale_by_sigma");
    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;
    // TIME_END("scale_by_sigma")

    //we store here the class logits directly
    Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, nr_classes}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


    //assume we have already splatting weight and indices
    if( !splatting_indices_tensor.defined() || !splatting_weights_tensor.defined()  || splatting_indices_tensor.size(0)!=nr_positions*(this->pos_dim()+1) ||  splatting_weights_tensor.size(0)!=nr_positions*(this->pos_dim()+1)  ){
        LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(this->pos_dim()+1) << "but indices has size " << splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << splatting_weights_tensor.sizes();
    }
    m_hash_table->update_impl();


    // TIME_START("slice_classify_cuda");
    m_impl->slice_classify_with_precomputation( positions.data_ptr<float>(), 
                                              sliced_values_hom_tensor.data_ptr<float>(), 
                                              delta_weights.data_ptr<float>(), 
                                              linear_clasify_weight.data_ptr<float>(), 
                                              linear_clasify_bias.data_ptr<float>(), 
                                              nr_classes,
                                              this->pos_dim(), 
                                              this->val_dim(),  
                                              nr_positions, 
                                              splatting_indices_tensor.data_ptr<int>(), 
                                              splatting_weights_tensor.data_ptr<float>(), 
                                              *(m_hash_table->m_impl) );
    // TIME_END("slice_classify_cuda");

    return sliced_values_hom_tensor;

}





void Lattice::slice_backwards_standalone_with_precomputation(torch::Tensor& positions_raw, const torch::Tensor& sliced_values_hom, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";




    TIME_START("slice_back");
    m_impl->slice_backwards_standalone_with_precomputation( sliced_values_hom.data_ptr<float>(), grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions, *(m_hash_table->m_impl) );
    TIME_END("slice_back");

}


void Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous(torch::Tensor& positions_raw, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values.dim()==2) <<"grad_sliced_values should be nr_positions x m_val_dim, so it should have 2 dimensions. However it has "<< grad_sliced_values.dim();

    m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), grad_sliced_values.size(1)},  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    m_hash_table->update_impl();



    TIME_START("slice_back");
    m_impl->slice_backwards_standalone_with_precomputation_no_homogeneous(grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions, *(m_hash_table->m_impl) );
    TIME_END("slice_back");

}


void Lattice::slice_classify_backwards_with_precomputation(const torch::Tensor& grad_class_logits, torch::Tensor& positions_raw, torch::Tensor& initial_values, torch::Tensor& delta_weights, torch::Tensor&  linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& grad_lattice_values, torch::Tensor& grad_delta_weights, torch::Tensor& grad_linear_clasify_weight, torch::Tensor& grad_linear_clasify_bias, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_class_logits.dim()==2) <<"grad_class_logits should be  nr_positions x nr_classes, so it should have 2 dimensions. However it has "<< grad_class_logits.dim();
    // m_val_dim=initial_values.size(1);


    // TIME_START("slice_clasify_back");
    m_impl->slice_classify_backwards_with_precomputation(grad_class_logits.data_ptr<float>(), initial_values.data_ptr<float>(),  splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions,
    delta_weights.data_ptr<float>(), linear_clasify_weight.data_ptr<float>(), linear_clasify_bias.data_ptr<float>(), nr_classes, grad_lattice_values.data_ptr<float>(), grad_delta_weights.data_ptr<float>(), grad_linear_clasify_weight.data_ptr<float>(),grad_linear_clasify_bias.data_ptr<float>(),
     *(m_hash_table->m_impl) );
    // TIME_END("slice_clasify_back");

}

void Lattice::gather_backwards_standalone_with_precomputation(const torch::Tensor& positions_raw, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw); 
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    int val_dim=grad_sliced_values.size(1)/(pos_dim+1)-1; //we will acumulate the gradient into the value tensor. And it should have the same val_dim as the values that were in the lattice_we_gathered from
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values.dim()==2) <<"grad_sliced_values should be nr_positions x ((m_val_dim+1)*(m_pos_dim+1)), so it should have 2 dimensions. However it has "<< grad_sliced_values.dim();


    m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    m_hash_table->update_impl();



    // TIME_START("gather_back");
    m_impl->gather_backwards_standalone_with_precomputation(grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(),  this->val_dim(), nr_positions, *(m_hash_table->m_impl) );
    // TIME_END("gather_back");


}



std::shared_ptr<Lattice> Lattice::clone_lattice(){
    std::shared_ptr<Lattice> new_lattice=create(this); //create a lattice with no config but takes the config from this one
    return new_lattice;
}

// //retuns the keys of the lattice as vertices. We cannot retunr a mesh because nvcc complains about compiling the MeshCore with betterenum
// Eigen::MatrixXd Lattice::keys_to_verts(){
//     CHECK(m_pos_dim==2) << "In order to show the keys as a mesh the pos_dim has to be 2 because only then the keys will be in 3D space and not in something bigger";

//     Tensor keys=m_hash_table->m_keys_tensor.clone();
//     keys=keys.unsqueeze(0);
//     keys=keys.to(at::kFloat);
//     EigenMatrixXfRowMajor keys_eigen_2D=tensor2eigen(keys);
//     CHECK(keys_eigen_2D.cols()==2) << "The keys should be 2D keys because storing the full 3D one would be redundant as the key digits sum up to zero";


//     //those keys only store the 2 dimensional part, we need to recreate the full m_pos_dim+1 key
//     Eigen::MatrixXd V; 
//     V.resize(keys_eigen_2D.rows(), 3);
//     Eigen::VectorXf summed = keys_eigen_2D.rowwise().sum();
//     for (int i=0; i < keys_eigen_2D.rows(); i++) {
//         V(i,0)=keys_eigen_2D(i,0);
//         V(i,1)=keys_eigen_2D(i,1);
//         V(i,2)=-summed(i);
//     }

//     return V;
// }

// Eigen::MatrixXd Lattice::elevate(torch::Tensor& positions_raw){

//     int nr_positions=positions_raw.size(0);
//     m_pos_dim=positions_raw.size(1);

//     //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma");

//     Tensor elevated=torch::zeros({nr_positions,m_pos_dim+1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     elevated.fill_(0);

//     TIME_START("elevate");
//     m_impl->elevate(positions.data_ptr<float>(),  m_pos_dim, nr_positions, elevated.data_ptr<float>());
//     TIME_END("elevate");

//     elevated=elevated.unsqueeze(0);
//     EigenMatrixXfRowMajor elevated_eigen_rowmajor=tensor2eigen(elevated);
//     Eigen::MatrixXd elevated_eigen;
//     elevated_eigen=elevated_eigen_rowmajor.cast<double>();
//     return elevated_eigen;
// }

// Eigen::MatrixXd Lattice::deelevate(const torch::Tensor& keys){

//     //get keys as eigen matrix
//     Tensor keys_valid=keys.slice(0, 0, nr_lattice_vertices()).clone();
//     EigenMatrixXfRowMajor keys_eigen_row_major=tensor2eigen(keys_valid.to(at::kFloat).unsqueeze(0));
//     Eigen::MatrixXd keys_eigen(keys_eigen_row_major.rows(), m_pos_dim+1); //reconstruct the full key
//     for (int i=0; i < keys_eigen.rows(); i++) {
//         float sum=0;
//         for (int j=0; j < m_pos_dim; j++) {
//             keys_eigen(i,j)=keys_eigen_row_major(i,j);
//             sum+=keys_eigen_row_major(i,j);
//         }
//         keys_eigen(i,m_pos_dim)=sum;
//     }


//     //create E matrix 
//     Eigen::MatrixXd E=create_E_matrix(m_pos_dim);
//     //inverse it 
//     Eigen::MatrixXd E_inv=E.completeOrthogonalDecomposition().pseudoInverse();
//     //multiply by inverse
//     //scale by inv stddev 
//     float invStdDev = (m_pos_dim + 1) * sqrt(2.0f / 3);
//     //scale my sigmas
//     Eigen::MatrixXd deelevated_vertices(keys_eigen_row_major.rows(),3);
//     for (int i=0; i < keys_eigen_row_major.rows(); i++) {
//         Eigen::VectorXd key=keys_eigen.row(i);
//         Eigen::VectorXd vertex_deelevated= (E_inv*key).array()*invStdDev;
//         for (int j=0; j < m_sigmas.size(); j++) {
//             vertex_deelevated(j)=vertex_deelevated(j)*m_sigmas[j];
//         }
//         deelevated_vertices.row(i)=vertex_deelevated;
//     }


//     return deelevated_vertices;    
// }

// Eigen::MatrixXd Lattice::color_no_neighbours(){
//     CHECK(m_lattice_rowified.size(0)==nr_lattice_vertices()) << "the lattice rowified should have rows for each vertex lattice. However we have a lattice rowified of size " << m_lattice_rowified.sizes() << " and nr of vertices is " << nr_lattice_vertices();

//     VLOG(1) << "color_no_neihbours: Lattice rowified has size" << m_lattice_rowified.sizes();
//     VLOG(1) << "color_no_neihbours: nr_lattice_vertices is " << nr_lattice_vertices();
//     EigenMatrixXfRowMajor rowified_row_major=tensor2eigen(m_lattice_rowified.to(at::kFloat).unsqueeze(0));
//     Eigen::MatrixXd C(nr_lattice_vertices(), 3);
//     C.setZero();
//     for (int i=0; i < rowified_row_major.rows(); i++) {
//         float sum=0;
//         for (int j=0; j < rowified_row_major.cols(); j++) {
//             sum+=rowified_row_major(i,j);
//         }
//         if(sum==0){
//             VLOG(1) << "setting row to red at idx " << i;
//             C.row(i) << 1.0, 0.0, 0.0;
//         }
//     }

//     return C;
// }
// Eigen::MatrixXd Lattice::create_E_matrix(const int pos_dim){

//     //page 30 of Andrew Adams thesis
//     Eigen::MatrixXf E_left(pos_dim+1, pos_dim );
//     Eigen::MatrixXf E_right(pos_dim, pos_dim );
//     E_left.setZero();
//     E_right.setZero();
//     //E left is has at the bottom a square matrix which has an upper triangular part of ones. Afterwards the whole E_left gets appended another row on top of all ones
//     E_left.bottomRows(pos_dim).triangularView<Eigen::Upper>().setOnes();
//     //the diagonal of the bottom square is linearly incresing from [-1, -m_pos_dim]
//     E_left.bottomRows(pos_dim).diagonal().setLinSpaced(pos_dim,1,pos_dim);
//     E_left.bottomRows(pos_dim).diagonal()= -E_left.bottomRows(pos_dim).diagonal();
//     //E_left has the first row all set to ones
//     E_left.row(0).setOnes();
//     // VLOG(1) << "E left is \n" << E_left;
//     //E right is just a diagonal matrix with entried in the diag set to 1/sqrt((d+1)(d+2)). Take into account that the d in the paper starts at 1 and we start at 0 so we add a +1 to diag_idx
//     for(int diag_idx=0; diag_idx<pos_dim; diag_idx++){
//         E_right(diag_idx, diag_idx) =  1.0 / (sqrt((diag_idx + 1) * (diag_idx + 2))) ;
//     }
//     // VLOG(1) << "E right is \n" << E_right;

//     //rotate into H_d
//     Eigen::MatrixXf E = E_left*E_right;

//     return E.cast<double>();
// }

void Lattice::increase_sigmas(const float stepsize){
        // m_sigmas.clear();
    for(size_t i=0; i<m_sigmas.size(); i++){
        m_sigmas[i]+=stepsize;
    }

    m_sigmas_tensor=vec2tensor(m_sigmas);

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
  
    m_impl->wait_to_create_vertices(); //we synchronize the event and wait until whatever kernel was launched to create vertices has also finished
    // cudaEventSynchronize(m_event_nr_vertices_lattice_changed);  //we synchronize the event and wait until whatever kernel was launched to create vertices has also finished
    int nr_verts=0;
    cudaMemcpy ( &nr_verts,  m_hash_table->m_nr_filled_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost );
    CHECK(nr_verts>=0) << "nr vertices cannot be negative. However it is " << nr_verts;
    CHECK(nr_verts<1e+8) << "nr vertices cannot be that high. However it is " << nr_verts;
    return nr_verts;
}
int Lattice::get_filter_extent(const int neighborhood_size) {
    CHECK(neighborhood_size==1) << "At the moment we only have implemented a filter with a neighbourhood size of 1. I haven't yet written the more general formula for more neighbourshood size";
    CHECK(this->pos_dim()!=-1) << "m pos dim is not set. It is -1";

    return 2*(this->pos_dim()+1) + 1; //because we have 2 neighbour for each axis and we have pos_dim+1 axes. Also a +1 for the center vertex
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
torch::Tensor Lattice::values(){
    return  m_hash_table->m_values_tensor;
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
void Lattice::set_values(const torch::Tensor& new_values){
    // m_values_tensor=new_values.contiguous();
    // update_impl();
    m_hash_table->set_values(new_values);
}
 


