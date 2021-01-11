#pragma once

#include <memory>
#include <stdarg.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


#include "torch/torch.h"

// #include "lattice_net/jitify_helper/jitify_options.hpp" //Needs to be added BEFORE jitify because this defined the include paths so that the kernels cna find each other
// #include "jitify/jitify.hpp"
#include <Eigen/Dense>


class HashTable;

// class Lattice : public torch::autograd::Variable, public std::enable_shared_from_this<Lattice>{
// class Lattice : public at::Tensor, public std::enable_shared_from_this<Lattice>{
class Lattice : public std::enable_shared_from_this<Lattice>{
// class Lattice :public THPVariable, public std::enable_shared_from_this<Lattice>{
public:
    template <class ...Args>
    static std::shared_ptr<Lattice> create( Args&& ...args ){
        return std::shared_ptr<Lattice>( new Lattice(std::forward<Args>(args)...) );
    }
    ~Lattice();

    void set_sigmas(std::initializer_list<  std::pair<float, int> > sigmas_list); //its nice to call as a whole function which gets a list of std pairs
    void set_sigmas(std::vector<  std::pair<float, int> > sigmas_list); // in the init_params code I need to pass an explicit std vector so in this case I would need this
   
    //getters
    int val_dim();
    int pos_dim();
    int capacity();
    std::string name();
    int nr_lattice_vertices(); //cannot obtain from here because we need the cuda part to perform a wait
    int get_filter_extent(const int neighborhood_size); //how many lattice points does a certain filter touch (eg for a pos_dim of 2 and neighbouhood of 1 we touch 7 verts, 6 for the hexagonal shape and 1 for the center)
    torch::Tensor sigmas_tensor();
    torch::Tensor positions(); 
    std::shared_ptr<HashTable> hash_table();
   

    //setters
    void set_sigma(const float sigma);
    void set_name(const std::string name);

private:

    Lattice(const std::string config_file);
    Lattice(const std::string config_file, const std::string name);
    Lattice(Lattice* other);
    void init_params(const std::string config_file);
    void check_input(torch::Tensor& positions_raw, torch::Tensor& values); //sets pos dim and val dim and then also checks that the positions and values are correct and we have sigmas for all posiitons dims


    std::string m_name;
    int m_lvl; //lvl of coarsenes of the lattice, it starts at 1 for the finest lattice and increases by 1 for each applicaiton for coarsen()

    std::shared_ptr<HashTable> m_hash_table;
    torch::Tensor m_positions; //positions that were used originally for creating the lattice
    //for syncronization
    cudaEvent_t m_event_nr_vertices_lattice_changed; //when doing splatting, distribute, or a create_coarse_verts, we must record this even after the kernel. Afterwards when we call nr_lattice_vertices we wait for this event to have finished indicating that the kernel has finished

   
    std::vector<float> m_sigmas;
    torch::Tensor m_sigmas_tensor;
    std::vector< std::pair<float, int> > m_sigmas_val_and_extent; //for each sigma we store here the value and the number of dimensions it affect. In the Gui we modify this one
  
};


