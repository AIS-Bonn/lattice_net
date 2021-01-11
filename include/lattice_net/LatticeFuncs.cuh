#pragma once

#include <memory>
#include <stdarg.h>

#include <cuda.h>


#include "torch/torch.h"

#include "lattice_net/jitify_helper/jitify_options.hpp" //Needs to be added BEFORE jitify because this defined the include paths so that the kernels cna find each other
#include "jitify/jitify.hpp"
#include <Eigen/Dense>


class LatticeGPU;
class HashTable;

// class Lattice : public torch::autograd::Variable, public std::enable_shared_from_this<Lattice>{
// class Lattice : public at::Tensor, public std::enable_shared_from_this<Lattice>{
class LatticeFuncs : public std::enable_shared_from_this<LatticeFuncs>{
// class Lattice :public THPVariable, public std::enable_shared_from_this<Lattice>{
public:
    template <class ...Args>
    static std::shared_ptr<LatticeFuncs> create( Args&& ...args ){
        return std::shared_ptr<LatticeFuncs>( new LatticeFuncs(std::forward<Args>(args)...) );
    }
    ~LatticeFuncs();


    // void begin_splat(); //clears the hashtable and new_values matris so we can use them as fresh
    // void splat_standalone(torch::Tensor& positions_raw, torch::Tensor& values); 
    // void just_create_verts(torch::Tensor& positions_raw ); 
    // void distribute(torch::Tensor& positions_raw, torch::Tensor& values); 
    // torch::Tensor create_splatting_mask(const torch::Tensor& nr_points_per_simplex, const int nr_positions, const int max_nr_points);
    // std::shared_ptr<Lattice> blur_standalone(); 
    // torch::Tensor slice_standalone_no_precomputation(torch::Tensor& positions_raw); //slice at the position and don't use the m_matrix, but rather query the simplex and get the barycentric coordinates and all that. This is useful for when we slice at a different position than the one used for splatting
    // std::shared_ptr<Lattice> slice_elevated_verts(const std::shared_ptr<Lattice> lattice_to_slice_from);
    // torch::Tensor gather_standalone_no_precomputation(torch::Tensor& positions_raw); //gathers the features of the neighbouring vertices and concats them all together, together with the barycentric weights. The output tensor will be size 1 x nr_positions x ( (m_pos_dim+1) x (val_full_dim +1) ). On each row we store sequencially the values of each vertex and then at the end we add the last m_pos_dim+1 barycentric weights
    // torch::Tensor gather_standalone_with_precomputation(torch::Tensor& positions_raw);
    // torch::Tensor gather_elevated_standalone_no_precomputation(const std::shared_ptr<Lattice> lattice_to_gather_from); //gatheres the values from lattice_to_gather_from into the vertices of this current lattice.
    // torch::Tensor slice_classify_no_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes); //slices a lattices with some deltas applied to the barycentric coordinates and clasifies it in one go. Returns a tensor of class_logits of size 1 x nr_positions x nr_classes
    // torch::Tensor slice_classify_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes);
    
    // std::shared_ptr<Lattice> convolve_standalone(torch::Tensor& filter_bank); // convolves the lattice with a filter bank, creating a new values matrix. kernel_bank is a of size nr_filters x filter_extent x in_val_dim
    // std::shared_ptr<Lattice> convolve_im2row_standalone(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours, const bool use_center_vertex, const bool flip_neighbours);
    // std::shared_ptr<Lattice> depthwise_convolve(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours, const bool use_center_vertex, const bool flip_neighbours);
    // torch::Tensor im2row(std::shared_ptr<Lattice> lattice_neighbours, const int filter_extent, const int dilation, const bool use_center_vertex_from_lattice_neigbhours, const bool flip_neighbours);

    // std::shared_ptr<Lattice> create_coarse_verts();  //creates another lattice which would be the result of splatting the positions/2. The values of the new coarse lattice are set to 0
    // std::shared_ptr<Lattice> create_coarse_verts_naive(torch::Tensor& positions_raw); //the previous one causes some positions to end up in empty space for some reason, so instead we use this to create vertices around all the positions, will be slower but possibly more correct

    // //backwards passes 
    // void slice_backwards_standalone_with_precomputation(torch::Tensor& positions_raw, const torch::Tensor& sliced_values_hom, const torch::Tensor& grad_sliced_values);
    // void slice_backwards_standalone_with_precomputation_no_homogeneous(torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values);
    // torch::Tensor row2im(const torch::Tensor& lattice_rowified,  const int dilation, const int filter_extent, const int nr_filters, std::shared_ptr<Lattice> lattice_neighbours, const bool use_center_vertex_from_lattice_neighbours, const bool do_test);
    // void slice_backwards_elevated_verts_with_precomputation(const std::shared_ptr<Lattice> lattice_sliced_from, const torch::Tensor& grad_sliced_values, const int nr_verts_to_slice_from);
    // void slice_classify_backwards_with_precomputation(const torch::Tensor& grad_class_logits, torch::Tensor& positions_raw, torch::Tensor& initial_values,  torch::Tensor& delta_weights, torch::Tensor&  linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& grad_lattice_values, torch::Tensor& grad_delta_weights,  torch::Tensor& grad_linear_clasify_weight, torch::Tensor& grad_linear_clasify_bias);
    // void gather_backwards_standalone_with_precomputation(const torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values);
    // void gather_backwards_elevated_standalone_with_precomputation(const std::shared_ptr<Lattice> lattice_gathered_from, const torch::Tensor& grad_sliced_values);


    // std::shared_ptr<Lattice> clone_lattice();
    // Eigen::MatrixXd keys_to_verts();
    // Eigen::MatrixXd elevate(torch::Tensor& positions_raw);
    // Eigen::MatrixXd color_no_neighbours();
    // void increase_sigmas(const float stepsize);

    // //getters
    // int val_dim();
    // int val_full_dim();
    // int pos_dim();
    // int get_filter_extent(const int neighborhood_size); //how many lattice points does a certain filter touch (eg for a pos_dim of 2 and neighbouhood of 1 we touch 7 verts, 6 for the hexagonal shape and 1 for the center)
    // int nr_lattice_vertices();
    // int capacity();
    // torch::Tensor sigmas_tensor();

    // //setters
    // void set_val_dim(const int);
    // void set_sigma(const float);
    // void set_nr_lattice_vertices(const int nr_verts);


    // std::vector< std::pair<float, int> > m_sigmas_val_and_extent; //for each sigma we store here the value and the number of dimensions it affect. In the Gui we modify this one

    // std::shared_ptr<HashTable> m_hash_table;

    // std::string m_name;

    // int m_lvl; //lvl of coarsenes of the lattice, it starts at 1 for the finest lattice and increases by 1 for each applicaiton for coarsen()

    // torch::Tensor m_positions; //positions that were used originally for creating the lattice
   

private:
    LatticeFuncs();
    // Lattice(const std::string config_file, const std::string name);
    // Lattice(Lattice* other);
    // void init_params(const std::string config_file);
    // void set_and_check_input(torch::Tensor& positions_raw, torch::Tensor& values); //sets pos dim and val dim and then also checks that the positions and values are correct and we have sigmas for all posiitons dims
    // void update_impl();

   
    std::shared_ptr<LatticeGPU> m_impl;


    
    // std::vector<float> m_sigmas;
    // torch::Tensor m_sigmas_tensor;
  
};

