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
class Lattice : public std::enable_shared_from_this<Lattice>{
// class Lattice : public torch::Tensor, public std::enable_shared_from_this<Lattice>{
// class Lattice :public THPVariable, public std::enable_shared_from_this<Lattice>{
public:
    template <class ...Args>
    static std::shared_ptr<Lattice> create( Args&& ...args ){
        return std::shared_ptr<Lattice>( new Lattice(std::forward<Args>(args)...) );
    }
    ~Lattice();

    void set_sigmas(std::initializer_list<  std::pair<float, int> > sigmas_list); //its nice to call as a whole function which gets a list of std pairs
    void set_sigmas(std::vector<  std::pair<float, int> > sigmas_list); // in the init_params code I need to pass an explicit std vector so in this case I would need this
    torch::Tensor bilateral_filter(torch::Tensor& positions_raw, torch::Tensor& values); //runs a bilateral filter on the positions and values and returns the output values

    void begin_splat(); //clears the hashtable and new_values matris so we can use them as fresh
    // void begin_splat_modify_only_values(); //clears the hashtable and new_values matris so we can use them as fresh
    std::tuple<torch::Tensor, torch::Tensor> splat_standalone(torch::Tensor& positions_raw, torch::Tensor& values); 
    std::tuple<torch::Tensor, torch::Tensor> just_create_verts(torch::Tensor& positions_raw,  const bool return_indices_and_weights );  //creates splatting indices and splatting weights
    std::shared_ptr<Lattice> expand(torch::Tensor& positions_raw, const int point_multiplier, const float noise_stddev, const bool expand_values );
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> distribute(torch::Tensor& positions_raw, torch::Tensor& values); 
    // torch::Tensor create_splatting_mask(const torch::Tensor& nr_points_per_simplex, const int nr_positions, const int max_nr_points);
    // void blur_standalone(torch::Tensor& positions_raw, torch::Tensor& values); 
    // std::shared_ptr<Lattice> blur_standalone(); 
    torch::Tensor slice_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slice_standalone_no_precomputation(torch::Tensor& positions_raw); //slice at the position and don't use the m_matrix, but rather query the simplex and get the barycentric coordinates and all that. This is useful for when we slice at a different position than the one used for splatting
    // std::shared_ptr<Lattice> slice_elevated_verts(const std::shared_ptr<Lattice> lattice_to_slice_from);    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gather_standalone_no_precomputation(torch::Tensor& positions_raw); //gathers the features of the neighbouring vertices and concats them all together, together with the barycentric weights. The output tensor will be size 1 x nr_positions x ( (m_pos_dim+1) x (val_full_dim +1) ). On each row we store sequencially the values of each vertex and then at the end we add the last m_pos_dim+1 barycentric weights
    torch::Tensor gather_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // torch::Tensor gather_elevated_standalone_no_precomputation(const std::shared_ptr<Lattice> lattice_to_gather_from); //gatheres the values from lattice_to_gather_from into the vertices of this current lattice.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  slice_classify_no_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes); //slices a lattices with some deltas applied to the barycentric coordinates and clasifies it in one go. Returns a tensor of class_logits of size 1 x nr_positions x nr_classes
    torch::Tensor slice_classify_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    
    std::shared_ptr<Lattice> convolve_standalone(torch::Tensor& filter_bank); // convolves the lattice with a filter bank, creating a new values matrix. kernel_bank is a of size nr_filters x filter_extent x in_val_dim
    std::shared_ptr<Lattice> convolve_im2row_standalone(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours, const bool flip_neighbours);
    // std::shared_ptr<Lattice> depthwise_convolve(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours, const bool use_center_vertex, const bool flip_neighbours);
    torch::Tensor im2row(std::shared_ptr<Lattice> lattice_neighbours, const int filter_extent, const int dilation, const bool flip_neighbours);

    std::shared_ptr<Lattice> create_coarse_verts();  //creates another lattice which would be the result of splatting the positions/2. The values of the new coarse lattice are set to 0
    std::shared_ptr<Lattice> create_coarse_verts_naive(torch::Tensor& positions_raw); //the previous one causes some positions to end up in empty space for some reason, so instead we use this to create vertices around all the positions, will be slower but possibly more correct

    //backwards passes 
    void slice_backwards_standalone_with_precomputation(torch::Tensor& positions_raw, const torch::Tensor& sliced_values_hom, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    void slice_backwards_standalone_with_precomputation_no_homogeneous(torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    torch::Tensor row2im(const torch::Tensor& lattice_rowified,  const int dilation, const int filter_extent, const int nr_filters, std::shared_ptr<Lattice> lattice_neighbours );
    void slice_backwards_elevated_verts_with_precomputation(const std::shared_ptr<Lattice> lattice_sliced_from, const torch::Tensor& grad_sliced_values, const int nr_verts_to_slice_from, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // torch::Tensor filter_rotate(const torch::Tensor); //backwards convolution means convolving the gradient with a filter rotated by 180 degrees. This is because a certain vertex by having a certan value and through the forward convolution, all its neighbour will use the value of this central vertex. Therefore it will contribute something (weighted by the kernel) to each neighbour during the forward pass. Therefore during the backward pass we accumulate the erros of the neighbours as a convolution, however the weights need to be adjusted so that the weight coming inot the central vertex should the same as if the kernel was centered around the neighbour. This is beucase that would be the weight with which the central vertex contributed during the forward pass. thinking about the weights in this enigbhour centric way is the same as convolving with a filter that is rotated
    void slice_classify_backwards_with_precomputation(const torch::Tensor& grad_class_logits, torch::Tensor& positions_raw, torch::Tensor& initial_values,  torch::Tensor& delta_weights, torch::Tensor&  linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& grad_lattice_values, torch::Tensor& grad_delta_weights,  torch::Tensor& grad_linear_clasify_weight, torch::Tensor& grad_linear_clasify_bias, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    void gather_backwards_standalone_with_precomputation(const torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    void gather_backwards_elevated_standalone_with_precomputation(const std::shared_ptr<Lattice> lattice_gathered_from, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);


    std::shared_ptr<Lattice> clone_lattice();
    Eigen::MatrixXd keys_to_verts();
    Eigen::MatrixXd elevate(torch::Tensor& positions_raw);
    // Eigen::MatrixXd deelevate(const torch::Tensor& keys);
    Eigen::MatrixXd color_no_neighbours();
    // Eigen::MatrixXd create_E_matrix(const int pos_dim);
    void increase_sigmas(const float stepsize);

  



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
    torch::Tensor values();
   

    //setters
    void set_sigma(const float sigma);
    void set_name(const std::string name);
    void set_values(const torch::Tensor& new_values); //use this to set new values because it will also call update_impl to set the float pointers to the correct place in the implementation
    void set_positions( const torch::Tensor& positions_raw ); // the positions that were used to create this lattice



    // std::vector< std::pair<float, int> > m_sigmas_val_and_extent; //for each sigma we store here the value and the number of dimensions it affect. In the Gui we modify this one

    // std::shared_ptr<HashTable> m_hash_table;

    // std::string m_name;

    // int m_lvl; //lvl of coarsenes of the lattice, it starts at 1 for the finest lattice and increases by 1 for each applicaiton for coarsen()

    // torch::Tensor m_positions; //positions that were used originally for creating the lattice
    // torch::Tensor m_sliced_values_hom_tensor; //sliced values in homogeneous coordinates. Size nr_positions x val_dim+1. We need to store the homogeneous coordinate because we need it for backwards pass
    // torch::Tensor m_lattice_rowified; // 2d tensor of size m_hash_table_capacity x filter_extent*(m_val_dim+1) stores for each lattice vertex (allocated or not) the values of all the neighbours in the filter extent (also it's center value)
    // torch::Tensor m_distributed_tensor; //tensor which just gathers all the values that gets distributed by the positions onto each lattice vertex. Has size nr_positions x (m_pos_dim+1) x  ( (m_pos_dim+1) + (m_val_dim+1 ) )  . this is because each positions splats onto m_pos_dim+1 vertices, and for each vertex we store the key-elevated position (which is pos_dim+1) and then the values*weight and then also the homogeneous coordinate
    // torch::Tensor m_distributed_tensor; //tensor which just gathers all the values that gets distributed by the positions onto each lattice vertex. Has size nr_positions x (m_pos_dim+1) x  ( (m_pos_dim + m_val_dim +1)  . this is because each positions splats onto m_pos_dim+1 vertices, and for each vertex we store the position (the one already divided by sigma) and then the values which is val_dim dimensional and then the barycentric coordinate
    // torch::Tensor m_splatting_indices_tensor; // 1 dimensional vector of size (m_nr_positions*(m_pos_dim+1)) of ints. Says for each input position the indices in the hash_table of the m_pos_dim+1 lattice points onto which it splats to
    // torch::Tensor m_splatting_weights_tensor; // 1 dimensional vector of size (m_nr_positions*(m_pos_dim+1)) of floats. Says for each input position the barycentric weights for the m_pos_dim+1 lattice points onto which it splats to
    // torch::Tensor m_gathered_values_tensor; // tensor of size 1 x nr_positions x ( (m_pos_dim+1) x (val_full_dim+1) ). Used as an alternative for slicing in which we just gather all the values and let the network learn how to deal with them

    //for debugging

private:
    Lattice(const std::string config_file);
    Lattice(const std::string config_file, const std::string name);
    Lattice(Lattice* other);
    void init_params(const std::string config_file);
    // void set_and_check_input(torch::Tensor& positions_raw, torch::Tensor& values); //sets pos dim and val dim and then also checks that the positions and values are correct and we have sigmas for all posiitons dims
    void check_positions(const torch::Tensor& positions_raw);
    void check_values(const torch::Tensor& values);
    void check_positions_and_values(const torch::Tensor& positions_raw, const torch::Tensor& values);
    void update_impl();

    std::string m_name;
    int m_lvl; //lvl of coarsenes of the lattice, it starts at 1 for the finest lattice and increases by 1 for each applicaiton for coarsen()

    std::shared_ptr<HashTable> m_hash_table;
    std::shared_ptr<LatticeGPU> m_impl;
    torch::Tensor m_positions; //positions that were used originally for creating the lattice

   
    std::vector<float> m_sigmas;
    torch::Tensor m_sigmas_tensor;
    std::vector< std::pair<float, int> > m_sigmas_val_and_extent; //for each sigma we store here the value and the number of dimensions it affect. In the Gui we modify this one



  
};


