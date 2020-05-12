
#pragma once

#include<memory>

#include "torch/torch.h"

// #include "kernels/hello_kernel.cuh"
// #include <cuda.h>

// struct MatrixEntry {
//     int index;
//     float weight;
// };

class HashTableGPU;

//adapted from https://github.com/MiguelMonteiro/permutohedral_lattice/blob/master/src/PermutohedralLatticeGPU.cuh
class HashTable : public torch::nn::Module, public std::enable_shared_from_this<HashTable> { 
public:
    HashTable();
    void init(int capacity, int pos_dim, int val_dim);

    int m_capacity;
    torch::Tensor m_keys_tensor; // size m_capacity x m_pos_dim  of int (or should it be short as in the original implementation)
    torch::Tensor m_values_tensor; // Size m_capacity x m_val_full_dim  of float  Stores homgeneous values, hence the m_val_full_dim
    torch::Tensor m_entries_tensor; // size m_capacity x 1 of int  entries of the matrix for recording where the splatting happened for each point. The hash value h of the key is used to index into this tensor. the result is an index that points into the rows of the values and keys tensor where the corresponding key is stored
    torch::Tensor m_nr_filled_tensor; // 1x1 tensor of int storing the nr of filled cells of the keys and values tensor
    int m_pos_dim;
    
    //pointer to implementation 
    std::shared_ptr<HashTableGPU> m_impl;
   

    void clear();
    bool is_initialized();
    void update_impl();
    void set_values(const torch::Tensor& new_values); //use this to set new values because it will also call update_impl to set the float pointers to the correct place in the implementation
   
};