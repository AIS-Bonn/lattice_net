
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
    // HashTable(int capacity, int pos_dim, int val_dim);
    void init(int capacity, int pos_dim, int val_dim);

    int m_capacity;
    torch::Tensor m_keys_tensor; // size m_capacity x m_pos_dim  of int (or should it be short as in the original implementation)
    torch::Tensor m_values_tensor; // Size m_capacity x m_val_full_dim  of float  Stores homgeneous values, hence the m_val_full_dim
    torch::Tensor m_entries_tensor; // size m_capacity x 1 of int  entries of the matrix for recording where the splatting happened for each point. The hash value h of the key is used to index into this tensor. the result is an index that points into the rows of the values and keys tensor where the corresponding key is stored
    torch::Tensor m_nr_filled_tensor; // 1x1 tensor of int storing the nr of filled cells of the keys and values tensor
    int m_pos_dim;
    // int* m_filled; //it doesnt actually store the number of filled elemnts but rather is more like an upper limit to the number of elements we inserted in the hashtable. tthe way the hashtable works is that when keys are inserted they lock a certain entry. However if they find a already locked entry, they skip and they insert the key somewhere else. This may lead to duplicate keys. M_filled counts also the duplicate keys. One approach would be that during cleanhashtable we do also an atomic min to check what was the minimum entry index, this will actually be our nr of filled elements
    
    //pointer to implementation 
    std::shared_ptr<HashTableGPU> m_impl;
    // HashTableGPU* m_impl;

    // int* keys(); //returns pointer to the data stored in the m_keys_tensor
    // float* values(); //retuns a pointer to the data stored in the m_values_tensor
    // int* entires(); //pointer to the data in m_entries_tensor
    // int* nr_filled_ptr(); //pointer to the one element stored in m_filled_tensor

    void clear();
    bool is_initialized();
    void update_impl();
    void set_values(torch::Tensor new_values); //use this to set new values because it will also call update_impl to set the float pointers to the correct place in the implementation
    // void to_cuda(); // inhereting from torch::nn:Module doesn't really work because nvrt tries to include this file and it doesnt have knowledge about it. it be nice because then we would have access to register_buffer which will alow to use .to("cuda") on the whole object

    // //cuda kernels 
    // __device__ int modHash(unsigned int n);
    // __device__ unsigned int hash(short *key);
    // __device__ int retrieve(short *key);
    // // __device__ int insert(short *key, unsigned int slot);
    // __device__ int insert(short *key);

   
};