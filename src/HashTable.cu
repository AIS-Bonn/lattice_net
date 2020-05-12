#include "lattice_net/HashTable.cuh"

#include <cuda.h>
// #include "torch/torch.h" 

//my stuff 
#include "lattice_net/kernels/HashTableGPU.cuh"



HashTable::HashTable():
    m_capacity(-1), 
    m_pos_dim(-1),
    m_impl( new HashTableGPU() )
    {
}


void HashTable::init(int capacity, int pos_dim, int val_dim){

    m_capacity=capacity;
    m_pos_dim=pos_dim;
    m_impl=std::make_shared<HashTableGPU>( capacity, pos_dim );

    m_keys_tensor=register_buffer("keys", torch::zeros({capacity, pos_dim}).to(torch::kInt32) ); //TODO should it be short so kInt16 as in the original implementation
    m_values_tensor=register_buffer("values", torch::zeros({capacity, val_dim+1}) );
    m_entries_tensor=register_buffer("entries", torch::zeros({capacity}).to(torch::kInt32) );
    m_nr_filled_tensor=register_buffer("nr_filled", torch::zeros({1}).to(torch::kInt32) );

    m_keys_tensor=m_keys_tensor.to("cuda");
    m_values_tensor=m_values_tensor.to("cuda");
    m_entries_tensor=m_entries_tensor.to("cuda");
    m_nr_filled_tensor=m_nr_filled_tensor.to("cuda");

    clear();
    update_impl();


}

void HashTable::clear(){
    if(is_initialized()){
        m_values_tensor.fill_(0);
        m_keys_tensor.fill_(0);
        m_entries_tensor.fill_(-1);
        m_nr_filled_tensor.fill_(0);
    }
}

bool HashTable::is_initialized(){
    if(m_keys_tensor.defined() ){
        return true;
    }else{
        return false;
    }

}

void HashTable::update_impl(){
    m_impl->m_capacity = m_capacity;
    if(m_keys_tensor.defined()){
        m_impl->m_keys = m_keys_tensor.data_ptr<int>();
    }
    if(m_values_tensor.defined()){
        m_impl->m_values = m_values_tensor.data_ptr<float>();
    }
    if(m_entries_tensor.defined()){
        m_impl->m_entries = m_entries_tensor.data_ptr<int>();
    }
    if(m_nr_filled_tensor.defined()){
        m_impl->m_nr_filled = m_nr_filled_tensor.data_ptr<int>();
    }

    m_impl->m_pos_dim = m_pos_dim;

 }

void HashTable::set_values(const torch::Tensor& new_values){
    m_values_tensor=new_values.contiguous();
    update_impl();
}
 