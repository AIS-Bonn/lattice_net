#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h" //needed for threadIdx and blockDim 
#include <device_functions.h> //for __syncthreads

//adapted from https://github.com/MiguelMonteiro/permutohedral_lattice/blob/master/src/PermutohedralLatticeGPU.cuh
class HashTableGPU { 
public:

    HashTableGPU(){

    }
    HashTableGPU(int capacity, int pos_dim){
        m_capacity=capacity;
        m_pos_dim=pos_dim;
    }

    int m_capacity;
    int* m_keys; // size m_capacity x m_pos_dim  of int (or should it be short as in the original implementation)
    float* m_values; // Size m_capacity x m_val_hom_dim  of float  Stores homgeneous values, hence the m_val_hom_dim
    int* m_entries; // size m_capacity x 1 of int  entries of the matrix for recording where the splatting happened for each point. The hash value h of the key is used to index into this tensor. the result is an index that points into the rows of the values and keys tensor where the corresponding key is stored
    int* m_nr_filled; // 1x1 tensor of int storing the nr of filled cells of the keys and values tensor
    int m_pos_dim;
    // int* m_filled; //it doesnt actually store the number of filled elemnts but rather is more like an upper limit to the number of elements we inserted in the hashtable. tthe way the hashtable works is that when keys are inserted they lock a certain entry. However if they find a already locked entry, they skip and they insert the key somewhere else. This may lead to duplicate keys. M_filled counts also the duplicate keys. One approach would be that during cleanhashtable we do also an atomic min to check what was the minimum entry index, this will actually be our nr of filled elements



    //cuda kernels 
    __device__ unsigned int hash(int *key) {
        unsigned int k = 0;
        for (int i = 0; i < m_pos_dim; i++) {
            k += key[i];
            k = k * 2531011;
        }
        // printf("k is %d \n", k);
        return k;

        // //do as in voxel hashing: arount line 200 in https://github.com/niessner/VoxelHashing/blob/master/DepthSensingCUDA/Source/VoxelUtilHashSDF.h
        // const int p0 = 73856093;
		// const int p1 = 19349669;
        // const int p2 = 83492791;
        // unsigned int res = ((key[0] * p0) ^ (key[1] * p1) ^ (key[2] * p2));
        // return res;
    }

    __device__ int modHash(unsigned int n){
        return(n % m_capacity);
    }

    inline __device__ void acquire( int* e ){
        while ( atomicCAS( e, -1, -2 ) +1 ); //the entires start at empty(-1) and if we succesfully change the value to locked(-2), then the old_value would be -1 and +1 would make it zero, breaking the loop
    }


    __device__ int insert(int *key ) {


        // //Attempt 2 at making a fast hashmap
        // // printf("inserting slot %d!\n",slot);
        // int h = modHash(hash(key));
        // // if(h<0){
        // //     h += m_capacity;
        // // }
        // // printf("h is %d \n ", h);
        // int nr_conflicts=0; //nr of times it tried to insert in a entry location but it was already used.
        // int max_nr_conflicts=300;
        // // int max_nr_conflicts=10;
        // int nr_retries=0; //nr of times we found a locked entry and we retried to query it in the hopes that it will be unlocked soon
        // int max_nr_retries=1000;
        // // while (1 && nr_conflicts< max_nr_conflicts && nr_retries<max_nr_retries) {
        // do{
        //     int *e = m_entries + h;

        //     // If the cell is empty (-1), lock it (-2)
        //     int contents;
        //     #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //         #warning CUDA ARCH IS FINE
        //         contents = atomicCAS(e, -1, -2);
        //     #else 
        //         #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //     #endif

        //     if (contents == -2){
        //         // If it was locked already, exit the thread (we lose the information that this thread was going to insert but it's fine because next frame we will get more data)
        //         //we retry the same hash location to see if it's unlocked now
        //         // return -1;
        //         nr_retries++;
        //         // printf("it was already locked, inserting at %d \n", h);
        //     }else if (contents == -1) {
        //         // printf("succesfully locked it after nr_retries %d and nr_conflicts %d \n", nr_retries, nr_conflicts);
        //         // If it was empty, we successfully locked it. Write our key.
        //         int old_filled=0;
        //         #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //             #warning CUDA ARCH IS FINE
        //             old_filled=atomicAdd( m_nr_filled , 1);
        //         #else 
        //             #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //         #endif
        //         for (int i = 0; i < m_pos_dim; i++) {
        //             m_keys[old_filled * m_pos_dim + i] = key[i];
        //             // __threadfence();
        //         }
        //         // Unlock
        //         #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //             #warning CUDA ARCH IS FINE
        //             atomicExch(e, old_filled);
        //         #else 
        //             #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //         #endif
        //         return h;
        //     } else {
        //         // The cell is unlocked and has a key in it, check if it matches
        //         bool match = true;
        //         for (int i = 0; i < m_pos_dim && match; i++) {
        //             match = (m_keys[contents*m_pos_dim+i] == key[i]);
        //         }
        //         if (match){
        //             return h;
        //         }

        //         // printf("conflict in the hash key when inserting at %d \n", h);
        //         //it doest match so we search for another spot in the hope that it matches 
        //         // increment the bucket with wraparound
        //         nr_conflicts++;
        //         h++; //linear probing
        //         // h+=nr_conflicts*nr_conflicts; //quadratic probing
        //         if (h >= m_capacity){
        //             h = 0;
        //         }

        //         // else{
        //         //     return -1;
        //         // }
        //     }
        // }while (nr_conflicts< max_nr_conflicts && nr_retries<max_nr_retries);


        // // do{
        // //     int *e = m_entries + h;
        // //     old = atomicCAS( e, -1, -2 );
        // //     h++; //linear probing
        // //     if (h >= m_capacity){
        // //         h = 0;
        // //     }
        // // } while (old==-2); //while it's still locked keep probing

        // //if we got out of the loop it mean we failed to insert
        // // printf("failed to insert key when nr filled is %d, nr_retries is %d and nr_conflicts is %d \n", *m_nr_filled, nr_retries, nr_conflicts);
        // return -1;








        //there seems to still be many interstions that fail due to the entry being locked. We try to implement it again following this https://github.com/NVlabs/fermat/blob/master/contrib/cugar/basic/cuda/hash.h
        //we have the same deadlock as in https://devtalk.nvidia.com/default/topic/1037511/cuda-programming-and-performance/problem-of-hash-table-lock-in-cuda/


        // int h = modHash(hash(key));

        // int invalid_key=-1;
        // int locked=-2;
        // int old=invalid_key;
        // int* e = m_entries + h;
        // int nr_loops=0;
        // do        
        // {
        //     e = m_entries + h;
        //     #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //         old = atomicCAS( e, invalid_key, locked );
        //     #endif
        //     h++; //linear probing
        //     // h+=nr_loops*nr_loops; //quadratic probing
        //     if (h >= m_capacity){
        //         h = 0;
        //     }
        //     nr_loops=nr_loops+1;
        //     printf("probing at %d and old was %d nr_loops is %d \n",h, old, nr_loops);
        // } while (old != invalid_key);

        // printf("got out\n");
        // // // assign compacted vertex slots
        // if (old == invalid_key)
        // {
        //     // const uint32 unique_id = atomic_add( count, 1 );
        //     // unique[ unique_id ] = key;
        //     int old_filled=0;
        //     old_filled=atomicAdd( m_nr_filled , 1);
        //     for (int i = 0; i < m_pos_dim; i++) {
        //         m_keys[old_filled * m_pos_dim + i] = key[i];
        //     }
        //     old = atomicCAS( e, invalid_key, locked );

        //     // Unlock
        //     #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //         #warning CUDA ARCH IS FINE
        //         atomicExch(e, old_filled);
        //     #else 
        //         #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //     #endif
        // }
        // // printf("returning h %d \n",h);

        // return h-1;







        // //while the previous thing may be fast, we run into race conditions and some positions do not get inserted at all..
        // int h = modHash(hash(key));
        // int nr_conflicts=0; //nr of times it tried to insert in a entry location but it was already used.
        // while (1) {
        //     int *e = m_entries + h;

        //     // If the cell is empty (-1), lock it (-2)
        //     int contents=-1;
        //     #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //         #warning CUDA ARCH IS FINE
        //         contents = atomicCAS(e, -1, -2);
        //     #else 
        //         #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //     #endif

        //     if (contents == -2){
        //         // If it was locked already, move to the next cell
        //         // return 0;
        //     }else if (contents == -1) {
        //         // If it was empty, we successfully locked it. Write our key.
        //         int old_filled=atomicAdd(m_filled,1);
        //         for (int i = 0; i < m_pos_dim; i++) {
        //             m_keys[old_filled * m_pos_dim + i] = key[i];
        //         }
        //         // Unlock
        //         #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //             #warning CUDA ARCH IS FINE
        //             // atomicExch(e, slot);
        //             // atomicExch(e, h);
        //             atomicExch(e, old_filled);
        //         #else 
        //             #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //         #endif
        //         return h;
        //     } else {
        //         // The cell is unlocked and has a key in it, check if it matches
        //         bool match = true;
        //         for (int i = 0; i < m_pos_dim && match; i++) {
        //             match = (m_keys[contents*m_pos_dim+i] == key[i]);
        //         }
        //         if (match)
        //             return h;
        //     }
        //     // increment the bucket with wraparound
        //     nr_conflicts++;
        //     // h++; //linear probing
        //     h+=nr_conflicts*nr_conflicts; //quadratic probing
        //     if (h >= m_capacity*2)
        //         h = 0;
        // }







        // //attempt 4, inspired by https://github.com/ArchaeaSoftware/cudahandbook/blob/master/memory/spinlockReduction.cu
        // int h = modHash(hash(key));

        // //adquire (do a compare un swap until we get a -1 (empty) or something >0). In other words do while we get a locked one
        // int old_contents=-3;
        // // while(true){
        //     int* e= m_entries + h;
        //     // do{
        //     //     e = m_entries + h;
        //     //     // printf("e is %d\n ",*e);
        //     //     #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //     //         #warning CUDA ARCH IS FINE
        //     //         old_contents = atomicCAS(e, -1, -2);//if it's empty (-1). lock it(-2)
        //     //     #else 
        //     //         #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //     //     #endif
        //     //     // printf("banging at position h %d old_contents is %d \n",h, old_contents);
        //     // }while(old_contents==-2); //keep doing this while the contents are locked by another thread
        //     printf("e is %d\n ",*e);
        //     acquire(e);
        //     printf("adquired it! old contents is %d \n ", old_contents);

        //     //we adquired it

        //     //we have to check if we adquired it because it was empty or because it was already allocated
        //     // int old_filled=0;
        //     if(old_contents==-1){
        //         //it was empty so we just add our key there
        //         // old_filled=atomicAdd( m_nr_filled , 1);

        //         int old_filled=0;
        //         #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //             #warning CUDA ARCH IS FINE
        //             old_filled=atomicAdd( m_nr_filled , 1);
        //         #else 
        //             #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //         #endif

        //         if(old_filled>m_capacity-1){
        //             printf("allocated more vertices than the capacity. old filled is %d \n", old_filled);
        //         }
        //         //fill the keys
        //         for (int i = 0; i < m_pos_dim; i++) {
        //             m_keys[old_filled * m_pos_dim + i] = key[i];
        //         }
        //         //we release it
        //         // atomicExch(e, old_filled);
        //         #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //             #warning CUDA ARCH IS FINE
        //             atomicExch(e, old_filled);
        //         #else 
        //             #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //         #endif
        //         printf("added new key, returning h_used_for_adquiring %d, old_filled is %d \n ", h, old_filled);
        //         return h;
        //     }else if(old_contents>=0){
        //         //we have already a key there so we check it it's the same as ours
        //         bool match = true;
        //         for (int i = 0; i < m_pos_dim && match; i++) {
        //             match = (m_keys[old_contents*m_pos_dim+i] == key[i]);
        //         }
        //         if (match){
        //             printf("found matching key, returning h_used_for_adquiring %d,  old_contents is %d \n ", h,  old_contents);
        //             return h;
        //         }else{
        //             //it's no match so we have to keep searching
        //             // return -1;

        //             h++; //linear probing
        //             if(h>=m_capacity){
        //                 h=0;
        //             }
        //         }
        //     }
    
        //     // __threadfence();
        // // }

        // return -1; //it should reach this







        // //attempt 5 in which you just insert the duplicates and then deal with them later. attemp4 deadlocks because of atomic cas doing a spinlock which is performed by all the threads in the warp
        // int h = modHash(hash(key));
        // while(1){
        //     int *e = m_entries + h;

        //     // If the cell is empty (-1), lock it (-2)
        //     int contents;
        //     #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //         #warning CUDA ARCH IS FINE
        //         contents = atomicCAS(e, -1, -2);
        //     #else 
        //         #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //     #endif

        //     if (contents == -2){
        //         // If it was locked already, we move to the next cell and try to insert there
        //     }else if (contents == -1) {
        //         // printf("succesfully locked it after nr_retries %d and nr_conflicts %d \n", nr_retries, nr_conflicts);
        //         // If it was empty, we successfully locked it. Write our key.
        //         int old_filled=0;
        //         #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //             #warning CUDA ARCH IS FINE
        //             old_filled=atomicAdd( m_nr_filled , 1);
        //         #else 
        //             #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //         #endif
        //         for (int i = 0; i < m_pos_dim; i++) {
        //             m_keys[old_filled * m_pos_dim + i] = key[i];
        //         }
        //         // Unlock
        //         #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
        //             #warning CUDA ARCH IS FINE
        //             atomicExch(e, old_filled);
        //         #else 
        //             #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
        //         #endif
        //         return h;
        //     } else {
        //         // The cell is unlocked and has a key in it, check if it matches
        //         bool match = true;
        //         for (int i = 0; i < m_pos_dim && match; i++) {
        //             match = (m_keys[contents*m_pos_dim+i] == key[i]);
        //         }
        //         if (match){
        //             return h;
        //         }

        //     }
            
        //     h++; //linear probing
        //     // h+=nr_conflicts*nr_conflicts; //quadratic probing
        //     if (h >= m_capacity){
        //         h = 0;
        //     }
        // }





        //attempt 6 following the spinlock of figure 10 in http://www0.cs.ucl.ac.uk/staff/j.alglave/papers/asplos15.pdf
        int h = modHash(hash(key));

        //the original spinlock of figure 10 of  http://www0.cs.ucl.ac.uk/staff/j.alglave/papers/asplos15.pdf
        // bool leaveLoop = false;
        // while(!leaveLoop) {
        //     int lockValue = atomicCAS(lockAddr,0,1);
        //     if(lockValue == 0) {
        //         leaveLoop = true;
        //         __threadfence();
        //         // critical section
        //         __threadfence();
        //         atomicExch(lockAddr, 0);
        //         *lockAddr = 0;
        //     }
        //     __threadfence();
        // }

        // int nr_probes=0;
        while(1){
            //to lock a certain positions
            int *e = m_entries + h;
            bool leaveLoop = false;
            while(!leaveLoop) {
                int contents = atomicCAS(e, -1, -2);;
                if(contents == -1) { //succesfuly locked it 
                    leaveLoop = true;
                    __threadfence();
                    // critical section

                    int old_filled=atomicAdd( m_nr_filled , 1);
                    for (int i = 0; i < m_pos_dim; i++) {
                        m_keys[old_filled * m_pos_dim + i] = key[i];
                    }


                    __threadfence();
                    atomicExch(e, old_filled);
                    return h;
                }else if(contents>=0){ //it has already a key inside, check if it's the same
                    leaveLoop = true; //if we match the key we would return the h, and if we don't then we leave the loop and go and check another position
                    // The cell is unlocked and has a key in it, check if it matches
                    bool match = true;
                    for (int i = 0; i < m_pos_dim && match; i++) {
                        match = (m_keys[contents*m_pos_dim+i] == key[i]);
                    }
                    if (match){
                        return h;
                    }
                }
                __threadfence();
            }

            //we left checking this position and we check the next one
            // nr_probes++;
            h++; //linear probing
            if (h >= m_capacity){
                h = 0;
            }
            // printf("nr of probes is %d\n ",nr_probes);
        }




    }

    __device__ int retrieve(int *key) {

        int h = modHash(hash(key));
        int nr_conflicts=0; //nr of times it tried to insert in a entry location but it was already used.
        int max_nr_conflicts=300;
        while (1 && nr_conflicts < max_nr_conflicts) {
            int *e = m_entries + h;

            if (*e == -1)
                return -1;

            bool match = true;
            for (int i = 0; i < m_pos_dim && match; i++) {
                match = (m_keys[(*e)*m_pos_dim+i] == key[i]);
            }
            if (match)
                return *e;

            nr_conflicts++;
            h++; //linear probing
            // h+=nr_conflicts*nr_conflicts; //quadratic probing
            if (h >= m_capacity)
                h = 0;
        }

        //if we got out of the loop it mean we failed to retreive (we had to many conflicts)
        return -1;

    }


   
};






