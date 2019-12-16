#pragma once

#include <cstring>
#include <memory>
#include "surfel_renderer/utils/Profiler.h" 

#include <Eigen/Core>
#include <Eigen/Eigenvalues> 
#include<Eigen/StdVector>

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

// #define HASH_TABLE_INIT_CAPACITY 32768
// #define HASH_TABLE_INIT_CAPACITY 262144
#define HASH_TABLE_INIT_CAPACITY 3048576

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXfRowMajor;
typedef Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXsRowMajor;


// struct CovMatrix3x3
// {
//     int _n=0;
//     Eigen::Vector3f _oldMean, _newMean,
//                     _oldVarianceSum, _newVarianceSum,
//                     _oldCovarianceSum, _newCovarianceSum;

//     void push(Eigen::Vector3f x)
//     {
//         _n++;
//         if (_n == 1)
//         {
//             _oldMean = _newMean = x;
//             _oldVarianceSum = new Eigen::Vector3f(0, 0, 0);
//             _oldCovarianceSum = new Eigen::Vector3f(0, 0, 0);
//         }
//         else
//         {
//             //_newM = _oldM + (x - _oldM) / _n;
//             _newM = _oldM + (x - _oldM).array() / _n;

//             // _newMean = new Eigen::Vector3f(
//             //     _oldMean.X + (x.X - _oldMean.X) / _n,
//             //     _oldMean.Y + (x.Y - _oldMean.Y) / _n,
//             //     _oldMean.Z + (x.Z - _oldMean.Z) / _n);

//             //_newS = _oldS + (x - _oldM) * (x - _newM);
//             _newS = _oldS + (x - _oldM) * (x - _newM);
//             // _newVarianceSum = new Vector(
//             //     _oldVarianceSum.X + (x.X - _oldMean.X) * (x.X - _newMean.X),
//             //     _oldVarianceSum.Y + (x.Y - _oldMean.Y) * (x.Y - _newMean.Y),
//             //     _oldVarianceSum.Z + (x.Z - _oldMean.Z) * (x.Z - _newMean.Z));

//             /* .X is X vs Y
//              * .Y is Y vs Z
//              * .Z is Z vs X
//              */
//             _newCovarianceSum = new Vector(
//                 _oldCovarianceSum.X + (x.X - _oldMean.X) * (x.Y - _newMean.Y),
//                 _oldCovarianceSum.Y + (x.Y - _oldMean.Y) * (x.Z - _newMean.Z),
//                 _oldCovarianceSum.Z + (x.Z - _oldMean.Z) * (x.X - _newMean.X));

//             // set up for next iteration
//             _oldMean = _newMean;
//             _oldVarianceSum = _newVarianceSum;
//         }
//     }
//     public int NumDataValues()
//     {
//         return _n;
//     }

//     public Vector Mean()
//     {
//         return (_n > 0) ? _newMean : new Vector(0, 0, 0);
//     }

//     public Vector Variance()
//     {
//         return _n <= 1 ? new Vector(0, 0, 0) : _newVarianceSum.DivideBy(_n - 1);
//     }
// }


struct CovMatrix3x3{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CovMatrix3x3(){
        m_mean.setZero();
        m_covariance.setZero();
        m_nr_samples=0;
        m_is_dirty=false;
        m_evecs.setZero();
        m_evals.setZero();
        m_look_dir.setZero();
    }

    void push(const Eigen::Vector3f& point, const float weight=1.0){
        if(m_nr_samples==0){
            m_mean=point;
        }else{
            Eigen::Vector3f diff = point - m_mean;
            diff*=weight;
            m_mean += diff / (m_nr_samples + 1);
            m_covariance += diff * diff.transpose() * m_nr_samples / (m_nr_samples + 1);
        } 
        m_nr_samples++;
        m_is_dirty=true;
    }

    //return the covariance matrix until this point
    Eigen::Matrix3f cov(){
        if(m_nr_samples==0){
            return m_covariance;
        }else{
            Eigen::Matrix3f cov_scaled;
            cov_scaled=m_covariance;
            cov_scaled.array()*=1.0/m_nr_samples;
            return cov_scaled;
        }
    }

    Eigen::Vector3f mean(){
        return m_mean;
    }
    
    //return the principal component of the covariance as columns of a 3x3 matirx, theu are ordered in increasing order of their eigenvalues
    Eigen::MatrixXf eigenvecs(){
        if(m_is_dirty){
            update_evecs_and_evals();
            m_is_dirty=false;
        }
        return m_evecs;
    }

    //return the principal component of the covariance as columns of a 3x3 matirx, theu are ordered in increasing order of their eigenvalues
    Eigen::VectorXf eigenvals(){
        if(m_is_dirty){
            update_evecs_and_evals();
            m_is_dirty=false;
        }
        return m_evals;
    }

    void update_evecs_and_evals(){
        Eigen::Matrix3f cov=this->cov();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
        m_evecs = eig.eigenvectors();
        m_evals = eig.eigenvalues();
    }

    int nr_samples(){
        return m_nr_samples;
    }

    void set_look_dir(const Eigen::Vector3f look_dir){
        m_look_dir=look_dir;
        m_look_dir.normalize();
    }

    Eigen::Vector3f normal(){
        if(m_is_dirty){
            update_evecs_and_evals();
            m_is_dirty=false;
        }
        //normal is the smallest eigenvalue
        Eigen::Vector3f normal=m_evecs.col(0).normalized();
        //if should point towards look dir 
        if(!m_look_dir.isZero()){
            float dot = normal.dot(m_look_dir);
            if(dot<0){
                normal=-normal;
            }
        }
        return normal;
    }



private:
    Eigen::Vector3f m_mean;
    Eigen::Matrix3f m_covariance;
    unsigned int m_nr_samples;
    bool m_is_dirty; //when you push a vector, this gets set to true to indicate that the eigenvector and eigenvalues need to be recalculated
    Eigen::MatrixXf m_evecs;
    Eigen::VectorXf m_evals;
    Eigen::Vector3f m_look_dir; // the direction in which the normal should point towards, so we can flip it if necesary 


};



/***************************************************************/
/* Hash table implementation for permutohedral lattice
 *
 * The lattice points are stored sparsely using a hash table.
 * The key for each point is its spatial location in the (m_pos_dim+1)-
 * dimensional space.
 */
/***************************************************************/
class HashTableCPU {
public:
     /* Constructor
     *  pd_: the dimensionality of the position vectors on the hyperplane.
     *  vd_: the dimensionality of the value vectors
     */
    HashTableCPU(int pos_dim, int val_dim_hom) : m_pos_dim(pos_dim), m_val_dim_hom(val_dim_hom) {
        capacity = HASH_TABLE_INIT_CAPACITY;
        filled = 0;
        entries = new int[HASH_TABLE_INIT_CAPACITY];
        memset(entries, -1, HASH_TABLE_INIT_CAPACITY*sizeof(int));
        // keys = new short[m_pos_dim * HASH_TABLE_INIT_CAPACITY / 2];
        // values = new float[m_val_dim_hom * HASH_TABLE_INIT_CAPACITY / 2]{0};
        keys.resize(HASH_TABLE_INIT_CAPACITY, pos_dim);
        values.resize(HASH_TABLE_INIT_CAPACITY, val_dim_hom);
        values.setZero();
        m_cov_matrices.resize(HASH_TABLE_INIT_CAPACITY);
        // m_carving_counter.resize(HASH_TABLE_INIT_CAPACITY); //here we will splat also the nr_time_penetrated which indicated how many times has a point penetrated the mesh around this certain lattice vertex. When a penetration does not occur we splat a value of 0 here but we still increase the homogeneous coordinate for this vertex
        // m_carving_counter.setZero();
    }

    ~HashTableCPU(){
        delete[](entries);
        // delete[](keys);
        // delete[](values);
    }

    void clear(){
        filled=0;
        values.block(0,0,filled,m_val_dim_hom).setZero();
        keys.block(0,0,filled, m_pos_dim).setZero();
    }


    /* Hash function used in this implementation. A simple base conversion. */
    size_t hash(const short *key) {
        size_t k = 0;
        for (int i = 0; i < m_pos_dim; i++) {
            k += key[i];
            k *= 2531011;
        }
        return k;
    }

    // /* Returns the index into the hash table for a given key.
    // *     key: a pointer to the position vector.
    // *       h: hash of the position vector.
    // *  create: a flag specifying whether an entry should be created,
    // *          should an entry with the given key not found.
    // */
    // int lookupOffset(const short *key, size_t h, bool create = true) {

    //     // Double hash table size if necessary
    //     if (filled >= (capacity / 2) - 1) { grow(); }

    //     // Find the entry with the given key
    //     int nr_increments=0;
    //     while (true) {
    //         int* e = entries + h;
    //         // check if the cell is empty
    //         if (*e == -1) {
    //             if (!create)
    //                 return -1; // Return not found.
    //             // need to create an entry. Store the given key.
    //             for (int i = 0; i < m_pos_dim; i++)
    //                 keys[filled * m_pos_dim + i] = key[i];
    //             *e = static_cast<int>(filled);
    //             filled++;
    //             return *e * m_val_dim_hom;
    //         }

    //         // check if the cell has a matching key
    //         bool match = true;
    //         for (int i = 0; i < m_pos_dim && match; i++)
    //             match = keys[*e*m_pos_dim + i] == key[i];
    //         if (match){
    //             if(nr_increments!=0){
    //                 // std::cout << "nr_increments" <<nr_increments << '\n';
    //             }
    //             return *e * m_val_dim_hom;
    //         }

    //         // increment the bucket with wraparound
    //         // std::cout << "increment bucket with wraparound" << '\n';
    //         nr_increments++;
    //         h++;
    //         if (h == capacity)
    //             h = 0;
    //     }
    // }


    int lookup_idx(short *key) {
        size_t h = hash(key) % capacity;

        // Find the entry with the given key
        while (true) {
            int* e = entries + h;
            // check if the cell is empty
            if (*e == -1) {
                    return -1; // Return not found.
            }

            // check if the cell has a matching key
            bool match = true;
            for (int i = 0; i < m_pos_dim && match; i++){
                // match = keys[*e*m_pos_dim + i] == key[i];
                match = keys(*e,i) == key[i];
            }
            if (match){
                return *e;
            }

            // increment the bucket with wraparound
            // std::cout << "increment bucket with wraparound" << '\n';
            h++;
            if (h == capacity)
                h = 0;
        }


        // int offset = lookupOffset(k, h, create);
        // if (offset < 0)
        //     return nullptr;
        // else
        //     return values + offset;
    }  



public:

    // Returns the number of vectors stored.
    int size() { return filled; }
    int get_capacity() { return capacity; }

    // Returns a pointer to the keys array.
    short *getKeys() { return keys.data(); }

    // Returns a pointer to the values array.
    float *getValues() { return values.data(); }

    void set_new_vals(EigenMatrixXfRowMajor& new_vals){
        // for(int i=0; i<capacity; i++){
            // values[i]=new_vals[i];
        // }
        values=new_vals;
        // std::swap(values, new_vals);
    }

    /* Looks up the value vector associated with a given key vector.
     *        k : pointer to the key vector to be looked up.
     *   create : true if a non-existing key should be created.
     */
    float *lookup(short *key, bool create = true) {
        size_t h = hash(key) % capacity;

        // Double hash table size if necessary
        if (filled >= (capacity / 2) - 1) { grow(); }

        // Find the entry with the given key
        int nr_increments=0;
        while (true) {
            int* e = entries + h;
            // check if the cell is empty
            if (*e == -1) {
                if (!create)
                    return nullptr; // Return not found.
                // need to create an entry. Store the given key.
                for (int i = 0; i < m_pos_dim; i++)
                    // keys[filled * m_pos_dim + i] = key[i];
                    keys(filled,i) = key[i];
                *e = static_cast<int>(filled);
                filled++;
                int offset=*e * m_val_dim_hom;
                // return &values[offset];
                return values.row(*e).data();
            }

            // check if the cell has a matching key
            bool match = true;
            for (int i = 0; i < m_pos_dim && match; i++){
                // match = keys[*e*m_pos_dim + i] == key[i];
                match = keys(*e,i) == key[i];
            }
            if (match){
                if(nr_increments!=0){
                    // std::cout << "nr_increments" <<nr_increments << '\n';
                }
                int offset=*e * m_val_dim_hom;
                // Eigen::VectorXf val= values.row(*e);
                // return &values[offset];
                return values.row(*e).data();
                // return *e * m_val_dim_hom;
            }

            // increment the bucket with wraparound
            // std::cout << "increment bucket with wraparound" << '\n';
            nr_increments++;
            h++;
            if (h == capacity)
                h = 0;
        }


        // int offset = lookupOffset(k, h, create);
        // if (offset < 0)
        //     return nullptr;
        // else
        //     return values + offset;
    }  

    /* Grows the size of the hash table */
    void grow() {
        printf("Resizing hash table\n");
        // TIME_SCOPE("resize_hashtable");

        // size_t oldCapacity = capacity;
        // capacity *= 2;

        // // Migrate the value vectors.
        // // auto newValues = new float[m_val_dim_hom * capacity / 2]{0};
        // // std::memcpy(newValues, values, sizeof(float) * m_val_dim_hom * filled);
        // // delete[] values;
        // // values = newValues;
        // EigenMatrixXfRowMajor newValues;
        // newValues.resize(capacity,m_val_dim_hom);
        // newValues.block(0,0,filled,m_val_dim_hom)=values;
        // values=newValues;

        // // Migrate the key vectors.
        // // auto newKeys = new short[m_pos_dim * capacity / 2];
        // // std::memcpy(newKeys, keys, sizeof(short) * m_pos_dim* filled);
        // // delete[] keys;
        // // keys = newKeys;
        // // EigenMatrixXfRowMajor values;
        // EigenMatrixXsRowMajor newKeys;
        // newKeys.resize(capacity,m_pos_dim);
        // newKeys.block(0,0,filled,m_pos_dim)=keys;
        // keys=newKeys;



        // auto newEntries = new int[capacity];
        // memset(newEntries, -1, capacity*sizeof(int));

        // // Migrate the table of indices.
        // for (size_t i = 0; i < oldCapacity; i++) {
        //     if (entries[i] == -1)
        //         continue;
        //     // size_t h = hash(keys + entries[i] * m_pos_dim) % capacity;
        //     size_t h = hash(keys.row(entries[i]).data() ) % capacity;
        //     while (newEntries[h] != -1) {
        //         h++;
        //         if (h == capacity) h = 0;
        //     }
        //     newEntries[h] = entries[i];
        // }
        // delete[] entries;
        // entries = newEntries;
    }

public:    

    // short *keys;
    // float *values;
    EigenMatrixXsRowMajor keys;
    EigenMatrixXfRowMajor values;
    std::vector<CovMatrix3x3, Eigen::aligned_allocator<Eigen::Vector4f> > m_cov_matrices;
    // Eigen::MatrixXi m_carving_counter;
    int *entries;
    size_t capacity, filled;
    int m_pos_dim, m_val_dim_hom;

};


class PermutohedralLatticeCPU_IMPL {

public:
    int m_pos_dim, m_val_dim_hom, N;
    std::unique_ptr<float[]> scaleFactor;
    HashTableCPU hashTable;
    std::unique_ptr<float[]> elevated;
    std::unique_ptr<float[]> rem0;
    std::unique_ptr<short[]> rank;
    std::unique_ptr<float[]> barycentric;
    // std::unique_ptr<short[]> key;
    std::unique_ptr<float[]> val;
    Eigen::VectorXi m_closest_vertex_idxs; // stores for each splatted position the idx to the closest vertex on the lattice
    std::vector<float> m_sigmas; //for each dimension of the positions stores the sigma which we will divide
    EigenMatrixXfRowMajor m_positions;
    EigenMatrixXfRowMajor m_values; 
    int m_spacial_dim; //number of spacial dimensions, set as the first pair in the set_sigmas
    std::vector<bool> m_lattice_vertices_modified; // stored which vertices of the lattice were modified in the last step
    

    // slicing is done by replaying splatting (ie storing the sparse matrix)
    struct MatrixEntry {
        int offset; //idx * vd
        float weight;
    };
    std::unique_ptr<MatrixEntry[]> matrix;
    int idx;

    std::unique_ptr<float[]> compute_scale_factor() {
        auto scaleFactor = std::unique_ptr<float[]>(new float[m_pos_dim]);

        /* We presume that the user would like to do a Gaussian blur of standard deviation
         * 1 in each dimension (or a total variance of pd, summed over dimensions.)
         * Because the total variance of the blur performed by this algorithm is not pd,
         * we must scale the space to offset this.
         *
         * The total variance of the algorithm is (See pg.6 and 10 of paper):
         *  [variance of splatting] + [variance of blurring] + [variance of splatting]
         *   = pd(pd+1)(pd+1)/12 + pd(pd+1)(pd+1)/2 + pd(pd+1)(pd+1)/12
         *   = 2d(pd+1)(pd+1)/3.
         *
         * So we need to scale the space by (pd+1)sqrt(2/3).
         */
        float invStdDev = (m_pos_dim + 1) * sqrt(2.0 / 3);

        // Compute parts of the rotation matrix E. (See pg.4-5 of paper.)
        for (int i = 0; i < m_pos_dim; i++) {
            // the diagonal entries for normalization
            scaleFactor[i] = 1.0 / (sqrt((i + 1) * (i + 2))) * invStdDev;
        }
        return scaleFactor;
    }


    // void embed_position_vector(const float *position) {
    //     // embed position vector into the hyperplane
    //     // first rotate position into the (pd+1)-dimensional hyperplane
    //     // sm contains the sum of 1..n of our feature vector
    //     float sm = 0;
    //     for (int i = m_pos_dim; i > 0; i--) {
    //         float cf = position[i - 1] * scaleFactor[i - 1];
    //         elevated[i] = sm - i * cf;
    //         sm += cf;
    //     }
    //     elevated[0] = sm;
    // }

    void find_enclosing_simplex(){
        // Find the closest 0-colored simplex through rounding
        // greedily search for the closest zero-colored lattice point
        short sum = 0;
        for (int i = 0; i <= m_pos_dim; i++) {
            float v = elevated[i] * (1.0 / (m_pos_dim + 1));
            float up = ceil(v) * (m_pos_dim + 1);
            float down = floor(v) * (m_pos_dim + 1);
            if (up - elevated[i] < elevated[i] - down) {
                rem0[i] = (short) up;
            } else {
                rem0[i] = (short) down;
            }
            sum += rem0[i];
        }
        sum /= m_pos_dim + 1;

        // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
        for (int i = 0; i <= m_pos_dim; i++)
            rank[i] = 0;
        for (int i = 0; i < m_pos_dim; i++) {
            float di = elevated[i] - rem0[i];
            for (int j = i + 1; j <= m_pos_dim; j++)
                if (di < elevated[j] - rem0[j])
                    rank[i]++;
                else
                    rank[j]++;
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i <= m_pos_dim; i++) {
            rank[i] += sum;
            if (rank[i] < 0) {
                rank[i] += m_pos_dim + 1;
                rem0[i] += m_pos_dim + 1;
            } else if (rank[i] > m_pos_dim) {
                rank[i] -= m_pos_dim + 1;
                rem0[i] -= m_pos_dim + 1;
            }
        }
    }

    void compute_barycentric_coordinates() {
        for(int i = 0; i < m_pos_dim + 2; i++)
            barycentric[i]=0;
        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for (int i = 0; i <= m_pos_dim; i++) {
            float delta = (elevated[i] - rem0[i]) *  (1.0 / (m_pos_dim + 1));
            barycentric[m_pos_dim - rank[i]] += delta;
            barycentric[m_pos_dim - rank[i] + 1] -= delta;
        }
        // Wrap around
        barycentric[0] += 1.0 + barycentric[m_pos_dim + 1];
    }

    // void splat_point(const float *position, const float * value) {

    //     // embed_position_vector(position);
    //     // embed position vector into the hyperplane
    //     // first rotate position into the (pd+1)-dimensional hyperplane
    //     // sm contains the sum of 1..n of our feature vector
    //     float sm = 0;
    //     for (int i = m_pos_dim; i > 0; i--) {
    //         float cf = position[i - 1] * scaleFactor[i - 1];
    //         elevated[i] = sm - i * cf;
    //         sm += cf;
    //     }
    //     elevated[0] = sm;

    //     find_enclosing_simplex();

    //     compute_barycentric_coordinates();

    //     auto key = new short[m_pos_dim];
    //     for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
    //         // Compute the location of the lattice point explicitly (all but
    //         // the last coordinate - it's redundant because they sum to zero)
    //         for (int i = 0; i < m_pos_dim; i++) {
    //             key[i] = static_cast<short>(rem0[i] + remainder);
    //             if (rank[i] > m_pos_dim - remainder)
    //                 key[i] -= (m_pos_dim + 1);
    //         }

    //         // Retrieve pointer to the value at this vertex.
    //         // TIME_START("create latice points");
    //         float *val = hashTable.lookup(key, true);
    //         // TIME_PAUSE("create latice points");
    //         // Accumulate values with barycentric weight.
    //         for (int i = 0; i < m_val_dim_hom ; i++)
    //             val[i] += barycentric[remainder] * value[i];

    //         //val[vd - 1] += barycentric[remainder]; //homogeneous coordinate (as if value[vd-1]=1)

    //         // Record this interaction to use later when slicing
    //         matrix[idx].offset = val - hashTable.getValues();
    //         matrix[idx].weight = barycentric[remainder];
    //         idx++;
    //     }
    //     delete[] key;
    // }

    // void splat(const EigenMatrixXfRowMajor& positions, const EigenMatrixXfRowMajor& values){

    //     // EigenMatrixXfRowMajor values_hom;
    //     // values_hom.resize(values.rows(),m_val_dim_hom);
    //     // values_hom.block(0,0,values.rows(),m_val_dim_hom-1)=values;
    //     // values_hom.block(0,m_val_dim_hom-1,values.rows(),1).setOnes();


    //     for (int n = 0; n < positions.rows(); n++) {
    //         // std::cout << "splatting position" << n << "\m"; 

    //         // splat_point(&(positions[n*m_pos_dim]), &(values[n*(m_val_dim_hom)]));
    //         // Eigen::VectorXf pos = positions.row(n);
    //         // Eigen::VectorXf val = values.row(n);
    //         // splat_point(pos.data(), val.data() );

    //         // embed position vector into the hyperplane
    //         // first rotate position into the (pd+1)-dimensional hyperplane
    //         // sm contains the sum of 1..n of our feature vector
    //         float sm = 0;
    //         for (int i = m_pos_dim; i > 0; i--) {
    //             float cf = positions(n,i - 1) * scaleFactor[i - 1];
    //             elevated[i] = sm - i * cf;
    //             sm += cf;
    //         }
    //         elevated[0] = sm;

    //         find_enclosing_simplex();

    //         compute_barycentric_coordinates();

    //         auto key = new short[m_pos_dim];
    //         // // splat to all the vertices in the simplex
    //         // for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
    //         //     // Compute the location of the lattice point explicitly (all but
    //         //     // the last coordinate - it's redundant because they sum to zero)
    //         //     for (int i = 0; i < m_pos_dim; i++) {
    //         //         key[i] = static_cast<short>(rem0[i] + remainder);
    //         //         if (rank[i] > m_pos_dim - remainder)
    //         //             key[i] -= (m_pos_dim + 1);
    //         //     }

    //         //     // Retrieve pointer to the value at this vertex.
    //         //     float *val = hashTable.lookup(key, true);
    //         //     // Accumulate values with barycentric weight.
    //         //     for (int i = 0; i < m_val_dim_hom-1 ; i++)
    //         //         val[i] += barycentric[remainder] * values_hom(n,i);
    //         //     val[m_val_dim_hom-1] += barycentric[remainder]; //homogeneous coordinate (as if value[vd-1]=1)

    //         //     // Record this interaction to use later when slicing
    //         //     matrix[idx].offset = val - hashTable.getValues();
    //         //     matrix[idx].weight = barycentric[remainder];
    //         //     idx++;


    //         //     //splat also the covariance matrix
    //         //     Eigen::Vector3f point; //point is the xy position of the pixel
    //         //     point <<  positions(n,0), positions(n,1), 0.0;
    //         //     int vertex_idx= hashTable.lookup_idx(key);
    //         //     // hashTable.m_cov_matrices[vertex_idx].push(point,  barycentric[remainder] );
    //         //     hashTable.m_cov_matrices[vertex_idx].push(point);
    //         // }


    //         //splat only on the closest vertex
    //         int closest_vertex_idx=-1;
    //         float largest_barycentric_coord=-1;
    //         for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
    //             if(barycentric[remainder]>largest_barycentric_coord){
    //                 largest_barycentric_coord=barycentric[remainder];
    //                 closest_vertex_idx=remainder ;
    //             }
    //         }
    //         //splat
    //         for (int i = 0; i < m_pos_dim; i++) {
    //             key[i] = static_cast<short>(rem0[i] + closest_vertex_idx);
    //             if (rank[i] > m_pos_dim - closest_vertex_idx)
    //                 key[i] -= (m_pos_dim + 1);
    //         }
    //         // Retrieve pointer to the value at this vertex.
    //         float *val = hashTable.lookup(key, true);
    //         // Accumulate values with barycentric weight.
    //         for (int i = 0; i < m_val_dim_hom-1 ; i++)
    //             val[i] += barycentric[closest_vertex_idx] * values(n,i); //  
    //         val[m_val_dim_hom-1] += barycentric[closest_vertex_idx]; //homogeneous coordinate (as if value[vd-1]=1)
    //         // for (int i = 0; i < m_val_dim_hom ; i++)
    //         //     val[i] += 1.0 * values_hom(n,i);
    //         // Record this interaction to use later when slicing
    //         for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
    //             matrix[idx].offset = val - hashTable.getValues();
    //             matrix[idx].weight = barycentric[closest_vertex_idx];
    //             idx++;
    //         }
    //         //splat also the covariance matrix
    //        Eigen::Vector3f point; //point is the xy position of the pixel
    //        point <<  values(n,0), values(n,1), 0.0;
    //        int vertex_idx= hashTable.lookup_idx(key);
    //        hashTable.m_cov_matrices[vertex_idx].push(point);

    //        //store also the vertex idx in which each position has been splatted to
    //        m_closest_vertex_idxs(n)=vertex_idx;



    //         delete[] key; 

            



    //     }
    //     // TIME_END("create latice points");
    // }


    /* Performs slicing out of position vectors. Note that the barycentric weights and the simplex
    * containing each position vector were calculated and stored in the splatting step.
    * We may reuse this to accelerate the algorithm. (See pg. 6 in paper.)
    */
    // void slice_point(float* out, int n) {
    // }

    // void slice(float* out){
    //     for (int n = 0; n < N; n++) {
    //         // slice_point(out, n);

    //         float* base = hashTable.getValues();

    //         for (int j = 0; j < m_val_dim_hom; j++)
    //             val[j] = 0;

    //         for (int i = 0; i <= m_pos_dim; i++) {
    //             MatrixEntry r = matrix[n * (m_pos_dim + 1) + i];
    //             for (int j = 0; j < m_val_dim_hom; j++) {
    //                 val[j] += r.weight * base[r.offset + j];
    //             }
    //         }

    //         float scale = 1.0 / val[m_val_dim_hom - 1];
    //         for (int j = 0; j < m_val_dim_hom - 1; j++) {
    //             out[n * (m_val_dim_hom - 1) + j] = val[j] * scale;
    //         }


    //     }
    // }   
    void slice(EigenMatrixXfRowMajor& out){
        for (int n = 0; n < N; n++) {
            // slice_point(out, n);

            float* base = hashTable.getValues();

            for (int j = 0; j < m_val_dim_hom; j++)
                val[j] = 0;

            for (int i = 0; i <= m_pos_dim; i++) {
                MatrixEntry r = matrix[n * (m_pos_dim + 1) + i];
                for (int j = 0; j < m_val_dim_hom; j++) {
                    val[j] += r.weight * base[r.offset + j];
                }
            }

            float scale = 1.0 / val[m_val_dim_hom - 1];
            for (int j = 0; j < m_val_dim_hom - 1; j++) {
                out(n, j) = val[j] * scale;
            }


        }
    }


    /* Performs a Gaussian blur along each projected axis in the hyperplane. */
    void blur(bool reverse) {

        // Prepare arrays
        auto n1_key = new short[m_pos_dim + 1];
        auto n2_key = new short[m_pos_dim + 1];

        //old and new values contain the lattice points before and after blur
        //auto new_values = new T[vd * hashTable.size()];
        // auto new_values = new float[m_val_dim_hom * hashTable.get_capacity()];
        EigenMatrixXfRowMajor new_values;
        new_values.resize(HASH_TABLE_INIT_CAPACITY, m_val_dim_hom);

        // std::cout << "created new_values" << "\n";

        auto zero = new float[m_val_dim_hom]{0.0};
        // std::cout << "created zero" << "\n";
        //for (int k = 0; k < vd; k++)
        //    zero[k] = 0;

        // For each of pd+1 axes,
        for (int remainder=reverse?m_pos_dim:0; remainder >= 0 && remainder <= m_pos_dim; reverse?remainder--:remainder++){
            // For each vertex in the lattice,
            for (int i = 0; i < hashTable.size(); i++) { // blur point i in dimension j
                // std::cout << "blurring point " << i << "\n";


                short *key = hashTable.getKeys() + i * m_pos_dim; // keys to current vertex
                for (int k = 0; k < m_pos_dim; k++) {
                    n1_key[k] = key[k] + 1;
                    n2_key[k] = key[k] - 1;
                }

                n1_key[remainder] = key[remainder] - m_pos_dim;
                n2_key[remainder] = key[remainder] + m_pos_dim; // keys to the neighbors along the given axis.

                float *oldVal = hashTable.getValues() + i * m_val_dim_hom;
                // float *newVal = new_values + i * m_val_dim_hom;

                float *n1_value, *n2_value;

                n1_value = hashTable.lookup(n1_key, false); // look up first neighbor
                if (n1_value == nullptr)
                    n1_value = zero;

                n2_value = hashTable.lookup(n2_key, false); // look up second neighbor
                if (n2_value == nullptr)
                    n2_value = zero;

                // Mix values of the three vertices
                for (int k = 0; k < m_val_dim_hom; k++)
                    new_values(i,k)=(0.25 * n1_value[k] + 0.5 * oldVal[k] + 0.25 * n2_value[k]);

                    // newVal[k] = (0.25 * n1_value[k] + 0.5 * oldVal[k] + 0.25 * n2_value[k]);
                    // // newVal[k] = std::fabs(1.0 * n1_value[k] + 0.0 * oldVal[k] - 1.0 * n2_value[k]);
                    // if(n1_value==zero && n2_value==zero ){
                    //     newVal[k]=1.0;
                    // }else{
                    //     newVal[k]=0.0;
                    // }
            }
            // the freshest data is now in old_values, and new_values is ready to be written over
            hashTable.set_new_vals(new_values);
            // std::swap(hashTable.getValues(), new_values);
        }

        // delete[](new_values);
        delete[] zero;
        delete[] n1_key;
        delete[] n2_key;
    }
    
    void set_sigmas(std::initializer_list<  std::pair<float, int> > sigmas_list){
        m_sigmas.clear();
        bool first=true;
        for(auto sigma_pair : sigmas_list){
            float sigma=sigma_pair.first; //value of the sigma
            int nr_dim=sigma_pair.second; //how many dimensions are affected by this sigma
            for(int i=0; i < nr_dim; i++){
                m_sigmas.push_back(sigma);
            }

            //set the spacial dimensions
            if(first){
                m_spacial_dim=nr_dim;
                first=false;
            }
        }
    }


    EigenMatrixXfRowMajor compute_scaled_positions(const EigenMatrixXfRowMajor& positions){     
        CHECK((int)m_sigmas.size()>=positions.cols()) << "Nr sigmas does not correspond with nr of cols in positions matrix. Please set some sigmas first with set_sigmas(). m_sigmas is " << m_sigmas.size() << " position cols is " << positions.cols();
        CHECK(m_pos_dim!=-1) << "m_pos_dim not set";
        CHECK(m_val_dim_hom!=-1) << "m_val_dim_hom not set";

        // TODO set the scaling factor here tooQ

        EigenMatrixXfRowMajor positions_scaled;
        positions_scaled.resizeLike(positions);
        for(int i = 0; i < positions.cols(); i++){
            positions_scaled.col(i)=positions.col(i)/m_sigmas[i];
        }

        return positions_scaled;

    }



    void splat(const EigenMatrixXfRowMajor& positions_raw, const EigenMatrixXfRowMajor& values, const std::string splatting_type,  bool scale_positions=true){

        EigenMatrixXfRowMajor positions;
        // std::cout << "doign sclaing " << std::endl;
        if(scale_positions){
            positions=compute_scaled_positions(positions_raw); //scale by the sigmas 
        }else{
            positions=positions_raw;
        }

        for (int n = 0; n < positions.rows(); n++) {

            // std::cout << "embedding "<<n << " out of " << positions.rows() << std::endl;
            // embed position vector into the hyperplane
            // first rotate position into the (pd+1)-dimensional hyperplane
            // sm contains the sum of 1..n of our feature vector
            // float sm = 0;
            // for (int i = m_pos_dim; i > 0; i--) {
            //     float cf = positions(n,i - 1) * scaleFactor[i - 1];
            //     elevated[i] = sm - i * cf;
            //     sm += cf;
            // }
            // elevated[0] = sm;

            //scale the position by (m_pos_dim+1)/sqrt(2.0/3/0) to account for the variance that occurs in whole algorithm (under the assumption that we want to do a gaussian blur of variance 1). This step will likely not need to be done for the neural network but you would need to retweak your sigmas if you comment this out
            float invStdDev = (m_pos_dim + 1) * sqrt(2.0 / 3);
            positions.row(n) = positions.row(n).array()*invStdDev;


            //attempt 2 to calculate Ep as explained on page 30 of Andrew Adams thesis. Also we don't scale the position by the inverse variance as we will learn the weights either way and that would just scale them by a constant factor. In his equation d is what we call here m_pos_dim
            Eigen::MatrixXf E_left(m_pos_dim+1, m_pos_dim );
            Eigen::MatrixXf E_right(m_pos_dim, m_pos_dim );
            E_left.setZero();
            E_right.setZero();
            //E left is has at the bottom a square matrix which has an upper triangular part of ones. Afterwards the whole E_left gets appended another row on top of all ones
            E_left.bottomRows(m_pos_dim).triangularView<Eigen::Upper>().setOnes();
            //the diagonal of the bottom square is linearly incresing from [-1, -m_pos_dim]
            E_left.bottomRows(m_pos_dim).diagonal().setLinSpaced(m_pos_dim,1,m_pos_dim);
            E_left.bottomRows(m_pos_dim).diagonal()= -E_left.bottomRows(m_pos_dim).diagonal();
            //E_left has the first row all set to ones
            E_left.row(0).setOnes();
            // VLOG(1) << "E left is \n" << E_left;
            //E right is just a diagonal matrix with entried in the diag set to 1/sqrt((d+1)(d+2)). Take into account that the d in the paper starts at 1 and we start at 0 so we add a +1 to diag_idx
            for(int diag_idx=0; diag_idx<m_pos_dim; diag_idx++){
                E_right(diag_idx, diag_idx) =  1.0 / (sqrt((diag_idx + 1) * (diag_idx + 2))) ;
            }
            // VLOG(1) << "E right is \n" << E_right;

            //rotate into H_d
            Eigen::MatrixXf E = E_left*E_right;
            // VLOG(1) << "E is \n" << E;
            Eigen::VectorXf elevated_eigen = E*positions.row(n).transpose();
            // VLOG(1) << "Elevated eigen is " << elevated_eigen;
            for(int p_idx=0; p_idx<m_pos_dim+1; p_idx++){
                elevated[p_idx] = elevated_eigen(p_idx);
            }




            // std::cout << "finding simplex "<<n << std::endl;
            find_enclosing_simplex();

            // std::cout << "barycentric coords"<<n << std::endl;
            compute_barycentric_coordinates();

            // std::cout << "pos dim is"<<m_pos_dim << std::endl;
            auto key = new short[m_pos_dim];

            if(splatting_type=="barycentric"){
                // splat to all the vertices in the simplex    

                //calculate the closest one just so we can store it in our vector of the closest_Vertex_idx
                int closest_vertex_idx=-1;
                float largest_barycentric_coord=-1;
                for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
                    if(barycentric[remainder]>largest_barycentric_coord){
                        largest_barycentric_coord=barycentric[remainder];
                        closest_vertex_idx=remainder ;
                    }
                }

                for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
                    // Compute the location of the lattice point explicitly (all but
                    // the last coordinate - it's redundant because they sum to zero)
                    for (int i = 0; i < m_pos_dim; i++) {
                        key[i] = static_cast<short>(rem0[i] + remainder);
                        if (rank[i] > m_pos_dim - remainder)
                            key[i] -= (m_pos_dim + 1);
                    }

                    // Retrieve pointer to the value at this vertex.
                    float *val = hashTable.lookup(key, true);
                    // Accumulate values with barycentric weight.
                    for (int i = 0; i < m_val_dim_hom-1 ; i++)
                        val[i] += barycentric[remainder] * values(n,i);
                    val[m_val_dim_hom-1] += barycentric[remainder]; //homogeneous coordinate (as if value[vd-1]=1)

                    // Record this interaction to use later when slicing
                    matrix[idx].offset = val - hashTable.getValues();
                    matrix[idx].weight = barycentric[remainder];
                    idx++;

                     //splat also the covariance matrix
                    Eigen::Vector3f point; //point is the xy position of the pixel
                    point.setZero();
                    for(int i = 0; i < m_spacial_dim; i++){
                        point(i) = values(n,i);
                    }
                    //    point <<  values(n,0), values(n,1), 0.0;
                    int vertex_idx= hashTable.lookup_idx(key);
                    hashTable.m_cov_matrices[vertex_idx].push(point);

                    //store also the vertex idx in which each position has been splatted to
                    if(remainder==closest_vertex_idx){
                        m_closest_vertex_idxs(n)=vertex_idx;
                    }
                    m_lattice_vertices_modified[vertex_idx]=true;
                }

            }else if(splatting_type=="nearest"){
                //splat only on the closest vertex
                int closest_vertex_idx=-1;
                float largest_barycentric_coord=-1;
                for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
                    if(barycentric[remainder]>largest_barycentric_coord){
                        largest_barycentric_coord=barycentric[remainder];
                        closest_vertex_idx=remainder ;
                    }
                }
                //splat
                for (int i = 0; i < m_pos_dim; i++) {
                    key[i] = static_cast<short>(rem0[i] + closest_vertex_idx);
                    if (rank[i] > m_pos_dim - closest_vertex_idx)
                        key[i] -= (m_pos_dim + 1);
                }
                // Retrieve pointer to the value at this vertex.
                float *val = hashTable.lookup(key, true);
                // Accumulate values with barycentric weight.
                for (int i = 0; i < m_val_dim_hom-1 ; i++)
                    val[i] += barycentric[closest_vertex_idx] * values(n,i); //  
                val[m_val_dim_hom-1] += barycentric[closest_vertex_idx]; //homogeneous coordinate (as if value[vd-1]=1)
                // for (int i = 0; i < m_val_dim_hom ; i++)
                //     val[i] += 1.0 * values_hom(n,i);
                // Record this interaction to use later when slicing
                for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
                    matrix[idx].offset = val - hashTable.getValues();
                    matrix[idx].weight = barycentric[closest_vertex_idx];
                    idx++;
                }
                //splat also the covariance matrix
                CHECK(m_spacial_dim==3) << "To splat into a covariance matrix the spacial dimension needs to be 3. It is: " <<m_spacial_dim;
                Eigen::Vector3f point; //point is the xy position of the pixel
                point.setZero();
                for(int i = 0; i < m_spacial_dim; i++){
                    point(i) = values(n,i);
                }
                //    point <<  values(n,0), values(n,1), 0.0;
                int vertex_idx= hashTable.lookup_idx(key);
                hashTable.m_cov_matrices[vertex_idx].push(point);

                //store also the vertex idx in which each position has been splatted to
                m_closest_vertex_idxs(n)=vertex_idx;
                m_lattice_vertices_modified[vertex_idx]=true;
            }else{
                LOG(FATAL) << "Not a known type of splatting_type. Should be either nearest or barycentric, but it is " << splatting_type;
            }
        




            delete[] key; 
        }

    }

    void blur(){
        // Prepare arrays
        auto n1_key = new short[m_pos_dim + 1];
        auto n2_key = new short[m_pos_dim + 1];

        //old and new values contain the lattice points before and after blur
        //auto new_values = new T[vd * hashTable.size()];
        // auto new_values = new float[m_val_dim_hom * hashTable.get_capacity()];
        EigenMatrixXfRowMajor new_values;
        new_values.resize(HASH_TABLE_INIT_CAPACITY, m_val_dim_hom);

        // std::cout << "created new_values" << "\n";

        auto zero = new float[m_val_dim_hom]{0.0};
        // std::cout << "created zero" << "\n";
        //for (int k = 0; k < vd; k++)
        //    zero[k] = 0;

        // For each of pd+1 axes,
        for (int remainder=0; remainder >= 0 && remainder <= m_pos_dim; remainder++){
            // For each vertex in the lattice,
            for (int i = 0; i < hashTable.size(); i++) { // blur point i in dimension j
                // std::cout << "blurring point " << i << "\n";


                short *key = hashTable.getKeys() + i * m_pos_dim; // keys to current vertex
                for (int k = 0; k < m_pos_dim; k++) {
                    n1_key[k] = key[k] + 1;
                    n2_key[k] = key[k] - 1;
                }

                n1_key[remainder] = key[remainder] - m_pos_dim;
                n2_key[remainder] = key[remainder] + m_pos_dim; // keys to the neighbors along the given axis.

                float *oldVal = hashTable.getValues() + i * m_val_dim_hom;
                // float *newVal = new_values + i * m_val_dim_hom;

                float *n1_value, *n2_value;

                n1_value = hashTable.lookup(n1_key, false); // look up first neighbor
                if (n1_value == nullptr)
                    n1_value = zero;

                n2_value = hashTable.lookup(n2_key, false); // look up second neighbor
                if (n2_value == nullptr)
                    n2_value = zero;

                // Mix values of the three vertices
                for (int k = 0; k < m_val_dim_hom; k++)
                    new_values(i,k)=(0.25 * n1_value[k] + 0.5 * oldVal[k] + 0.25 * n2_value[k]);

                    // newVal[k] = (0.25 * n1_value[k] + 0.5 * oldVal[k] + 0.25 * n2_value[k]);
                    // // newVal[k] = std::fabs(1.0 * n1_value[k] + 0.0 * oldVal[k] - 1.0 * n2_value[k]);
                    // if(n1_value==zero && n2_value==zero ){
                    //     newVal[k]=1.0;
                    // }else{
                    //     newVal[k]=0.0;
                    // }
            }
            // the freshest data is now in old_values, and new_values is ready to be written over
            hashTable.set_new_vals(new_values);
            // std::swap(hashTable.getValues(), new_values);
        }

        // delete[](new_values);
        delete[] zero;
        delete[] n1_key;
        delete[] n2_key;

    }

    EigenMatrixXfRowMajor slice(const EigenMatrixXfRowMajor& positions_raw, bool scale_positions=true){

        EigenMatrixXfRowMajor positions;
        if(scale_positions){
            positions=compute_scaled_positions(positions_raw); //scale by the sigmas and by the scaling factors
        }else{
            positions=positions_raw;
        }
        

        EigenMatrixXfRowMajor output;
        output.resize(positions.rows(), m_val_dim_hom-1);

        for (int n = 0; n < N; n++) {
            // VLOG(1) << "computing output " << n << " out of " << N;

            float* base = hashTable.getValues();


            for (int j = 0; j < m_val_dim_hom; j++)
                val[j] = 0;

            for (int i = 0; i <= m_pos_dim; i++) {
                MatrixEntry r = matrix[n * (m_pos_dim + 1) + i];
                for (int j = 0; j < m_val_dim_hom; j++) {
                    val[j] += r.weight * base[r.offset + j];
                }
            }

            float scale = 1.0 / val[m_val_dim_hom - 1];
            for (int j = 0; j < m_val_dim_hom - 1; j++) {
                output(n, j) = val[j] * scale;
                // float f = val[j] * scale;
            }
        }
        return output;
    }


    EigenMatrixXfRowMajor slice_no_precomputation(const EigenMatrixXfRowMajor& positions_raw, bool scale_positions=true){

        EigenMatrixXfRowMajor positions;
        // std::cout << "doign sclaing " << std::endl;
        if(scale_positions){
            positions=compute_scaled_positions(positions_raw); //scale by the sigmas and by the scaling factors
        }else{
            positions=positions_raw;
        }


        EigenMatrixXfRowMajor out_vals;
        out_vals.resize(positions.rows(), m_val_dim_hom-1);
            

        for (int n = 0; n < positions.rows(); n++) {

            // std::cout << "embedding "<<n << " out of " << positions.rows() << std::endl;
            // embed position vector into the hyperplane
            // first rotate position into the (pd+1)-dimensional hyperplane
            // sm contains the sum of 1..n of our feature vector
            float sm = 0;
            for (int i = m_pos_dim; i > 0; i--) {
                float cf = positions(n,i - 1) * scaleFactor[i - 1];
                elevated[i] = sm - i * cf;
                sm += cf;
            }
            elevated[0] = sm;

            // std::cout << "finding simplex "<<n << std::endl;
            find_enclosing_simplex();

            // std::cout << "barycentric coords"<<n << std::endl;
            compute_barycentric_coordinates();

            // std::cout << "pos dim is"<<m_pos_dim << std::endl;
            auto key = new short[m_pos_dim];

            //here we accumulate the values and the homogeneous term
            Eigen::VectorXf val_hom(m_val_dim_hom);
            val_hom.setZero();

            for (int remainder = 0; remainder <= m_pos_dim; remainder++) {
                // Compute the location of the lattice point explicitly (all but
                // the last coordinate - it's redundant because they sum to zero)
                for (int i = 0; i < m_pos_dim; i++) {
                    key[i] = static_cast<short>(rem0[i] + remainder);
                    if (rank[i] > m_pos_dim - remainder)
                        key[i] -= (m_pos_dim + 1);
                }

                // Retrieve pointer to the value at this vertex.
                float *val = hashTable.lookup(key, /*create*/ false);

                //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
                if(val!=nullptr){
                    for (int i = 0; i < m_val_dim_hom ; i++){
                        val_hom(i)+= val[i]* barycentric[remainder];
                    }
                }
            
            }

            //divide by the homogeneous coord
            Eigen::VectorXf out_val(m_val_dim_hom-1);
            for (int i = 0; i < m_val_dim_hom-1; i++){
                float weight=val_hom[m_val_dim_hom-1];
                if(weight!=0.0){ //to avoid divisionz by 0
                    out_val(i)= val_hom[i] / val_hom[m_val_dim_hom-1];
                }else{ //the weight is 0 which means we landed in a simplex that is not allocated. The value will just be 0 then
                    out_val(i)= 0.0;
                }
            }


            out_vals.row(n)=out_val;


            delete[] key; 
        }

        return out_vals;
    }

    EigenMatrixXfRowMajor filter(const EigenMatrixXfRowMajor& positions_raw, const EigenMatrixXfRowMajor& values){

        EigenMatrixXfRowMajor positions=compute_scaled_positions(positions_raw); //scale by the sigmas and by the scaling factors

        splat(positions,values, "barycentric", false);
        blur();
        return slice(positions, false);
    }

    void begin_new_splat(){
        std::fill(m_lattice_vertices_modified.begin(), m_lattice_vertices_modified.end(), false); 
        idx=0; //HACK to reset the writing into the replat matrix so we dont overflow it
    }


public:

    PermutohedralLatticeCPU_IMPL(int pos_dim, int val_dim, int N_): m_pos_dim(pos_dim), m_val_dim_hom(val_dim + 1), N(N_), hashTable(pos_dim, val_dim+1) {

        // Allocate storage for various arrays
        matrix = std::unique_ptr<MatrixEntry[]>(new MatrixEntry[N * (m_pos_dim + 1)]);
        //matrix = new MatrixEntry[N * (pd + 1)];
        idx = 0;

        //lattice properties
        scaleFactor = compute_scale_factor();

        //arrays that are used in splatting and slicing, they are overwritten for each point but we only allocate once for speed
        // position embedded in subspace Hd
        elevated = std::unique_ptr<float[]>(new float[m_pos_dim + 1]);
        // remainder-0 and rank describe the enclosing simplex of a point
        rem0 = std::unique_ptr<float[]>(new float[m_pos_dim + 1]);
        rank = std::unique_ptr<short[]>(new short[m_pos_dim + 1]);
        // barycentric coordinates of position
        barycentric = std::unique_ptr<float[]>(new float[m_pos_dim + 2]);
        //val
        val = std::unique_ptr<float[]>(new float[m_val_dim_hom]);

        m_closest_vertex_idxs.resize(N);
        m_closest_vertex_idxs.setZero();

        m_lattice_vertices_modified.resize(HASH_TABLE_INIT_CAPACITY,false);

    }

    // void filter(float * output, const float* values, const float* positions, bool reverse) {
    void filter(EigenMatrixXfRowMajor& output, const EigenMatrixXfRowMajor values, const EigenMatrixXfRowMajor& positions, bool reverse) {

        // auto values_hom = new float[N * 4];
        // for (int n = 0; n < N; n++) {
        //     values_hom[n*4+0] = values[n*(3)+0];
        //     values_hom[n*4+1] = values[n*(3)+1];
        //     values_hom[n*4+2] = values[n*(3)+2];
        //     values_hom[n*4+3] = 1.0;
        // }

        // EigenMatrixXfRowMajor values_hom;
        // values_hom.resize(N,m_val_dim_hom);
        // values_hom.block(0,0,N,m_val_dim_hom-1)=values;
        // values_hom.block(0,m_val_dim_hom-1,N,1).setOnes();


        // TIME_START("splat");
        std::cout << "splatting" << "\n";
        splat(positions, values, "barycentric" ,true);
        // TIME_END("splat");

        // TIME_START("blur");
        std::cout << "blurring" << "\n";
        blur(reverse);
        // TIME_END("blur");

        // TIME_START("slice");
        std::cout << "slicing" << "\n";
        slice(output);
        // TIME_END("slice");

        // delete[] values_hom;
    }

};



static void compute_kernel_cpu(const float * reference,
                               float * positions,
                               int num_super_pixels,
                               int reference_channels,
                               int n_sdims,
                               const int *sdims,
                               float spatial_std,
                               float feature_std){

    int num_dims = n_sdims + reference_channels;

    for(int idx = 0; idx < num_super_pixels; idx++){
        int divisor = 1;
        for(int sdim = n_sdims - 1; sdim >= 0; sdim--){
            positions[num_dims * idx + sdim] = ((idx / divisor) % sdims[sdim]) / spatial_std;
            divisor *= sdims[sdim];
        }
        for(int channel = 0; channel < reference_channels; channel++){
            positions[num_dims * idx + n_sdims + channel] = reference[idx * reference_channels + channel] / feature_std;
        }
    }
};

