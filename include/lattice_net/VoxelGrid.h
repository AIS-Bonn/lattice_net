#pragma once 

#include <Eigen/Core>

#include "surfel_renderer/core/MeshCore.h" 
#include "surfel_renderer/lattice/Voxel.h"


// struct Voxel{
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//     // bool m_is_empty=true; //not really needed. Only for debugging
//     // Eigen::MatrixXf m_points; //samples that fell inside this voxel
//     // int m_label_gt;
//     // // int nr_samples=0;
//     // // std::vector<int> m_local2global; //for and index of a point local to this voxel (in m_points) it gives the index of the point in the original big cloud (received in voxelize function)
//     // // torch::Tensor m_samples_tensor; //samples that fell inside this voxel
//     // // Eigen::MatrixXf m_samples_eigen; //samples that fell inside this voxel

//     //attempt 2 at making a good voxel grid
//     void add_point(const Eigen::VectorXd& point){

//     }

//     void add_label(const int label){

//     }

//     void compute_argmax_label(){

//     } 

//     int nr_contained_points(){

//     }

//     int label_gt(){

//     }

// };

class VoxelGrid
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VoxelGrid();
    
    //params
    // void set_max_points_per_voxel(const int max_nr_points); //each voxel stores a maximum amount of points inside of it
    void set_voxel_size(const float voxel_size);
    void set_grid_extents(const Eigen::Vector3d& grid_extents); // the coordinates of the voxels are defined in a cube of a certain size around the origin. The size of that cube in units is defined with this. It should be a large enough value so that it contains the whole point cloud

    // void voxelize(const MeshCore& cloud);
    // Eigen::MatrixXf get_all_points(); //get the points contained in all the voxels sorted by how the voxels are laid out
    // std::vector<int> get_nr_points_per_voxel();



    void voxelize(const MeshCore& cloud);
    MeshCore grid_dual(); //get a mesh represeniting the dual of the voxel grid, so for each center of a voxel you get a point in the cloud
    // void show();

    void clear(); //clears the current voxels
    int nr_active_voxels(); //return M which is the number of actually allocated voxels
    Eigen::VectorXi nr_points_per_voxel(); //return a matrix of form [M] containing the number of points stored inside of it
    Eigen::MatrixXf voxel_points(); //returns a matrix of [nr_points, ndim] where nr_points is the number of points in the original point cloud and ndim is the dimension of the points(usually for xyz). Effectivelly get the points contained in all the voxels sorted by how the voxels are laid out
    Eigen::MatrixXf voxel_points_normalized(); //returns a matrix of [nr_points, ndim]. Unlike voxel_points, the center of the voxel is substracted from each point, so the positions/features will be normalized in between [-1,1]
    Eigen::VectorXi voxel_labels_gt(); //return a matrix of the form [M] where for each voxel we store the class labels that is more in consensus with all of the points inside of it
    Eigen::MatrixXi voxel_coords(); //return matrix of [M,3] containing for each voxel their corrdinates in xyz space. The origin is with respect to the first voxel in the voxel grid not with respect to the world coordinate system
    Eigen::VectorXi voxel_ids_per_point(); //returns a vector of [nr_points] where for each point it says which voxel id it is contained in. Voxel id is the monotonically increasing id
    Eigen::Vector3i grid_sizes();

   

private:

    int find_enclosing_simplex(const Eigen::Vector3d& point);
    Eigen::Vector3i compute_grid_sizes();

    //params
    float m_voxel_size;

    //internal
    Eigen::Vector3i m_grid_sizes; // how many voxels are in x,y,z directions 
    Eigen::Vector3d m_grid_extents; // how many units does the grid span around in x,y,z direction (it only expressed it in one direction, in the sense that a 3x3 grid has extent 1 because it's a 1 ring around the center)
    // std::vector<Voxel> m_voxels;
    // Eigen::Vector3d m_min_bounding_box; // the voxel grid starts at 0,0,0 but the cloud can be in any arbitrary stating position. We keep track of the min of the clouds bounding box so we can internally substract it
    std::map<int, Voxel> m_idx2voxel_map; //maps from the 1D linear idx of the voxels to the voxels themselves. Needs to be a map because the idx_1d is ordered internally
    MeshCore m_last_voxelized_cloud;

   
};
