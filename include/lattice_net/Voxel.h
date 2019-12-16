#pragma once 


#include <Eigen/Core>
#include <vector>

class Voxel
{
public:
    Voxel();

    std::vector<Eigen::MatrixXd> m_points; // points contained inside the voxel
    std::vector<int> m_labels_gt; //labels_gt of each of the points inside the voxel
    int m_label_gt; //maximum consensus between m_labels_gt. Only valid after compute_argmax_label()
    int id; //monotonically increasing id, even if they are jump in the voxels because of their sparsity, this id will increase linearly
    // int idx_1D; //idx of the linearly laid 3D voxel grid, ordering is zyx. use MiscUtils idx3d_to_1d
    // int idx_3D; //idx of the voxel inside a 3D grid. Expressed the center fo this voxel in xyz where the 0,0,0 of the grid is the first voxel 

    void add_point(const Eigen::VectorXd& point);
    void add_label_gt(const int label);
    void compute_argmax_label(); 
    int nr_contained_points() const;
    int label_gt() const;

};