#include "surfel_renderer/lattice/VoxelGrid.h"


//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//my stuff
#include "surfel_renderer/viewer/Scene.h"
#include "surfel_renderer/utils/MiscUtils.h"
#include "surfel_renderer/utils/Profiler.h"
#include "surfel_renderer/data_loader/LabelMngr.h"

//c++
#include <math.h>
#include <iostream>

using namespace er::utils;



VoxelGrid::VoxelGrid():
    m_voxel_size(0.01),
    // m_voxel_size(1.0),
    m_grid_extents(1,1,1)
{
    m_grid_sizes=compute_grid_sizes();

}

// void VoxelGrid::set_voxel_size(const float voxel_size){
//     m_voxel_size=voxel_size;
// }

// void VoxelGrid::voxelize(const MeshCore& cloud){

//     TIME_SCOPE("voxelize");

//     //calculate size of voxel_grid
//     Eigen::Vector3d max= cloud.V.colwise().maxCoeff();
//     m_min_bounding_box= cloud.V.colwise().minCoeff();
//     Eigen::Vector3d cloud_span=max-m_min_bounding_box;
//     // Eigen::Vector3d min= cloud.V.colwise().minCoeff();
//     // Eigen::Vector3d cloud_span=max-min;
//     // VLOG(1) << "cloud_span is " << cloud_span.transpose();
//     m_grid_sizes=(cloud_span/m_voxel_size).array().ceil().cast<int>(); // Divide uniformly the space
//     // VLOG(1) << "grid_size is " << m_grid_sizes.transpose();
//     //restrict the grid size from beign too big 
//     if(m_grid_sizes.prod()>1e6){
//         LOG(FATAL)  << "The voxel grid that will be created will be too big. Try to increase the voxel size. The nr of voxels would be " <<m_grid_sizes.prod() << " with a grid size of " << m_grid_sizes.transpose() ;
//     }
//     // VLOG(1) << "Creating voxels nr " << m_grid_sizes.prod();
//     m_voxels.clear();
//     m_voxels.resize(m_grid_sizes.prod());
//     // VLOG(1) << "nr of voxels" << m_voxels.size();

//     //precompute where the points and the labels go
//     std::vector< std::vector<int> >  local2global_per_voxel; //for and index of a point local to this voxel (in m_points) it gives the index of the point in the original big cloud 
//     local2global_per_voxel.resize(m_voxels.size());
//     Eigen::MatrixXi probs_per_voxel; //assign for each voxel a label corresponding to the maximum amount of points voting for a label 
//     const int max_nr_classes=128;
//     probs_per_voxel.resize(m_voxels.size(), max_nr_classes); // assume we have for each voxel a row of 128 labels
//     probs_per_voxel.setZero();
//     for(int i = 0; i < cloud.V.rows(); i++){
//         int v_idx=find_enclosing_simplex(Eigen::Vector3d(cloud.V.row(i)));
//         m_voxels[v_idx].m_is_empty=false;
//         local2global_per_voxel[v_idx].push_back(i);

//         //labels
//         int label_gt=cloud.L_gt(i);
//         probs_per_voxel(v_idx, label_gt )++;
//     }

//     //assign points for the voxel
//     for(int i=0; i<m_voxels.size(); i++){
//         int nr_points_contained=local2global_per_voxel[i].size();
//         for(int s_idx=0; s_idx<nr_points_contained; s_idx++){
//             int global_sample_idx=local2global_per_voxel[i][s_idx];
//             m_voxels[i].m_points.row(s_idx) = cloud.V.row(global_sample_idx).cast<float>();
//         }
//     }


//     //assign the labels for each voxel by argmaxing over all the votes it got for the classes
//     for(int i=0; i<m_voxels.size(); i++){
//         int max_prob=0;
//         int location_max_prob=0;
//         for(int c=0; c<max_nr_classes; c++){
//            if(probs_per_voxel(i,c)>max_prob){
//                max_prob=probs_per_voxel(i,c);
//                location_max_prob=c;
//            } 
//         }
//         m_voxels[i].m_label_gt=location_max_prob;
//     }

  


// }

// int VoxelGrid::find_enclosing_simplex(const Eigen::Vector3d& point){
//     Eigen::Vector3d point_shifted=point-m_min_bounding_box;
//     // Eigen::Vector3d point_shifted=point; //introduce back the shift
//     // Eigen::Vector3i idx_3D= point_shifted.cast<int>() / m_voxel_size; //this automatically floors the point_shifted
//     Eigen::Vector3i idx_3D= (point_shifted/ m_voxel_size).cast<int>(); //this automatically floors the point_shifted
//     int v_idx= idx_3D_to_1D(idx_3D, m_grid_sizes);
    
//     return v_idx;
// }

// Eigen::MatrixXf VoxelGrid::get_all_points(){
//     //get total nr of points 
//     int nr_total=0;
//     for(int i=0; i<m_voxels.size(); i++){
//         nr_total+=m_voxels[i].m_points.rows();
//     }

//     //make a big matrix to contain all points
//     Eigen::MatrixXf points_all;
//     points_all.resize(nr_total,3);

//     //add them to the big matrix
//     int idx_insert=0;
//     for(int i=0; i<m_voxels.size(); i++){
//         for(int p_idx=0; p_idx<m_voxels[i].m_points.rows(); p_idx++){
//             points_all.row(idx_insert) = m_voxels[i].m_points.row(p_idx);
//         }
//     }

//     return points_all;
// }

// std::vector<int> VoxelGrid::get_nr_points_per_voxel(){

//     std::vector<int> nr_points_per_voxel(m_voxels.size(), 0);
//     for(int i=0; i<m_voxels.size(); i++){
//         nr_points_per_voxel[i]+=m_voxels[i].m_points.rows();
//     }

//     return nr_points_per_voxel;
// }






//attempt 2 at making a good voxelizer 
// void VoxelGrid::set_max_points_per_voxel(const int max_nr_points){
//     m_max_points_per_voxel=max_nr_points;
// } 

void VoxelGrid::set_voxel_size(const float voxel_size){
    m_voxel_size=voxel_size;
    m_grid_sizes=compute_grid_sizes();
}
void VoxelGrid::set_grid_extents(const Eigen::Vector3d& grid_extents){
    m_grid_extents=grid_extents;
    m_grid_sizes=compute_grid_sizes();
}

Eigen::Vector3i VoxelGrid::compute_grid_sizes(){
    // returns how many voxels are in x,y,z directions
    Eigen::Vector3d grid_sizes_d;
    //m_grid_extents has the extent in one direction, but we have to take into accout that the thing is symetrical so we have also on the other size and also we have the middle column and the middle row
    Eigen::Vector3d grid_extent_full=m_grid_extents.array()*2+1;
    grid_sizes_d=grid_extent_full / (double)m_voxel_size;
    // std::cout << "m_grid extent is " << m_grid_extents << " m_voxel size is " << " grid_sizes_d is " << grid_sizes_d <<std::endl;
    grid_sizes_d.array().ceil();
    return grid_sizes_d.cast<int>();
}


void VoxelGrid::voxelize(const MeshCore& cloud){
    m_last_voxelized_cloud=cloud;

    for(int i = 0; i < cloud.V.rows(); i++){
        int idx_1D=find_enclosing_simplex( Eigen::Vector3d(cloud.V.row(i)) );

        Voxel& voxel=m_idx2voxel_map[idx_1D]; //alocate a voxel which is stored in a hashmap with a key given by their 1D idx
        // voxel.idx_1D=idx_1D;
        // voxel.idx_1D=idx_1D_to_3D(idx_1D, m_grid_sizes);
        voxel.add_point( Eigen::Vector3d(cloud.V.row(i)) );
        voxel.add_label_gt( cloud.L_gt(i) );

    }

    //asign for each voxel and monotonically increasing id 
    int id=0;
    for( auto & [idx_1D, voxel] : m_idx2voxel_map ){
        voxel.id=id;
        id++;
    }

    //once all the points are assigned to the voxels then we calculate also the labels that are the maximum over all of the points stored inside 
    for( auto & [idx_1D, voxel] : m_idx2voxel_map ){
        voxel.compute_argmax_label(); //this computes the m_label_gt inside each voxel by argmaxing over the votes it gets from all the containing point
    }

}

int VoxelGrid::nr_active_voxels(){
    return m_idx2voxel_map.size();
}

Eigen::VectorXi VoxelGrid::nr_points_per_voxel(){
    Eigen::VectorXi mat( nr_active_voxels()); 
    mat.setZero();
    for( auto const& [idx_1D, voxel] : m_idx2voxel_map ){
        mat(voxel.id)=voxel.nr_contained_points();
    }
    return mat;
}

Eigen::MatrixXf VoxelGrid::voxel_points(){
    Eigen::MatrixXf mat( m_last_voxelized_cloud.V.rows(), m_last_voxelized_cloud.V.cols() );
    //TODO
    int nr_points_inserted=0;
    for( auto const& [idx_1D, voxel] : m_idx2voxel_map ){ //iterating over std::map gives the keys in order
        //add the point from this voxel
        for(size_t p_idx = 0; p_idx < voxel.nr_contained_points(); p_idx++){
            CHECK(nr_points_inserted<mat.rows()) << "Illegal access. nr_points_inserted is " << nr_points_inserted << " mat.rows() os " << mat.rows();
            // mat.row(nr_points_inserted) = voxel.m_points[p_idx].cast<float>();
            mat(nr_points_inserted,0) = voxel.m_points[p_idx](0);
            mat(nr_points_inserted,1) = voxel.m_points[p_idx](1);
            mat(nr_points_inserted,2) = voxel.m_points[p_idx](2);
            nr_points_inserted++;
        }
    }

    return mat;    

}


Eigen::MatrixXf VoxelGrid::voxel_points_normalized(){
    Eigen::MatrixXf mat( m_last_voxelized_cloud.V.rows(), m_last_voxelized_cloud.V.cols() );
    //TODO
    int nr_points_inserted=0;
    for( auto const& [idx_1D, voxel] : m_idx2voxel_map ){ //iterating over std::map gives the keys in order

        Eigen::Vector3i idx_3D=idx_1D_to_3D(idx_1D, m_grid_sizes); //assume that the grid is laid in order zyx where x is the fastest changing dimension and z is the slowest
        Eigen::Vector3d corner=m_voxel_size*idx_3D.cast<double>();
        corner-=m_grid_extents;
        Eigen::Vector3d voxel_center= corner.array() +m_voxel_size/2.0; //it's the center of the voxel
        // Eigen::Vector3f voxel_center_f=voxel_center.cast<float>();
        //divide also by the voxel size so the points inside the voxel have mean 0 and extent [-1,1]

        //add the point from this voxel
        for(size_t p_idx = 0; p_idx < voxel.nr_contained_points(); p_idx++){
            CHECK(nr_points_inserted<mat.rows()) << "Illegal access. nr_points_inserted is " << nr_points_inserted << " mat.rows() os " << mat.rows();
            Eigen::Vector3d point_normalized= voxel.m_points[p_idx] - voxel_center; //mean normalized
            point_normalized=point_normalized.array()/ (m_voxel_size/2.0); //divide by the size so the point normalized is in range [-1 , 1]

            mat(nr_points_inserted,0) = point_normalized(0);
            mat(nr_points_inserted,1) = point_normalized(1);
            mat(nr_points_inserted,2) = point_normalized(2);
            nr_points_inserted++;
        }
    }

    return mat;    

}

Eigen::VectorXi VoxelGrid::voxel_labels_gt(){
    Eigen::VectorXi mat( nr_active_voxels());
    mat.setZero();
    for( auto const& [idx_1D, voxel] : m_idx2voxel_map ){
        mat(voxel.id)=voxel.label_gt();
    }
    return mat;
}

Eigen::MatrixXi VoxelGrid::voxel_coords(){
    Eigen::MatrixXi mat( nr_active_voxels(), 3);
    mat.setZero();
    for( auto const& [idx_1D, voxel] : m_idx2voxel_map ){
        Eigen::Vector3i idx_3D=idx_1D_to_3D(idx_1D, m_grid_sizes); 
        mat.row(voxel.id)=idx_3D;
    }
    return mat;
}

Eigen::VectorXi VoxelGrid::voxel_ids_per_point(){
    Eigen::VectorXi mat( m_last_voxelized_cloud.V.rows() );
    //TODO
    int nr_points_inserted=0;
    int voxel_id=0;
    for( auto const& [idx_1D, voxel] : m_idx2voxel_map ){ //iterating over std::map gives the keys in order

        //add the point from this voxel
        for(size_t p_idx = 0; p_idx < voxel.nr_contained_points(); p_idx++){
            CHECK(nr_points_inserted<mat.rows()) << "Illegal access. nr_points_inserted is " << nr_points_inserted << " mat.rows() os " << mat.rows();
            mat(nr_points_inserted) = voxel_id;
            nr_points_inserted++;
        }

        voxel_id++;
    }

    return mat; 

}

Eigen::Vector3i VoxelGrid::grid_sizes(){
    return m_grid_sizes;
}


void VoxelGrid::clear(){
    m_idx2voxel_map.clear();
}


int VoxelGrid::find_enclosing_simplex(const Eigen::Vector3d& point){
    //the point is expressed in some world coordinates but we want to express it in the frame of the voxel grid, there the first voxel starts at 0,0,0
    //the m_grid_extents expresses how much does the grid extend from the origin of the world in directions x,y and z

    Eigen::Vector3d point_shifted=point + m_grid_extents;
    Eigen::Vector3i idx_3D= (point_shifted/ m_voxel_size).cast<int>(); //this automatically floors the point_shifted
    int v_idx= idx_3D_to_1D(idx_3D, m_grid_sizes);
    
    return v_idx;
}




MeshCore VoxelGrid::grid_dual(){
    TIME_SCOPE("grid_dual");

    MeshCore dual; //the dual represent each voxel as only the center and not as the 8 corners it actually has

    //get a vertex for each center of the voxel
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > centers_vec;
    std::vector<int > labels_dual_vec;
    for( auto const& [idx, voxel] : m_idx2voxel_map ){
        Eigen::Vector3i idx_3D=idx_1D_to_3D(idx, m_grid_sizes); //assume that the grid is laid in order zyx where x is the fastest changing dimension and z is the slowest
        Eigen::Vector3d corner=m_voxel_size*idx_3D.cast<double>();
        corner-=m_grid_extents;
        Eigen::Vector3d center= corner.array() +m_voxel_size/2.0; //it's the center of the voxel

        centers_vec.push_back(center);
        labels_dual_vec.push_back(voxel.label_gt() );
    }
    dual.V=vec2eigen(centers_vec);
    dual.L_gt=vec2eigen(labels_dual_vec);


    //sensible view settings
    dual.m_vis.m_show_mesh=false;
    dual.m_vis.m_show_points=true;
    dual.m_vis.m_color_type=+MeshColorType::SemanticGT;

    //set the labelmngr so that the viewer can use it to plot the semantic colors
    dual.m_label_mngr=m_last_voxelized_cloud.m_label_mngr->shared_from_this();

    return dual;

    // Scene::show(dual,"grid_dual");
}





// void VoxelGrid::show(){

//     TIME_SCOPE("show");

//     MeshCore dual; //the dual represent each voxel as only the center and not as the 8 corners it actually has

//     //get a vertex for each center of the voxel
//     std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > centers_vec;
//     std::vector<int > labels_dual_vec;
//     for(int i=0; i<m_voxels.size(); i++){
//         if(!m_voxels[i].m_is_empty){
//             Eigen::Vector3i idx_3D=idx_1D_to_3D(i, m_grid_sizes); //assume that the grid is laid in order zyx where x is the fastest changing dimension and z is the slowest

//             Eigen::Vector3d corner=m_voxel_size*idx_3D.cast<double>();
//             corner+=m_min_bounding_box;
//             Eigen::Vector3d center= corner.array() +m_voxel_size/2.0; //it's the center of the voxel

//             centers_vec.push_back(center);
//             labels_dual_vec.push_back(m_voxels[i].m_label_gt);
//         }
//     }
//     dual.V=vec2eigen(centers_vec);
//     dual.L_gt=vec2eigen(labels_dual_vec);


//     //sensible view settings
//     dual.m_vis.m_show_mesh=false;
//     dual.m_vis.m_show_points=true;
//     dual.m_vis.m_color_type=+MeshColorType::SemanticGT;

//     Scene::show(dual,"voxels_dual");




//     //grid repsresnting each voxel as the 8 corners it has
//     MeshCore grid; 

//     //get 8 vertices for each voxel
//     std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > points_vec;
//     std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > lines_vec;
//     std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > faces_vec;
//     std::vector<int > labels_vec;
//     for(int i=0; i<m_voxels.size(); i++){
//         if(!m_voxels[i].m_is_empty){
//             // VLOG(1) << "calling with grid sizes" << m_grid_sizes.transpose() << " and i " << i;
//             Eigen::Vector3i idx_3D=idx_1D_to_3D(i, m_grid_sizes); //assume that the grid is laid in order zyx where x is the fastest changing dimension and z is the slowest

//             Eigen::Vector3d corner=m_voxel_size*idx_3D.cast<double>();
//             corner+=m_min_bounding_box;
            
//             // //make all the lines
//             // Eigen::Vector2i line=Eigen::Vector2i(0,0).array()+i*1;
//             // VLOG(2) << "line is " << line << " i is " << i;
//             int n=points_vec.size()/8; //current number of voxels processed
//             lines_vec.push_back(Eigen::Vector2i(0,1).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(1,2).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(2,3).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(3,0).array()+n*8);
//             //close face
//             lines_vec.push_back(Eigen::Vector2i(4,5).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(5,6).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(6,7).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(7,4).array()+n*8);
//             //sides
//             lines_vec.push_back(Eigen::Vector2i(0,4).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(1,5).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(2,6).array()+n*8);
//             lines_vec.push_back(Eigen::Vector2i(3,7).array()+n*8);

//             //get the 8 vertices
//             points_vec.push_back(corner);
//             points_vec.push_back(corner + Eigen::Vector3d(m_voxel_size,0,0) );
//             points_vec.push_back(corner + Eigen::Vector3d(m_voxel_size,m_voxel_size,0) );
//             points_vec.push_back(corner + Eigen::Vector3d(0,m_voxel_size,0) );
//             //close plane
//             points_vec.push_back(corner + Eigen::Vector3d(0,0,m_voxel_size));
//             points_vec.push_back(corner + Eigen::Vector3d(m_voxel_size,0,m_voxel_size) );
//             points_vec.push_back(corner + Eigen::Vector3d(m_voxel_size,m_voxel_size,m_voxel_size) );
//             points_vec.push_back(corner + Eigen::Vector3d(0,m_voxel_size,m_voxel_size) );

//             //faces far plane
//             faces_vec.push_back(Eigen::Vector3i(0,2,1).array()+n*8);
//             faces_vec.push_back(Eigen::Vector3i(0,3,2).array()+n*8);
//             //close plane
//             faces_vec.push_back(Eigen::Vector3i(4,5,7).array()+n*8);
//             faces_vec.push_back(Eigen::Vector3i(7,5,6).array()+n*8);
//             //side lefs
//             faces_vec.push_back(Eigen::Vector3i(3,0,4).array()+n*8);
//             faces_vec.push_back(Eigen::Vector3i(3,4,7).array()+n*8);
//             //side right
//             faces_vec.push_back(Eigen::Vector3i(5,1,2).array()+n*8);
//             faces_vec.push_back(Eigen::Vector3i(6,5,2).array()+n*8);
//             //top
//             faces_vec.push_back(Eigen::Vector3i(7,6,2).array()+n*8);
//             faces_vec.push_back(Eigen::Vector3i(7,2,3).array()+n*8);
//             //bottom
//             faces_vec.push_back(Eigen::Vector3i(1,5,0).array()+n*8);
//             faces_vec.push_back(Eigen::Vector3i(0,5,4).array()+n*8);

//             //for all 8 vertices add the gt label corresponding to the voxel
//             labels_vec.push_back(m_voxels[i].m_label_gt);
//             labels_vec.push_back(m_voxels[i].m_label_gt);
//             labels_vec.push_back(m_voxels[i].m_label_gt);
//             labels_vec.push_back(m_voxels[i].m_label_gt);
//             labels_vec.push_back(m_voxels[i].m_label_gt);
//             labels_vec.push_back(m_voxels[i].m_label_gt);
//             labels_vec.push_back(m_voxels[i].m_label_gt);
//             labels_vec.push_back(m_voxels[i].m_label_gt);

    

//         }
//     }
//     grid.V=vec2eigen(points_vec);
//     grid.F=vec2eigen(faces_vec);
//     grid.E=vec2eigen(lines_vec);
//     grid.L_gt=vec2eigen(labels_vec);

//     // VLOG(1) << "grid is "<<grid;
 
//     //sensible view settings
//     grid.m_vis.m_show_mesh=true;
//     grid.m_vis.m_show_points=true;
//     // grid.m_vis.m_show_lines=true;
//     grid.m_vis.m_color_type=+MeshColorType::SemanticGT;

//     Scene::show(grid,"voxels_grid");
// }