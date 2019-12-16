#include <surfel_renderer/lattice/LatticeCPU_test.h>
#include <surfel_renderer/lattice/PermutohedralLatticeCPU_IMPL.h>

#include <surfel_renderer/utils/MiscUtils.h>
#include "surfel_renderer/utils/Profiler.h" 
#include "surfel_renderer/core/MeshCore.h"


#include <cstring>
#include <memory>

//for showing textures and meshes
#include "surfel_renderer/viewer/Gui.h"
#include "surfel_renderer/viewer/Scene.h"

using namespace er::utils;

LatticeCPU_test::LatticeCPU_test(const std::string config_file):
    m_spacial_sigma(5),
    m_color_sigma(0.15),
    m_surfel_scaling(1.0),
    m_rand_gen( new RandGenerator() ){

}


void LatticeCPU_test::compute(){

    //load an image
    // cv::Mat cv_img=cv::imread("/media/rosu/Data/data/imgs/lena.png");
    cv::Mat cv_img=cv::imread("/media/rosu/Data/phd/c_ws/src/surfel_renderer/data/dog_1.bmp");
    // cv::Mat cv_img=cv::imread("/media/rosu/Data/data/imgs/bw_1.jpg");
    // cv::Mat cv_img=cv::imread("/media/rosu/Data/data/imgs/bw_cats_1.jpg");
    cv::resize(cv_img, cv_img, cv::Size(), 0.25, 0.25);
    VLOG(1) << "img size is " << cv_img.rows << " " << cv_img.cols << std::endl;
    cv::Mat cv_img_float;
    cv_img.convertTo(cv_img_float, CV_32FC3, 1.0/255.0);
    cv::Mat cv_img_gray_float;
    cv::cvtColor( cv_img_float, cv_img_gray_float, CV_BGR2GRAY );
    VLOG(1) << " cv_img_gray_float type is " << type2string(cv_img_gray_float.type()) ;
    // VLOG(1) << "type of cv img float is " << cv_img_float.type();
    // VLOG(1) << "type of cv img gray float is " << cv_img_gray_float.type();


    //calculate ngf
    cv::Mat grad_x_32f, grad_y_32f;
    cv::Scharr( cv_img_gray_float, grad_x_32f, CV_32F, 1, 0 ,1.0/16);
    cv::Scharr( cv_img_gray_float, grad_y_32f, CV_32F, 0, 1, 1.0/16);
    VLOG(1) << " grad_x_32f type is " << type2string(grad_x_32f.type()) ;


    float epsilon=0.2;
    cv::Mat gx2, gy2;
    cv::multiply(grad_x_32f, grad_x_32f, gx2);
    cv::multiply(grad_y_32f, grad_y_32f, gy2);
    // cv::Mat mag=grad_x_32f*grad_x_32f+grad_y_32f*grad_y_32f;
    cv::Mat mag=gx2+gy2;
    cv::Mat normalization;
    cv::sqrt( mag + epsilon, normalization);
    cv::Mat normalized_grad;
    cv::divide(mag, normalization, normalized_grad);
    double min, max;
    cv::minMaxLoc(grad_x_32f, &min, &max);
    // cv::minMaxLoc(cv_img_gray_float, &min, &max);
    VLOG(1) << "min max is " << min << " " << max; 
   

    Scene::clear();
    bilateral_test(cv_img_float);
    // filter_positions_test(cv_img_float);
    // filter_positions_gray_test(cv_img_gray_float);
    // filter_positions_ngf_test(normalized_grad);
    



}



void LatticeCPU_test::bilateral_test(const cv::Mat& cv_img_float){
    
    CHECK(cv_img_float.type()==21) << "image has to be of three channels and float type"; 
    
    Gui::show(cv_img_float, "lattice_in");    


    int pd = 5;
    int vd = 3;
    int N = cv_img_float.rows * cv_img_float.cols;

    TIME_START("construct_positions_and_vals");
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    positions.resize(N,pd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(i,j);
            // positions(idx_insert,0)=i / m_spacial_sigma;
            // positions(idx_insert,1)=j / m_spacial_sigma;
            // positions(idx_insert,2)=rgb[0] / m_color_sigma;
            // positions(idx_insert,3)=rgb[1] / m_color_sigma;
            // positions(idx_insert,4)=rgb[2] / m_color_sigma;
            
            positions(idx_insert,0)=i;
            positions(idx_insert,1)=j;
            positions(idx_insert,2)=rgb[0];
            positions(idx_insert,3)=rgb[1];
            positions(idx_insert,4)=rgb[2];

        }
    }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
    values.resize(N,vd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(i,j);
            values(idx_insert,0)=rgb[0];
            values(idx_insert,1)=rgb[1];
            values(idx_insert,2)=rgb[2];

        }
    }
    TIME_END("construct_positions_and_vals");

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output;
    output.resize(N,vd);
    auto lattice = PermutohedralLatticeCPU_IMPL(pd, vd, N);
    lattice.set_sigmas({ {m_spacial_sigma,2}, {m_color_sigma,3} });
    TIME_START("filter");
    // lattice.filter(output, values, positions, false);
    output=lattice.filter(positions, values);
    VLOG(1) << "finished filtering";
    TIME_END("filter");


    cv::Mat cv_mat_out=cv_img_float.clone();
    std::memcpy(cv_mat_out.data, output.data(), N * vd * sizeof(float));
    Gui::show(cv_mat_out, "lattice_out");    

}

void LatticeCPU_test::bilateral_test_from_path(const std::string& cv_img_path){
    cv::Mat cv_mat= cv::imread(cv_img_path);
    cv::resize(cv_mat, cv_mat, cv::Size(), 0.25, 0.25);
    cv::Mat cv_img_float;
    cv_mat.convertTo(cv_img_float, CV_32FC3, 1.0/255.0);


    CHECK(cv_img_float.type()==21) << "image has to be of three channels and float type"; 
    
    Gui::show(cv_img_float, "lattice_in");    


    int pd = 5;
    int vd = 3;
    int N = cv_img_float.rows * cv_img_float.cols;

    TIME_START("construct_positions_and_vals");
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    positions.resize(N,pd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(i,j);
            // positions(idx_insert,0)=i / m_spacial_sigma;
            // positions(idx_insert,1)=j / m_spacial_sigma;
            // positions(idx_insert,2)=rgb[0] / m_color_sigma;
            // positions(idx_insert,3)=rgb[1] / m_color_sigma;
            // positions(idx_insert,4)=rgb[2] / m_color_sigma;
            
            positions(idx_insert,0)=i;
            positions(idx_insert,1)=j;
            positions(idx_insert,2)=rgb[0];
            positions(idx_insert,3)=rgb[1];
            positions(idx_insert,4)=rgb[2];

        }
    }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
    values.resize(N,vd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(i,j);
            values(idx_insert,0)=rgb[0];
            values(idx_insert,1)=rgb[1];
            values(idx_insert,2)=rgb[2];

        }
    }
    TIME_END("construct_positions_and_vals");

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output;
    output.resize(N,vd);
    auto lattice = PermutohedralLatticeCPU_IMPL(pd, vd, N);
    lattice.set_sigmas({ {m_spacial_sigma,2}, {m_color_sigma,3} });
    TIME_START("pl_filter");
    // lattice.filter(output, values, positions, false);
    // output=lattice.filter(positions, values);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions_scaled;

    TIME_START("scale positions");
    positions_scaled=lattice.compute_scaled_positions(positions); //scale by the sigmas 
    TIME_END("scale positions");

    TIME_START("splat");
    lattice.splat(positions_scaled,values, "barycentric", false);
    TIME_END("splat");

    TIME_START("blur");
    lattice.blur();
    TIME_END("blur");

    TIME_START("slice");
    output=lattice.slice(positions_scaled, false);
    TIME_END("slice");


    VLOG(1) << "finished filtering";
    TIME_END("pl_filter");


    cv::Mat cv_mat_out=cv_img_float.clone();
    std::memcpy(cv_mat_out.data, output.data(), N * vd * sizeof(float));
    Gui::show(cv_mat_out, "lattice_out");    

}

void LatticeCPU_test::filter_positions_test(const cv::Mat& cv_img_float){
    
    CHECK(cv_img_float.type()==21) << "image has to be of three channels and float type"; 
    
    Gui::show(cv_img_float, "lattice_in");    


    int pd = 5; //spacial and rgb
    int vd = 5; //spacial and rgb (the spacial is just for display purposes)
    int N = cv_img_float.rows * cv_img_float.cols;

    TIME_START("construct_positions_and_vals");
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    positions.resize(N,pd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(i,j);
            positions(idx_insert,0)=j;
            positions(idx_insert,1)=i;
            positions(idx_insert,2)=rgb[0];
            positions(idx_insert,3)=rgb[1];
            positions(idx_insert,4)=rgb[2];

        }
    }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
    values.resize(N,vd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(i,j);
            values(idx_insert,0)=j;
            values(idx_insert,1)=i;
            values(idx_insert,2)=rgb[0];
            values(idx_insert,3)=rgb[1];
            values(idx_insert,4)=rgb[2];

        }
    }
    TIME_END("construct_positions_and_vals");

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output;
    output.resize(N,vd);
    auto lattice = PermutohedralLatticeCPU_IMPL(pd, vd, N);
    lattice.set_sigmas({ {m_spacial_sigma,2}, {m_color_sigma,3} });
    TIME_START("filter");
    output=lattice.filter(positions, values);
    TIME_END("filter");


    //out
    cv::Mat cv_img_out=cv_img_float.clone();
    for(int i=0; i<cv_img_out.rows; i++){
        for(int j=0; j<cv_img_out.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv_img_out.at<cv::Vec3f>(i,j)[0]=output(idx_insert,2);
            cv_img_out.at<cv::Vec3f>(i,j)[1]=output(idx_insert,3);
            cv_img_out.at<cv::Vec3f>(i,j)[2]=output(idx_insert,4);

        }
    }
    Gui::show(cv_img_out, "lattice_out");    


    // //create a mesh with vertices at x.y of the image and color being the rgb color 
    // MeshCore mesh;
    // mesh.V.resize(N,3);
    // mesh.V.setZero();
    // mesh.C.resize(N,3);
    // mesh.C.setZero();
    // for(int i = 0; i < N; i++){
    //     mesh.V(i,0)=output(i,0);
    //     mesh.V(i,1)=-output(i,1);
    //     // mesh.V(i,2)=lattice.hashTable.values(i,vd); //the z will be the weight stored in the vertex which correspond to how many pixels splatted to it
    //     mesh.C(i,0)=output(i,2);
    //     mesh.C(i,1)=output(i,3);
    //     mesh.C(i,2)=output(i,4);
    // }
    // mesh.m_vis.m_show_points=true;
    // mesh.m_vis.m_show_mesh=false;


    //create a mesh with vertices at x.y of the image and color being the rgb color but this time only the vertices of the lattice 
    int nr_verts_lattice=lattice.hashTable.filled;
    MeshCore mesh;
    mesh.V.resize(nr_verts_lattice,3);
    mesh.V.setZero();
    mesh.C.resize(nr_verts_lattice,3);
    mesh.C.setZero();
    for(int i = 0; i < nr_verts_lattice; i++){
        float weight=lattice.hashTable.values(i,vd);
        Eigen::VectorXf mean=lattice.hashTable.m_cov_matrices[i].mean();
         mesh.V(i,0)=mean.x();
         mesh.V(i,1)=-mean.y();
        // mesh.V(i,0)=lattice.hashTable.values(i,0) / weight;
        // mesh.V(i,1)=-lattice.hashTable.values(i,1) / weight;
        // mesh.V(i,2)=weight*50; //the z will be the weight stored in the vertex which correspond to how many pixels splatted to it
        mesh.C(i,0)=lattice.hashTable.values(i,2) / weight;
        mesh.C(i,1)=lattice.hashTable.values(i,3) / weight;
        mesh.C(i,2)=lattice.hashTable.values(i,4) / weight;
    }
    //make some radii for the surfels
    mesh.NV.resize(nr_verts_lattice,3);
    mesh.NV.setZero();
    mesh.V_tangent_u.resize(nr_verts_lattice,3);
    mesh.V_tangent_u.setZero();
    mesh.V_length_v.resize(nr_verts_lattice,1);
    mesh.V_length_v.setZero();
    // for(int i = 0; i < nr_verts_lattice; i++){
    //     float weight=lattice.hashTable.values(i,vd);
    //     int nr_spacial_dim=2; 
    //     float radius=std::pow(weight,1.0/nr_spacial_dim); // the radius is not linear on the weight, because otherwise when we have only once vertex on which all pixels splat the weight is too big and it should be scaled by however dimension it has. Think that the radius is like meters squared and the weight is only meters which scale linearly
    //     radius*=5;
    //     //normals will be pointing toward the camera 
    //     mesh.NV(i,0)=0.0;
    //     mesh.NV(i,1)=0.0;
    //     mesh.NV(i,2)=1.0;
    //     //tangent is pointing upwards 
    //     mesh.V_tangent_u(i,0)=0.0*radius;
    //     mesh.V_tangent_u(i,1)=1.0*radius;
    //     mesh.V_tangent_u(i,2)=0.0*radius;
    //     //the other lenght is the same as the radius
    //     mesh.V_length_v(i)=radius; 
    // }

    //base the surfels on the covariance matrices calculated
    for(int i = 0; i < nr_verts_lattice; i++){
        Eigen::MatrixXf eigen_vecs=lattice.hashTable.m_cov_matrices[i].eigenvecs();
        Eigen::VectorXf eigen_vals=lattice.hashTable.m_cov_matrices[i].eigenvals();
        Eigen::VectorXf mean=lattice.hashTable.m_cov_matrices[i].mean();
        int nr_samples=lattice.hashTable.m_cov_matrices[i].nr_samples();
        if(nr_samples<5){
            continue;
        }

        // VLOG(1) << "eigen_vecs is \n" << eigen_vecs ; 
        // VLOG(1) << "eigen_vals is \n" << eigen_vals ; 
        // VLOG(1) << "mean is \n" << mean ; 
        // int nr_spacial_dim=2; 
        // float radius=std::pow(weight,1.0/nr_spacial_dim); // the radius is not linear on the weight, because otherwise when we have only once vertex on which all pixels splat the weight is too big and it should be scaled by however dimension it has. Think that the radius is like meters squared and the weight is only meters which scale linearly
        // radius*=5;
        //normals will be pointing toward the camera 
        mesh.NV(i,0)=0.0;
        mesh.NV(i,1)=0.0;
        mesh.NV(i,2)=1.0;
        float scaling=1.0;
        //tangent is pointing upwards 
        // Eigen::VectorXf biggest_eigenvector=eigen_vecs.rightCols(1) * sqrt(eigen_vals.z())*scaling;
        Eigen::VectorXf biggest_eigenvector=eigen_vecs.rightCols(1) *sqrt(eigen_vals.z())*scaling;
        mesh.V_tangent_u.row(i)=biggest_eigenvector.cast<double>();
        //the other lenght is the same as the middle eigenvector
        mesh.V_length_v(i)=sqrt(eigen_vals.y())*scaling; 
    }

    // VLOG(1) << mesh.V_tangent_u;
    // VLOG(1) << mesh.V_length_v;


    mesh.m_vis.m_show_points=false;
    mesh.m_vis.m_show_mesh=false;
    mesh.m_vis.m_show_surfels=true;

    Scene::show(mesh,"lattice_mesh_img");


    

}



void LatticeCPU_test::filter_positions_gray_test(const cv::Mat& cv_img_float){

    CHECK(cv_img_float.type()==5) << "image has to be of one channels and float type"; 
    
    Gui::show(cv_img_float, "lattice_in");    


    int pd = 3; //spacial and intensity
    int vd = 3; //spacial and intensity (the spacial is just for display purposes)
    int N = cv_img_float.rows * cv_img_float.cols;

    TIME_START("construct_positions_and_vals");
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    positions.resize(N,pd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            float intensity=cv_img_float.at<float>(i,j);
            positions(idx_insert,0)=j / m_spacial_sigma;
            positions(idx_insert,1)=i / m_spacial_sigma;
            positions(idx_insert,2)=intensity / m_color_sigma;
            // positions(idx_insert,3)=rgb[1] / m_color_sigma;
            // positions(idx_insert,4)=rgb[2] / m_color_sigma;

        }
    }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
    values.resize(N,vd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            float intensity=cv_img_float.at<float>(i,j);
            values(idx_insert,0)=j;
            values(idx_insert,1)=i;
            values(idx_insert,2)=intensity;
            // values(idx_insert,3)=rgb[1];
            // values(idx_insert,4)=rgb[2];

        }
    }
    TIME_END("construct_positions_and_vals");

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output;
    output.resize(N,vd);
    auto lattice = PermutohedralLatticeCPU_IMPL(pd, vd, N);
    TIME_START("filter");
    lattice.filter(output, values, positions, false);
    TIME_END("filter");


    //out
    cv::Mat cv_img_out=cv_img_float.clone();
    for(int i=0; i<cv_img_out.rows; i++){
        for(int j=0; j<cv_img_out.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv_img_out.at<float>(i,j)=output(idx_insert,2); //rows 0 and 1 are the positions in x and i

        }
    }
    Gui::show(cv_img_out, "lattice_out");    


    //show enclosing simplex by finding idx in the hastable for each position in positions vector and coloring it differently for each one
    cv::Mat cv_img_simplices=cv::Mat(cv_img_float.rows, cv_img_float.cols, CV_32FC3);
    srand (0); //so every time we create the random colors we get the same colors
    Eigen::MatrixXd random_colors(lattice.hashTable.filled,3); // we assign a random color for each of the vertex of the lattice
    for(int i = 0; i < random_colors.rows(); i++){
        random_colors.row(i) = random_color(m_rand_gen);
    }
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_linear=i*cv_img_float.cols + j;

            int closest_vertex_idx=lattice.m_closest_vertex_idxs(idx_linear);
            Eigen::VectorXd color_of_vertex=random_colors.row(closest_vertex_idx);
            cv_img_simplices.at<cv::Vec3f>(i,j)[0]=color_of_vertex.x();
            cv_img_simplices.at<cv::Vec3f>(i,j)[1]=color_of_vertex.y();
            cv_img_simplices.at<cv::Vec3f>(i,j)[2]=color_of_vertex.z();
        }
    }
    Gui::show(cv_img_simplices, "cv_img_simplices");
    





    // //create a mesh with vertices at x.y of the image and color being the rgb color 
    // MeshCore mesh;
    // mesh.V.resize(N,3);
    // mesh.V.setZero();
    // mesh.C.resize(N,3);
    // mesh.C.setZero();
    // for(int i = 0; i < N; i++){
    //     mesh.V(i,0)=output(i,0);
    //     mesh.V(i,1)=-output(i,1);
    //     // mesh.V(i,2)=lattice.hashTable.values(i,vd); //the z will be the weight stored in the vertex which correspond to how many pixels splatted to it
    //     mesh.C(i,0)=output(i,2);
    //     mesh.C(i,1)=output(i,3);
    //     mesh.C(i,2)=output(i,4);
    // }
    // mesh.m_vis.m_show_points=true;
    // mesh.m_vis.m_show_mesh=false;


    //create a mesh with vertices at x.y of the image and color being the rgb color but this time only the vertices of the lattice 
    int nr_verts_lattice=lattice.hashTable.filled;
    MeshCore mesh;
    mesh.V.resize(nr_verts_lattice,3);
    mesh.V.setZero();
    mesh.C.resize(nr_verts_lattice,3);
    mesh.C.setZero();
    for(int i = 0; i < nr_verts_lattice; i++){
        float weight=lattice.hashTable.values(i,vd);
        Eigen::VectorXf mean=lattice.hashTable.m_cov_matrices[i].mean();
         mesh.V(i,0)=mean.x();
         mesh.V(i,1)=-mean.y();
        // mesh.V(i,0)=lattice.hashTable.values(i,0) / weight;
        // mesh.V(i,1)=-lattice.hashTable.values(i,1) / weight;
        // mesh.V(i,2)=weight*50; //the z will be the weight stored in the vertex which correspond to how many pixels splatted to it
        mesh.C(i,0)=lattice.hashTable.values(i,2) / weight;
        mesh.C(i,1)=lattice.hashTable.values(i,2) / weight;
        mesh.C(i,2)=lattice.hashTable.values(i,2) / weight;
    }
    //make some radii for the surfels
    mesh.NV.resize(nr_verts_lattice,3);
    mesh.NV.setZero();
    mesh.V_tangent_u.resize(nr_verts_lattice,3);
    mesh.V_tangent_u.setZero();
    mesh.V_length_v.resize(nr_verts_lattice,1);
    mesh.V_length_v.setZero();
    // for(int i = 0; i < nr_verts_lattice; i++){
    //     float weight=lattice.hashTable.values(i,vd);
    //     int nr_spacial_dim=2; 
    //     float radius=std::pow(weight,1.0/nr_spacial_dim); // the radius is not linear on the weight, because otherwise when we have only once vertex on which all pixels splat the weight is too big and it should be scaled by however dimension it has. Think that the radius is like meters squared and the weight is only meters which scale linearly
    //     radius*=5;
    //     //normals will be pointing toward the camera 
    //     mesh.NV(i,0)=0.0;
    //     mesh.NV(i,1)=0.0;
    //     mesh.NV(i,2)=1.0;
    //     //tangent is pointing upwards 
    //     mesh.V_tangent_u(i,0)=0.0*radius;
    //     mesh.V_tangent_u(i,1)=1.0*radius;
    //     mesh.V_tangent_u(i,2)=0.0*radius;
    //     //the other lenght is the same as the radius
    //     mesh.V_length_v(i)=radius; 
    // }

    //base the surfels on the covariance matrices calculated
    for(int i = 0; i < nr_verts_lattice; i++){
        Eigen::Matrix3f eigen_vecs=lattice.hashTable.m_cov_matrices[i].eigenvecs();
        Eigen::Vector3f eigen_vals=lattice.hashTable.m_cov_matrices[i].eigenvals();
        Eigen::VectorXf mean=lattice.hashTable.m_cov_matrices[i].mean();
        int nr_samples=lattice.hashTable.m_cov_matrices[i].nr_samples();
        if(nr_samples<7){
            continue;
        }

        // VLOG(1) << "eigen_vecs is \n" << eigen_vecs ; 
        // VLOG(1) << "eigen_vals is \n" << eigen_vals ; 
        // VLOG(1) << "mean is \n" << mean ; 
        // int nr_spacial_dim=2; 
        // float radius=std::pow(weight,1.0/nr_spacial_dim); // the radius is not linear on the weight, because otherwise when we have only once vertex on which all pixels splat the weight is too big and it should be scaled by however dimension it has. Think that the radius is like meters squared and the weight is only meters which scale linearly
        // radius*=5;
        //normals will be pointing toward the camera 
        mesh.NV(i,0)=0.0;
        mesh.NV(i,1)=0.0;
        mesh.NV(i,2)=1.0;
        float scaling=m_surfel_scaling;
        //tangent is pointing upwards 
        Eigen::VectorXf biggest_eigenvector=eigen_vecs.col(2) * sqrt(eigen_vals.z())*scaling;
        // Eigen::VectorXf biggest_eigenvector=eigen_vecs.col(2) * sqrt(eigen_vals.z())*scaling;
        // Eigen::VectorXf biggest_eigenvector=eigen_vecs.col(2) * eigen_vals.z()*scaling;
        mesh.V_tangent_u.row(i)=biggest_eigenvector.cast<double>();
        //the other lenght is the same as the middle eigenvector
        mesh.V_length_v(i)=sqrt(eigen_vals.y())*scaling; 


        // //for each vertex we add a small mesh to show the axis 
        // //make a mesh with 4 points, 1 in the mean and 3 corresponding to the 3 eigenvectors
        // if(eigen_vals.y()/eigen_vals.z()*100 < 0.1 ){ //if y is less then 1% of z
        //     int nr_samples=lattice.hashTable.m_cov_matrices[i].nr_samples();
        //     VLOG(1) << "nr _samples is " << nr_samples;
        //     Eigen::Matrix3d eigen_vecs=lattice.hashTable.m_cov_matrices[i].eigenvecs().cast<double>();
        //     Eigen::Vector3d eigen_vals=lattice.hashTable.m_cov_matrices[i].eigenvals().cast<double>();
        //     MeshCore mesh_cov;
        //     mesh_cov.V.resize(4,3);
        //     // mesh_cov.V.row(0)=cov_matrix_estimator.mean().cast<double>() ;
        //     mesh_cov.V.row(0).setZero() ;
        //     mesh_cov.V.row(1)=eigen_vecs.col(0) * sqrt(eigen_vals.x()) ;
        //     mesh_cov.V.row(2)=eigen_vecs.col(1) * sqrt(eigen_vals.y()) ;
        //     mesh_cov.V.row(3)=eigen_vecs.col(2) * sqrt(eigen_vals.z()) ;
        //     mesh_cov.E.resize(3,2);
        //     mesh_cov.E.row(0) << 0,1;
        //     mesh_cov.E.row(1) << 0,2;
        //     mesh_cov.E.row(2) << 0,3;
        //     mesh_cov.C.resize(4,3);
        //     mesh_cov.C.row(0) << 0,0,0; //center point is black
        //     mesh_cov.C.row(1) << 1,0,0; // smalles eigenvector is red
        //     mesh_cov.C.row(2) << 1,1,0; // second smallest is yellow
        //     mesh_cov.C.row(3) << 0,1,0; // biggest eigenvector is green
        //     Eigen::Affine3d transform;
        //     transform.setIdentity();
        //     Eigen::Vector3d t; 
        //     t.setZero();
        //     Eigen::Vector3d mean=lattice.hashTable.m_cov_matrices[i].mean().cast<double>();
        //     t << mean.x(), -mean.y(), 0.0;
        //     transform.translate(t);
        //     mesh_cov.apply_transform(transform, true);

        //     mesh_cov.m_vis.m_show_mesh=false;
        //     mesh_cov.m_vis.m_show_lines=true;
        //     mesh_cov.m_vis.m_show_points=true;
        //     Scene::show(mesh_cov, "pca"+std::to_string(i));
        // }


    }

    // VLOG(1) << mesh.V_tangent_u;
    // VLOG(1) << mesh.V_length_v;


    mesh.m_vis.m_show_points=false;
    mesh.m_vis.m_show_mesh=false;
    mesh.m_vis.m_show_surfels=true;

    Scene::show(mesh,"lattice_mesh_img");


    

}


void LatticeCPU_test::filter_positions_ngf_test(const cv::Mat& cv_img_float){

    CHECK(cv_img_float.type()==5) << "image has to be of one channels and float type"; 
    
    Gui::show(cv_img_float, "lattice_in");    


    int pd = 3; //spacial and normalized gradient magnitude
    int vd = 3; //spacial and normalized gradietn magnitud (the spacial is just for display purposes)
    int N = cv_img_float.rows * cv_img_float.cols;

    TIME_START("construct_positions_and_vals");
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    positions.resize(N,pd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            float intensity=cv_img_float.at<float>(i,j);
            positions(idx_insert,0)=j / m_spacial_sigma;
            positions(idx_insert,1)=i / m_spacial_sigma;
            positions(idx_insert,2)=intensity / m_color_sigma;
            // positions(idx_insert,3)=rgb[1] / m_color_sigma;
            // positions(idx_insert,4)=rgb[2] / m_color_sigma;

        }
    }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
    values.resize(N,vd);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            float intensity=cv_img_float.at<float>(i,j);
            values(idx_insert,0)=j;
            values(idx_insert,1)=i;
            values(idx_insert,2)=intensity;
            // values(idx_insert,3)=rgb[1];
            // values(idx_insert,4)=rgb[2];

        }
    }
    TIME_END("construct_positions_and_vals");

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output;
    output.resize(N,vd);
    auto lattice = PermutohedralLatticeCPU_IMPL(pd, vd, N);
    TIME_START("filter");
    lattice.filter(output, values, positions, false);
    TIME_END("filter");


    //out
    cv::Mat cv_img_out=cv_img_float.clone();
    for(int i=0; i<cv_img_out.rows; i++){
        for(int j=0; j<cv_img_out.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            cv_img_out.at<float>(i,j)=output(idx_insert,2); //rows 0 and 1 are the positions in x and i

        }
    }
    Gui::show(cv_img_out, "lattice_out");    


    //show enclosing simplex by finding idx in the hastable for each position in positions vector and coloring it differently for each one
    cv::Mat cv_img_simplices=cv::Mat(cv_img_float.rows, cv_img_float.cols, CV_32FC3);
    srand (0); //so every time we create the random colors we get the same colors
    Eigen::MatrixXd random_colors(lattice.hashTable.filled,3); // we assign a random color for each of the vertex of the lattice
    for(int i = 0; i < random_colors.rows(); i++){
        random_colors.row(i) = random_color(m_rand_gen);
    }
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_linear=i*cv_img_float.cols + j;

            int closest_vertex_idx=lattice.m_closest_vertex_idxs(idx_linear);
            Eigen::VectorXd color_of_vertex=random_colors.row(closest_vertex_idx);
            cv_img_simplices.at<cv::Vec3f>(i,j)[0]=color_of_vertex.x();
            cv_img_simplices.at<cv::Vec3f>(i,j)[1]=color_of_vertex.y();
            cv_img_simplices.at<cv::Vec3f>(i,j)[2]=color_of_vertex.z();
        }
    }
    Gui::show(cv_img_simplices, "cv_img_simplices");
    





    // //create a mesh with vertices at x.y of the image and color being the rgb color 
    // MeshCore mesh;
    // mesh.V.resize(N,3);
    // mesh.V.setZero();
    // mesh.C.resize(N,3);
    // mesh.C.setZero();
    // for(int i = 0; i < N; i++){
    //     mesh.V(i,0)=output(i,0);
    //     mesh.V(i,1)=-output(i,1);
    //     // mesh.V(i,2)=lattice.hashTable.values(i,vd); //the z will be the weight stored in the vertex which correspond to how many pixels splatted to it
    //     mesh.C(i,0)=output(i,2);
    //     mesh.C(i,1)=output(i,3);
    //     mesh.C(i,2)=output(i,4);
    // }
    // mesh.m_vis.m_show_points=true;
    // mesh.m_vis.m_show_mesh=false;


    //create a mesh with vertices at x.y of the image and color being the rgb color but this time only the vertices of the lattice 
    int nr_verts_lattice=lattice.hashTable.filled;
    MeshCore mesh;
    mesh.V.resize(nr_verts_lattice,3);
    mesh.V.setZero();
    mesh.C.resize(nr_verts_lattice,3);
    mesh.C.setZero();
    for(int i = 0; i < nr_verts_lattice; i++){
        float weight=lattice.hashTable.values(i,vd);
        Eigen::VectorXf mean=lattice.hashTable.m_cov_matrices[i].mean();
         mesh.V(i,0)=mean.x()*m_spacial_sigma;
         mesh.V(i,1)=-mean.y()*m_spacial_sigma;
        // mesh.V(i,0)=lattice.hashTable.values(i,0) / weight;
        // mesh.V(i,1)=-lattice.hashTable.values(i,1) / weight;
        // mesh.V(i,2)=weight*50; //the z will be the weight stored in the vertex which correspond to how many pixels splatted to it
        mesh.C(i,0)=lattice.hashTable.values(i,2) / weight;
        mesh.C(i,1)=lattice.hashTable.values(i,2) / weight;
        mesh.C(i,2)=lattice.hashTable.values(i,2) / weight;
    }
    //make some radii for the surfels
    mesh.NV.resize(nr_verts_lattice,3);
    mesh.NV.setZero();
    mesh.V_tangent_u.resize(nr_verts_lattice,3);
    mesh.V_tangent_u.setZero();
    mesh.V_length_v.resize(nr_verts_lattice,1);
    mesh.V_length_v.setZero();
    // for(int i = 0; i < nr_verts_lattice; i++){
    //     float weight=lattice.hashTable.values(i,vd);
    //     int nr_spacial_dim=2; 
    //     float radius=std::pow(weight,1.0/nr_spacial_dim); // the radius is not linear on the weight, because otherwise when we have only once vertex on which all pixels splat the weight is too big and it should be scaled by however dimension it has. Think that the radius is like meters squared and the weight is only meters which scale linearly
    //     radius*=5;
    //     //normals will be pointing toward the camera 
    //     mesh.NV(i,0)=0.0;
    //     mesh.NV(i,1)=0.0;
    //     mesh.NV(i,2)=1.0;
    //     //tangent is pointing upwards 
    //     mesh.V_tangent_u(i,0)=0.0*radius;
    //     mesh.V_tangent_u(i,1)=1.0*radius;
    //     mesh.V_tangent_u(i,2)=0.0*radius;
    //     //the other lenght is the same as the radius
    //     mesh.V_length_v(i)=radius; 
    // }

    //base the surfels on the covariance matrices calculated
    for(int i = 0; i < nr_verts_lattice; i++){
        Eigen::MatrixXf eigen_vecs=lattice.hashTable.m_cov_matrices[i].eigenvecs();
        Eigen::VectorXf eigen_vals=lattice.hashTable.m_cov_matrices[i].eigenvals();
        Eigen::VectorXf mean=lattice.hashTable.m_cov_matrices[i].mean();
        int nr_samples=lattice.hashTable.m_cov_matrices[i].nr_samples();
        // if(nr_samples<5){
        //     continue;
        // }

        // VLOG(1) << "eigen_vecs is \n" << eigen_vecs ; 
        // VLOG(1) << "eigen_vals is \n" << eigen_vals ; 
        // VLOG(1) << "mean is \n" << mean ; 
        // int nr_spacial_dim=2; 
        // float radius=std::pow(weight,1.0/nr_spacial_dim); // the radius is not linear on the weight, because otherwise when we have only once vertex on which all pixels splat the weight is too big and it should be scaled by however dimension it has. Think that the radius is like meters squared and the weight is only meters which scale linearly
        // radius*=5;
        //normals will be pointing toward the camera 
        mesh.NV(i,0)=0.0;
        mesh.NV(i,1)=0.0;
        mesh.NV(i,2)=1.0;
        float scaling=10.0;
        //tangent is pointing upwards 
        Eigen::VectorXf biggest_eigenvector=eigen_vecs.col(2) * sqrt(eigen_vals.z())*scaling;
        mesh.V_tangent_u.row(i)=biggest_eigenvector.cast<double>();
        //the other lenght is the same as the middle eigenvector
        mesh.V_length_v(i)=sqrt(eigen_vals.y())*scaling; 
    }

    // VLOG(1) << mesh.V_tangent_u;
    // VLOG(1) << mesh.V_length_v;


    mesh.m_vis.m_show_points=false;
    mesh.m_vis.m_show_mesh=false;
    mesh.m_vis.m_show_surfels=true;

    Scene::show(mesh,"lattice_mesh_img");


    

}


