#include <surfel_renderer/lattice/LatticeGPU_test.h>

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it

#include <surfel_renderer/lattice/Lattice.cuh>
#include <surfel_renderer/utils/MiscUtils.h>
#include "surfel_renderer/utils/Profiler.h" 
#include "surfel_renderer/core/MeshCore.h"


#include <cstring>
#include <memory>

//for showing textures and meshes
#include "surfel_renderer/viewer/Gui.h"
#include "surfel_renderer/viewer/Scene.h"


//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason

using namespace er::utils;
using torch::Tensor;

LatticeGPU_test::LatticeGPU_test(const std::string config_file):
    m_lattice( Lattice::create(config_file) ),
    m_rand_gen( new RandGenerator() )
    {


}


void LatticeGPU_test::bilateral_test_from_path(const std::string& cv_img_path){
    cv::Mat cv_mat= cv::imread(cv_img_path);
    // cv::resize(cv_mat, cv_mat, cv::Size(), 0.25, 0.25);
    cv::Mat cv_mat_gray;
    cv::cvtColor(cv_mat, cv_mat_gray, CV_BGR2GRAY);
    cv::Mat cv_img_float_rgb;
    cv::Mat cv_img_float_gray;
    cv_mat_gray.convertTo(cv_img_float_gray, CV_32FC1, 1.0/255.0);
    cv_mat.convertTo(cv_img_float_rgb, CV_32FC3, 1.0/255.0);


    //compute positions and values for this image
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
    compute_positions_and_values(positions, values, cv_img_float_rgb);
    Tensor positions_tensor=eigen2tensor(positions);
    Tensor values_tensor=eigen2tensor(values);
    VLOG(1) << "computed nr of positions " << positions.rows();

    
    // Tensor output_values_tensor = m_lattice->bilateral_filter(positions_tensor, values_tensor);
    m_lattice->begin_splat();
    // m_lattice->splat_standalone(positions_tensor, values_tensor);
    // m_lattice->blur_standalone();
    // Tensor output_values_tensor = m_lattice->slice_standalone_no_precomputation(positions_tensor, values_tensor);

    // //output values after filtering to cv_mat and show
    // EigenMatrixXfRowMajor output_values_eigen=tensor2eigen(output_values_tensor);
    // cv::Mat cv_mat_filtered= eigen2mat(output_values_eigen, cv_mat.rows, cv_mat.cols);
    // Gui::show(cv_mat_filtered, "cv_mat_filtered");

    // //view the values 
    // cv::Mat hashed_values_mat = values_as_img();
    // Gui::show(hashed_values_mat, "hashed_values_mat");

}

// void LatticeGPU_test::bilateral_recursive_test_from_path(const std::string& cv_img_path){

//     cv::Mat cv_mat= cv::imread(cv_img_path);
//     // cv::resize(cv_mat, cv_mat, cv::Size(), 0.25, 0.25);
//     cv::Mat cv_mat_gray;
//     cv::cvtColor(cv_mat, cv_mat_gray, CV_BGR2GRAY);
//     cv::Mat cv_img_float_rgb;
//     cv::Mat cv_img_float_gray;
//     cv_mat_gray.convertTo(cv_img_float_gray, CV_32FC1, 1.0/255.0);
//     cv_mat.convertTo(cv_img_float_rgb, CV_32FC3, 1.0/255.0);

//     int nr_iters=15;
//     for(int i=0; i<nr_iters; i++){
//         //compute positions and values for this image
//         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
//         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
//         compute_positions_and_values(positions, values, cv_img_float_rgb);
//         Tensor positions_tensor=eigen2tensor(positions);
//         Tensor values_tensor=eigen2tensor(values);
//         VLOG(1) << "computed nr of positions " << positions.rows();


        
//         // Tensor output_values_tensor = m_lattice->bilateral_filter(positions_tensor, values_tensor);
//         m_lattice->begin_splat();
//         m_lattice->splat_standalone(positions_tensor, values_tensor);
//         m_lattice->blur_standalone();
//         Tensor output_values_tensor = m_lattice->slice_standalone_no_precomputation(positions_tensor, values_tensor);

//         //output values after filtering to cv_mat and show
//         EigenMatrixXfRowMajor output_values_eigen=tensor2eigen(output_values_tensor);
//         cv::Mat cv_mat_filtered= eigen2mat(output_values_eigen, cv_mat.rows, cv_mat.cols);
//         Gui::show(cv_mat_filtered, "cv_mat_filtered");

//         //setup the new output as the new cv_img_float_rgb
//         cv_img_float_rgb=cv_mat_filtered;
//     }


//     //view the values 
//     cv::Mat hashed_values_mat = values_as_img();
//     Gui::show(hashed_values_mat, "hashed_values_mat");

// }



// void LatticeGPU_test::bilateral_incremental_test_from_path(const std::string& cv_img_path){

//     cv::Mat cv_mat= cv::imread(cv_img_path);
//     // cv::resize(cv_mat, cv_mat, cv::Size(), 0.25, 0.25);
//     cv::Mat cv_mat_gray;
//     cv::cvtColor(cv_mat, cv_mat_gray, CV_BGR2GRAY);
//     cv::Mat cv_img_float_rgb;
//     cv::Mat cv_img_float_gray;
//     cv_mat_gray.convertTo(cv_img_float_gray, CV_32FC1, 1.0/255.0);
//     cv_mat.convertTo(cv_img_float_rgb, CV_32FC3, 1.0/255.0);

//     int nr_iters=10;
//     m_lattice->begin_splat();
//     //get positions and values in batches so to speak, so that we get X nr of rows together as positions and vals until we get all the image splatted
//     int start_row=0;
//     int row_increment=cv_mat.rows/nr_iters;
//     int end_row=row_increment;
//     for(int i=0; i<nr_iters; i++){
//         //compute positions and values for this image
//         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
//         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
//         // srand(0);
//         // compute_positions_and_values_from_random_pixels(positions, values, cv_img_float_rgb);
//         // compute_positions_and_values(positions, values, cv_img_float_rgb);
//         compute_positions_and_values_in_row_range(positions, values, cv_img_float_rgb, start_row, end_row);
//         Tensor positions_tensor=eigen2tensor(positions);
//         Tensor values_tensor=eigen2tensor(values);
//         VLOG(1) << "computed nr of positions " << positions.rows();

//         //show the image of the values we are about to splat 
//         // cv::Mat cv_mat_to_splat= eigen2mat(values, cv_mat.rows, cv_mat.cols);
//         // Gui::show(cv_mat_to_splat, "cv_mat_to_splat");
        
//         m_lattice->splat_standalone(positions_tensor, values_tensor);

//         start_row+=row_increment;
//         end_row+=row_increment;
//     }

//     //slice at the positons of all the original pixels
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
//     compute_positions_and_values(positions, values, cv_img_float_rgb);
//     Tensor positions_tensor=eigen2tensor(positions);
//     Tensor values_tensor=eigen2tensor(values);

//     m_lattice->blur_standalone();
//     Tensor output_values_tensor = m_lattice->slice_standalone_no_precomputation(positions_tensor, values_tensor);

//     //output values after filtering to cv_mat and show
//     EigenMatrixXfRowMajor output_values_eigen=tensor2eigen(output_values_tensor);
//     cv::Mat cv_mat_filtered= eigen2mat(output_values_eigen, cv_mat.rows, cv_mat.cols);
//     Gui::show(cv_mat_filtered, "cv_mat_filtered");

//     //view the values 
//     cv::Mat hashed_values_mat = values_as_img();
//     Gui::show(hashed_values_mat, "hashed_values_mat");


//     // //check if the assignment of a tensor to another one makes a deep copy or not. Assignment indeed does nto copy any data. We would have ot use clone for that
//     // //make a tensor 
//     // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_mat;
//     // eigen_mat.resize(2,2);
//     // eigen_mat<<1,2,3,4;
//     // Tensor tens=eigen2tensor(eigen_mat);
//     // VLOG(1) <<"tens is " <<tens;
//     // //assign a tensor to this one and check the value of it
//     // Tensor tens_copy=tens;
//     // VLOG(1) <<"tens_copy is " <<tens_copy;
//     // //modofy tens and check if tens_copy also modified
//     // auto a = tens.accessor<float, 3>();
//     // a[0][0][0]=9;
//     // VLOG(1) <<"after modif tens is " <<tens;
//     // VLOG(1) <<"after modif tens_copy is " <<tens_copy;



// }

void LatticeGPU_test::bilateral_shuffled_positions_test_from_path(const std::string& cv_img_path){
    cv::Mat cv_mat= cv::imread(cv_img_path);
    // cv::resize(cv_mat, cv_mat, cv::Size(), 0.25, 0.25);
    cv::Mat cv_mat_gray;
    cv::cvtColor(cv_mat, cv_mat_gray, CV_BGR2GRAY);
    cv::Mat cv_img_float_rgb;
    cv::Mat cv_img_float_gray;
    cv_mat_gray.convertTo(cv_img_float_gray, CV_32FC1, 1.0/255.0);
    cv_mat.convertTo(cv_img_float_rgb, CV_32FC3, 1.0/255.0);


    //compute positions and values for this image
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
    compute_positions_and_values(positions, values, cv_img_float_rgb);

    //shuffle positions and values the same
    std::vector<int> originalidx2shuffledidx (positions.rows());
    std::iota (std::begin(originalidx2shuffledidx), std::end(originalidx2shuffledidx), 0); //fill with consecutive elements 1,2,3,4,
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(originalidx2shuffledidx.begin(), originalidx2shuffledidx.end(), g);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions_shuffled;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values_shuffled;
    positions_shuffled=positions;
    values_shuffled=values;
    for(int i=0; i<positions.rows(); i++){
        int idx_insert= originalidx2shuffledidx[i];
        positions_shuffled.row(idx_insert) = positions.row(i);
        values_shuffled.row(idx_insert) = values.row(i);
    }


    Tensor positions_original_tensor=eigen2tensor(positions);
    Tensor positions_tensor=eigen2tensor(positions_shuffled);
    Tensor values_tensor=eigen2tensor(values_shuffled);
    VLOG(1) << "computed nr of positions " << positions.rows();

    
    // Tensor output_values_tensor = m_lattice->bilateral_filter(positions_tensor, values_tensor);
    TIME_START("full_filter");
    m_lattice->begin_splat();
    m_lattice->splat_standalone(positions_tensor, values_tensor, /*with_homogeneous_coord*/ true );
    std::shared_ptr<Lattice> blurred_lattice=m_lattice->blur_standalone();
    // Tensor output_values_tensor = blurred_lattice->slice_standalone_no_precomputation(positions_original_tensor, /*do_normalization*/true);
    Tensor output_values_tensor = blurred_lattice->slice_standalone_no_precomputation(positions_original_tensor, /*with_homogeneous_coord*/ true);
    TIME_END("full_filter");


    //output values after filtering to cv_mat and show
    EigenMatrixXfRowMajor output_values_eigen=tensor2eigen(output_values_tensor);


    // cv::Mat cv_mat_filtered= eigen2mat(output_values_eigen_unshuffled, cv_mat.rows, cv_mat.cols);
    cv::Mat cv_mat_filtered= eigen2mat(output_values_eigen, cv_mat.rows, cv_mat.cols);
    Gui::show(cv_mat_filtered, "cv_mat_filtered");

    // //view the values 
    // cv::Mat hashed_values_mat = values_as_img();
    // Gui::show(hashed_values_mat, "hashed_values_mat");


}

void LatticeGPU_test::show_values(const torch::Tensor& values, const int width, const int height){

    //output values after filtering to cv_mat and show
    EigenMatrixXfRowMajor output_values_eigen=tensor2eigen(values);

    cv::Mat cv_mat_filtered= eigen2mat(output_values_eigen, height, width);
    Gui::show(cv_mat_filtered, "cv_mat_filtered");

}

std::vector<torch::Tensor> LatticeGPU_test::compute_positions_and_values_from_path(const std::string& cv_img_path){
    cv::Mat cv_mat= cv::imread(cv_img_path);
    // cv::resize(cv_mat, cv_mat, cv::Size(), 0.25, 0.25);
    VLOG(1) <<"getting positions from cv_mat of size rows, cols" << cv_mat.rows << " " << cv_mat.cols;
    cv::Mat cv_mat_gray;
    cv::cvtColor(cv_mat, cv_mat_gray, CV_BGR2GRAY);
    cv::Mat cv_img_float_rgb;
    cv::Mat cv_img_float_gray;
    cv_mat_gray.convertTo(cv_img_float_gray, CV_32FC1, 1.0/255.0);
    cv_mat.convertTo(cv_img_float_rgb, CV_32FC3, 1.0/255.0);

    //compute positions and values for this image
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
    compute_positions_and_values(positions, values, cv_img_float_rgb);

    // //make a sobel filter
    // cv::Mat grad_x;
    // // cv::Scharr( cv_img_float_rgb, grad_x, CV_32F, 1, 0);
    // cv::Scharr( cv_mat, grad_x, CV_8U, 1, 0);
    // cv::imwrite( "./lena_grad_x.png", grad_x );

    Tensor positions_tensor=eigen2tensor(positions);
    Tensor values_tensor=eigen2tensor(values);

    return {positions_tensor, values_tensor};

}

void LatticeGPU_test::compute_positions_and_values(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& positions, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& values,  cv::Mat& cv_img_float){

    CHECK( type2byteDepth(cv_img_float.type() )==CV_32F ) <<"Can only calculate positions and values for cv mat of float depth";

    Gui::show(cv_img_float, "cv_mat_float");


    int channels=cv_img_float.channels();
    int pos_dim, val_dim;
    if(channels==1){
        pos_dim=3; //x,y,gray val
        val_dim=1;
    }else if(channels==3){
        pos_dim=5; //xy.rgb
        val_dim=3;
    }else{
        LOG(FATAL) << "Cannot compute positions and values for this image. We only support images of channels 3 and 1";
    }

    //create position and values for this image
    int nr_pixels=cv_img_float.rows*cv_img_float.cols;
    positions.resize(nr_pixels, pos_dim);
    values.resize(nr_pixels, val_dim);
    for(int i=0; i<cv_img_float.rows; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert=i*cv_img_float.cols + j;

            positions(idx_insert,0)=i;
            positions(idx_insert,1)=j;

            if(channels==1){
                float gray_val=cv_img_float.at<float>(i,j);
                positions(idx_insert,2)=gray_val;

                values(idx_insert,0)=gray_val;
            }

            if(channels==3){
                cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(i,j);
                positions(idx_insert,2)=rgb[0];
                positions(idx_insert,3)=rgb[1];
                positions(idx_insert,4)=rgb[2];

                values(idx_insert,0)=rgb[0];
                values(idx_insert,1)=rgb[1];
                values(idx_insert,2)=rgb[2];
            }

        }
    }

}

void LatticeGPU_test::compute_positions_and_values_from_random_pixels(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& positions, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& values,  cv::Mat& cv_img_float){

    CHECK( type2byteDepth(cv_img_float.type() )==CV_32F ) <<"Can only calculate positions and values for cv mat of float depth";

    Gui::show(cv_img_float, "cv_mat_float");


    int channels=cv_img_float.channels();
    int pos_dim, val_dim;
    if(channels==1){
        pos_dim=3; //x,y,gray val
        val_dim=1;
    }else if(channels==3){
        pos_dim=5; //xy.rgb
        val_dim=3;
    }else{
        LOG(FATAL) << "Cannot compute positions and values for this image. We only support images of channels 3 and 1";
    }

    //create position and values for this image
    int nr_pixels=cv_img_float.rows*cv_img_float.cols;

    // int nr_random_points=rand_int(0,nr_pixels/12);
    int nr_random_points=m_rand_gen->rand_int(0,nr_pixels/420);
    positions.resize(nr_random_points, pos_dim);
    values.resize(nr_random_points, val_dim);
    int idx_insert=0;
    for(int idx_insert=0; idx_insert<nr_random_points; idx_insert++){
            int rand_y=m_rand_gen->rand_int(0,cv_img_float.rows-1);
            int rand_x=m_rand_gen->rand_int(0,cv_img_float.cols-1);

            positions(idx_insert,0)=rand_y;
            positions(idx_insert,1)=rand_x;

            if(channels==1){
                float gray_val=cv_img_float.at<float>(rand_y, rand_x);
                positions(idx_insert,2)=gray_val;

                values(idx_insert,0)=gray_val;
            }

            if(channels==3){
                cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(rand_y, rand_x);
                positions(idx_insert,2)=rgb[0];
                positions(idx_insert,3)=rgb[1];
                positions(idx_insert,4)=rgb[2];

                values(idx_insert,0)=rgb[0];
                values(idx_insert,1)=rgb[1];
                values(idx_insert,2)=rgb[2];
            }

    }

}

void LatticeGPU_test::compute_positions_and_values_in_row_range(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& positions, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& values,  cv::Mat& cv_img_float, const int start_row, const int end_row){

    CHECK( type2byteDepth(cv_img_float.type() )==CV_32F ) <<"Can only calculate positions and values for cv mat of float depth";
    CHECK(end_row>start_row) << "End row must be higher than start row";
    CHECK(end_row<=cv_img_float.rows) << "end row cannot be bigger than cv_img_float.rows. end_row is " << end_row << " cv_img_float.rows " << cv_img_float.rows;

    Gui::show(cv_img_float, "cv_mat_float");


    int channels=cv_img_float.channels();
    int pos_dim, val_dim;
    if(channels==1){
        pos_dim=3; //x,y,gray val
        val_dim=1;
    }else if(channels==3){
        pos_dim=5; //xy.rgb
        val_dim=3;
    }else{
        LOG(FATAL) << "Cannot compute positions and values for this image. We only support images of channels 3 and 1";
    }

    //create position and values for this image
    int nr_pixels= (end_row-start_row) *cv_img_float.cols;
    positions.resize(nr_pixels, pos_dim);
    values.resize(nr_pixels, val_dim);
    for(int i=start_row; i<end_row; i++){
        for(int j=0; j<cv_img_float.cols; j++){
            int idx_insert= (i-start_row)  *cv_img_float.cols + j; //start inserting from 0 of the positions and vals. Even if the start_row is higher

            positions(idx_insert,0)=i;
            positions(idx_insert,1)=j;

            if(channels==1){
                float gray_val=cv_img_float.at<float>(i,j);
                positions(idx_insert,2)=gray_val;

                values(idx_insert,0)=gray_val;
            }

            if(channels==3){
                cv::Vec3f rgb=cv_img_float.at<cv::Vec3f>(i,j);
                positions(idx_insert,2)=rgb[0];
                positions(idx_insert,3)=rgb[1];
                positions(idx_insert,4)=rgb[2];

                values(idx_insert,0)=rgb[0];
                values(idx_insert,1)=rgb[1];
                values(idx_insert,2)=rgb[2];
            }

        }
    }

}

cv::Mat LatticeGPU_test::values_as_img(){
//     cudaDeviceSynchronize();
//     int nr_elements=m_lattice->m_hash_table.m_capacity * (m_lattice->m_hash_table.m_val_dim+1);
//     float hashed_values_cpu[nr_elements];
//     VLOG(1) << "Copying to cpu";
//     cudaMemcpy(hashed_values_cpu, m_lattice->m_hash_table.m_values , sizeof(float) *nr_elements , cudaMemcpyDeviceToHost);    
//     VLOG(1) << "creating cv mat";
//     // cv::Mat values_mat = cv::Mat(m_lattice->m_hash_table_capacity, m_lattice->m_hash_table.m_val_dim+1, CV_32FC1, hashed_values_cpu ).clone();
//     cv::Mat values_mat = cv::Mat(m_lattice->m_hash_table_capacity, m_lattice->m_hash_table.m_val_dim+1, CV_32FC1, hashed_values_cpu ).clone();
//     cv::resize(values_mat, values_mat, cv::Size(), 0.25, 0.25);
//     CHECK(values_mat.rows < 30000) << "Cannot create an image that is that big because we cannot show it in opengl. Maximum of opengl is around 32k";
//     VLOG(1) << "returning";
//     return values_mat;
}