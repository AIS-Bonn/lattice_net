#pragma once

#include <torch/torch.h>

#include <memory>
#include <stdarg.h>

#include "opencv2/opencv.hpp"

#include <Eigen/Core>


class Lattice;
class RandGenerator;

class LatticeGPU_test : public std::enable_shared_from_this<LatticeGPU_test>{
public:
    template <class ...Args>
    static std::shared_ptr<LatticeGPU_test> create( Args&& ...args ){
        return std::shared_ptr<LatticeGPU_test>( new LatticeGPU_test(std::forward<Args>(args)...) );
    }

    void bilateral_test_from_path(const std::string& cv_img_path); //calls bilateral test but takes a string as input
    void bilateral_recursive_test_from_path(const std::string& cv_img_path); //calls bilateral test and recursivelly uses the output image as the new input
    void bilateral_incremental_test_from_path(const std::string& cv_img_path); //calls bilateral test and splats incrementally an image by getting random pixels from it at each iter
    void bilateral_shuffled_positions_test_from_path(const std::string& cv_img_path); //calls bilateral test and splats suing randomly shuffled positions. This should reduce the number of threads that splat to the same simplex and reduce mutex pressure on the hastable entries
    std::vector<torch::Tensor> compute_positions_and_values_from_path(const std::string& cv_img_path); //returns tensors for positions and values, ready to be used by pytorch
    void show_values(const torch::Tensor& values, const int width, const int height); //gets the sliced values, reshapes them into a cv image and displays it


    //objects
    std::shared_ptr<Lattice> m_lattice;
    std::shared_ptr<RandGenerator> m_rand_gen;
private:
    LatticeGPU_test(const std::string config_file);
    void compute_positions_and_values(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& positions, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& values,  cv::Mat& cv_img_float);
    void compute_positions_and_values_from_random_pixels(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& positions, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& values,  cv::Mat& cv_img_float);
    void compute_positions_and_values_in_row_range(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& positions, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& values,  cv::Mat& cv_img_float, const int start_row, const int end_row);
    // void compute_random_positions_and_values(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& positions, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& values,  cv::Mat& cv_img_float);

    //returns the values stored in the hashtable as a cvMat/ The values have size m_capacity x (val_dim+1)
    cv::Mat values_as_img();
};
