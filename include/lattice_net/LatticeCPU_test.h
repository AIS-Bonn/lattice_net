#pragma once

#include <memory>
#include <stdarg.h>

#include "opencv2/opencv.hpp"

class RandGenerator;

class LatticeCPU_test : public std::enable_shared_from_this<LatticeCPU_test>{
public:
    template <class ...Args>
    static std::shared_ptr<LatticeCPU_test> create( Args&& ...args ){
        return std::shared_ptr<LatticeCPU_test>( new LatticeCPU_test(std::forward<Args>(args)...) );
    }
    void compute();

    void bilateral_test(const cv::Mat& cv_img_float); //normal bilateral filtering, should work just fine
    void filter_positions_test(const cv::Mat& cv_img_float); // the values will now include also x,y and also rgb
    void filter_positions_gray_test(const cv::Mat& cv_img_float); // the values will now include also x,y and also intensity
    void filter_positions_ngf_test(const cv::Mat& cv_img_float); // the values will now include also x,y and also intensity

    void bilateral_test_from_path(const std::string& cv_img_path); //calls bilateral test but takes a string as input

    //objects 
    std::shared_ptr<RandGenerator> m_rand_gen;

    //params 
    float m_spacial_sigma;
    float m_color_sigma;
    float m_surfel_scaling;

private:
    LatticeCPU_test(const std::string config_file);
};

