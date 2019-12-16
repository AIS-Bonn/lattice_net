#include "surfel_renderer/lattice/Voxel.h"

//my stuff 
#include "surfel_renderer/utils/MiscUtils.h"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//my stuff

//c++
#include <iostream>

#define MAX_NR_CLASSES 128 // when argmaxing over the classes we assume we have a maximum nr of classes of 128

Voxel::Voxel()
    {

}

void Voxel::add_point(const Eigen::VectorXd& point){
    m_points.push_back(point);
}

void Voxel::add_label_gt(const int label){
    CHECK(label<MAX_NR_CLASSES) << "The label is higher than the maximum nr of classes. You should increase the maximum nr of classes. Label is " << label << " max_nr_classes is " << MAX_NR_CLASSES; 
    m_labels_gt.push_back(label);
}

void Voxel::compute_argmax_label(){

    Eigen::VectorXi probs;
    probs.resize(MAX_NR_CLASSES); // assume we have for each voxel a row of 128 labels
    probs.setZero();
    for(int i = 0; i < m_labels_gt.size(); i++){
        probs(m_labels_gt[i])++;
    }

    //max over the probabilities 
    int max_idx=-1;
    probs.maxCoeff(&max_idx);
    m_label_gt=max_idx;


    CHECK(m_label_gt!=-1) << "Something went wrong with the argmax over the probs. m_label_gt is " << -1 << " max_nr_classes is " << MAX_NR_CLASSES; 
    CHECK(m_label_gt<MAX_NR_CLASSES) << "Something went wrong with the argmax over the probs. m_label_gt is " << m_label_gt << " max_nr_classes is " << MAX_NR_CLASSES; 
} 

int Voxel::nr_contained_points() const{
    return m_points.size();
}

int Voxel::label_gt() const{
    return m_label_gt;
}
