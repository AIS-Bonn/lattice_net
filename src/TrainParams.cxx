#include "lattice_net/TrainParams.h"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


TrainParams::TrainParams(const std::string config_file){

    init_params(config_file);
}

void TrainParams::init_params(const std::string config_file){

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config train_config=cfg["train"];
    m_dataset_name=(std::string)train_config["dataset_name"];
    m_with_viewer = train_config["with_viewer"];
    m_with_visdom = train_config["with_visdom"];
    m_lr = train_config["lr"];
    m_weight_decay = train_config["weight_decay"];
    m_save_checkpoint=train_config["save_checkpoint"];
    m_checkpoint_path=(std::string)train_config["checkpoint_path"];

    if(m_save_checkpoint && !fs::is_directory(m_checkpoint_path)) {
        LOG(FATAL) << "The directory for saving checkpoint was not created under " << m_checkpoint_path << ". Maybe you need to create it or maybe you are on the wrong machine.";
    }

}

std::string TrainParams::dataset_name(){
    return m_dataset_name;
}
bool TrainParams::with_viewer(){
    return m_with_viewer;
}
bool TrainParams::with_visdom(){
    return m_with_visdom;
}
float TrainParams::lr(){
    return m_lr;
}
float TrainParams::weight_decay(){
    return m_weight_decay;
}
bool TrainParams::save_checkpoint(){
    return m_save_checkpoint;
}
std::string TrainParams::checkpoint_path(){
    return m_checkpoint_path;
}



