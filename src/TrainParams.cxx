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
    m_with_debug_output = train_config["with_debug_output"];
    m_with_error_checking = train_config["with_error_checking"];
    m_batch_size = train_config["batch_size"];
    m_lr = train_config["lr"];
    m_base_lr = train_config["base_lr"];
    m_weight_decay = train_config["weight_decay"];
    m_nr_epochs_per_half_cycle = train_config["nr_epochs_per_half_cycle"];
    m_exponential_gamma = train_config["exponential_gamma"];
    m_max_training_epochs = train_config["max_training_epochs"];
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
bool TrainParams::with_debug_output(){
    return m_with_debug_output;
}
bool TrainParams::with_error_checking(){
    return m_with_error_checking;
}
int TrainParams::batch_size(){
    return m_batch_size;
}
float TrainParams::lr(){
    return m_lr;
}
float TrainParams::base_lr(){
    return m_base_lr;
}
float TrainParams::weight_decay(){
    return m_weight_decay;
}
float TrainParams::nr_epochs_per_half_cycle(){
    return m_nr_epochs_per_half_cycle;
}
float TrainParams::exponential_gamma(){
    return m_exponential_gamma;
}
int TrainParams::max_training_epochs(){
    return m_max_training_epochs;
}
bool TrainParams::save_checkpoint(){
    return m_save_checkpoint;
}

std::string TrainParams::checkpoint_path(){
    return m_checkpoint_path;
}

