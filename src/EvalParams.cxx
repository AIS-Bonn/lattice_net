#include "lattice_net/EvalParams.h"

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


EvalParams::EvalParams(const std::string config_file){

    init_params(config_file);
}

void EvalParams::init_params(const std::string config_file){

    //read all the parameters
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);

    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);

    Config eval_config=cfg["eval"];
    m_dataset_name=(std::string)eval_config["dataset_name"];
    m_with_viewer = eval_config["with_viewer"];
    m_checkpoint_path = (std::string)eval_config["checkpoint_path"];
    m_do_write_predictions=eval_config["do_write_predictions"];
    m_output_predictions_path=(std::string)eval_config["output_predictions_path"];

    if(!fs::is_regular_file(m_checkpoint_path)) {
        LOG(FATAL) << "The model file " << m_checkpoint_path << " does not exist";
    }

    if(m_do_write_predictions && !fs::is_directory(m_output_predictions_path)) {
        LOG(FATAL) << "The directory for saving predictions was not created under " << m_output_predictions_path << ". Maybe you need to create it or maybe you are on the wrong machine.";
    }
    

}

std::string EvalParams::dataset_name(){
    return m_dataset_name;
}
bool EvalParams::with_viewer(){
    return m_with_viewer;
}
std::string EvalParams::checkpoint_path(){
    return m_checkpoint_path;
}
bool EvalParams::do_write_predictions(){
    return m_do_write_predictions;
}
std::string EvalParams::output_predictions_path(){
    return m_output_predictions_path;
}