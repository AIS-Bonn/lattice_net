#include "lattice_net/ModelParams.h"

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


ModelParams::ModelParams(const std::string config_file){

    init_params(config_file);
}

void ModelParams::init_params(const std::string config_file){

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config train_config=cfg["model"];

    m_positions_mode=(std::string)train_config["positions_mode"];
    m_values_mode=(std::string)train_config["values_mode"];
    m_pointnet_start_nr_channels=train_config["pointnet_start_nr_channels"];
    m_nr_downsamples=train_config["nr_downsamples"];
    m_nr_blocks_down_stage=train_config["nr_blocks_down_stage"];
    m_nr_blocks_bottleneck=train_config["nr_blocks_bottleneck"];
    m_nr_blocks_up_stage=train_config["nr_blocks_up_stage"];
    m_nr_levels_down_with_normal_resnet=train_config["nr_levels_down_with_normal_resnet"];
    m_nr_levels_up_with_normal_resnet=train_config["nr_levels_up_with_normal_resnet"];
    m_compression_factor=train_config["compression_factor"];
    m_dropout_last_layer=train_config["dropout_last_layer"];

}

std::string ModelParams::positions_mode(){
    return m_positions_mode;
}
std::string ModelParams::values_mode(){
    return m_values_mode;
}
int ModelParams::pointnet_start_nr_channels(){
    return m_pointnet_start_nr_channels;
}
int ModelParams::nr_downsamples(){
    return m_nr_downsamples;
}
std::vector<int> ModelParams::nr_blocks_down_stage(){
    return m_nr_blocks_down_stage;
}
int ModelParams::nr_blocks_bottleneck(){
    return m_nr_blocks_bottleneck;
}
std::vector<int> ModelParams::nr_blocks_up_stage(){
    return m_nr_blocks_up_stage;
}
int ModelParams::nr_levels_down_with_normal_resnet(){
    return m_nr_levels_down_with_normal_resnet;
}
int ModelParams::nr_levels_up_with_normal_resnet(){
    return m_nr_levels_up_with_normal_resnet;
}
float ModelParams::compression_factor(){
    return m_compression_factor;
}
float ModelParams::dropout_last_layer(){
    return m_dropout_last_layer;
}




