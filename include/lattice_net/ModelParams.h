#pragma once 

#include <memory>
#include <stdarg.h>
#include <vector>

#include <Eigen/Core>

//class used to read some model parameters from a config file ( things like nr of layers and channels for each layer ) This class is also exposed to python so it can be used in pytorch

class ModelParams: public std::enable_shared_from_this<ModelParams>
{
public:
    template <class ...Args>
    static std::shared_ptr<ModelParams> create( Args&& ...args ){
        return std::shared_ptr<ModelParams>( new ModelParams(std::forward<Args>(args)...) );
    }

    std::string positions_mode(); //the values we feed into the lattice can be either: xyz, xyz+rgb
    std::string values_mode(); //the values we feed into the lattice can be either: none, intensity
    Eigen::Vector3i pointnet_layers();
    int pointnet_start_nr_channels(); //after pointnet architecture we add one more fully connected layer to elevate the lattice vertices up to this nr_channels
    int nr_downsamples(); //the network uses this many corsening of the lattice graph
    std::vector<int> nr_blocks_down_stage(); //each corsening stage inclues some amount of resnetblocks or bottleneck blocks. This says how many blocks for each stage we have
    int nr_blocks_bottleneck(); //after the last corsesning we have a certain amount of bottlneck blocks
    std::vector<int> nr_blocks_up_stage();
    int nr_levels_down_with_normal_resnet(); //starting from the top of the network (closer to the input) we count how many of the downsampling stages should include ResnetBlocks instead of BottleneckBlock (the first few stages have few channels so we can afford to use ResnetBlock instead of Bottleneck block)
    int nr_levels_up_with_normal_resnet(); //starting from the bottom of the network (closer to the output) we count how many of the upsampling stages should include ResnetBlocks instead of BottleneckBlocks (the first few stages have few channels so we can afford to use ResnetBlock instead of Bottleneck block)
    float compression_factor(); //each corsening of the graph increases the nr of channels by prev_nr_channels*2*compression_factor. So if the compression factor is 1.0 we will double the nr of channels
    float dropout_last_layer(); //Probability of dropout added to the last linear layer, the one that maps to the nr_classes and is just before the softmax 
    std::string experiment();




private:
    ModelParams(const std::string config_file);
    void init_params(const std::string config_file);

    std::string m_positions_mode;
    std::string m_values_mode;
    Eigen::Vector3i m_pointnet_layers;
    int m_pointnet_start_nr_channels;
    int m_nr_downsamples; 
    std::vector<int> m_nr_blocks_down_stage; 
    int m_nr_blocks_bottleneck; 
    std::vector<int> m_nr_blocks_up_stage;
    int m_nr_levels_down_with_normal_resnet; 
    int m_nr_levels_up_with_normal_resnet; 
    float m_compression_factor; 
    float m_dropout_last_layer;

    std::string m_experiment;

};