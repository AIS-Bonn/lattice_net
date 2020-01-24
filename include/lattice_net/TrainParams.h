#pragma once 

#include <memory>
#include <stdarg.h>

//class used to read some network training parameters from a config file ( things like learnign rate and batch size ) This class is also exposed to python so it can be used in pytorch

class TrainParams: public std::enable_shared_from_this<TrainParams>
{
public:
    template <class ...Args>
    static std::shared_ptr<TrainParams> create( Args&& ...args ){
        return std::shared_ptr<TrainParams>( new TrainParams(std::forward<Args>(args)...) );
    }

    bool with_viewer();
    bool with_debug_output();
    bool with_error_checking();
    std::string dataset_name();
    int batch_size();
    float lr();
    float base_lr();
    float weight_decay();
    float nr_epochs_per_half_cycle();
    float exponential_gamma();
    int max_training_epochs();
    bool save_checkpoint();
    std::string checkpoint_path();
    std::string experiment();


private:
    TrainParams(const std::string config_file);
    void init_params(const std::string config_file);

    std::string m_dataset_name;
    bool m_with_viewer; //wether the training script will show in a viewer the gt_cloud and prediciton cloud
    bool m_with_debug_output; //weather the training script should output a bunch of debug stuff
    bool m_with_error_checking; //weather the training script should check for erronoues things like how many positions we sliced correctly
    int m_batch_size;
    float m_lr; 
    float m_base_lr;
    float m_weight_decay;
    float m_nr_epochs_per_half_cycle; //for cyclic learning a half_cycle is the process of going from base_lr to max_lr or viceverse. This is how many epochs it takes to do a half_cycle Leslie paper recomend a value between 2 and 10
    float m_exponential_gamma; //the max_lr in cycle learning gets multiplied by epoch^exponential_gamma after every epoch
    int m_max_training_epochs;
    bool m_save_checkpoint;
    std::string m_checkpoint_path;
    std::string m_experiment;

};