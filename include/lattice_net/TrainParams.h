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
    float weight_decay();
    int max_training_epochs();
    bool save_checkpoint();
    std::string checkpoint_path();


private:
    TrainParams(const std::string config_file);
    void init_params(const std::string config_file);

    std::string m_dataset_name;
    bool m_with_viewer; //wether the training script will show in a viewer the gt_cloud and prediciton cloud
    bool m_with_debug_output; //weather the training script should output a bunch of debug stuff
    bool m_with_error_checking; //weather the training script should check for erronoues things like how many positions we sliced correctly
    int m_batch_size;
    float m_lr; 
    float m_weight_decay;
    int m_max_training_epochs;
    bool m_save_checkpoint;
    std::string m_checkpoint_path;

};