#pragma once 

#include <memory>
#include <stdarg.h>

//class used to read some network training parameters from a config file ( things like learnign rate and batch size ) This class is also exposed to python so it can be used in pytorch

class EvalParams: public std::enable_shared_from_this<EvalParams>
{
public:
    template <class ...Args>
    static std::shared_ptr<EvalParams> create( Args&& ...args ){
        return std::shared_ptr<EvalParams>( new EvalParams(std::forward<Args>(args)...) );
    }

    bool with_viewer();
    std::string dataset_name();
    std::string checkpoint_path(); //points to the model that we want to load
    bool do_write_predictions();
    std::string output_predictions_path();
    


private:
    EvalParams(const std::string config_file);
    void init_params(const std::string config_file);

    std::string m_dataset_name;
    bool m_with_viewer; //wether the training script will show in a viewer the gt_cloud and prediciton cloud
    std::string m_checkpoint_path;
    bool m_do_write_predictions;
    std::string m_output_predictions_path;
};