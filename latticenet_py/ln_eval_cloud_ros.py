#!/usr/bin/env python3.6

import torch

import sys
import os
import numpy as np
from tqdm import tqdm

from easypbr  import *
from dataloaders import *
from latticenet  import TrainParams
from latticenet  import ModelParams
from latticenet  import EvalParams
from lattice.lattice_py import LatticePy
from lattice.models import *

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.state_callback import *
from callbacks.phase import *

#semantickitt on infcuda2


config_file="ln_eval_cloud_ros.cfg"

torch.manual_seed(0)

# #initialize the parameters used for training
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
eval_params=EvalParams.create(config_path) 
model_params=ModelParams.create(config_path)    

def write_prediction(pred_softmax, cloud, pred_path):
    mesh_pred=cloud.clone()
    l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy()
    #l_pred=l_pred.unsqueeze(1)
    l_pred = np.expand_dims(l_pred, axis=1)
    #print("l_pred has shape ", l_pred.shape)
    mesh_pred.color_from_label_indices(l_pred)
    mesh_pred.L_pred=l_pred
    mesh_pred.save_to_file(pred_path)
    
def write_gt(cloud, gt_path):
    mesh_gt=cloud.clone()
    mesh_gt.color_from_label_indices(cloud.L_gt)
    mesh_gt.save_to_file(gt_path)


def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if eval_params.with_viewer():
        view=Viewer.create(config_path)

    first_time=True

    #torch stuff 
    lattice=LatticePy()
    lattice.create(config_path, "splated_lattice")

    cb_list = []
    # if(eval_params.with_viewer()):
        # cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    #create loaders
    # loader_test=create_loader(eval_params.dataset_name(), config_path)
    # loader_test.set_mode_test()
    bag=RosBagPlayer.create(config_path)
    loader=DataLoaderCloudRos(config_path)
    label_file="/media/rosu/Data/data/semantic_kitti/colorscheme_and_labels/labels.txt"
    colorscheme_file="/media/rosu/Data/data/semantic_kitti/colorscheme_and_labels/color_scheme_compacted_colors.txt"
    freq_file="/media/rosu/Data/data/semantic_kitti/colorscheme_and_labels/frequency.txt"
    label_mngr=LabelMngr(label_file, colorscheme_file, freq_file, 0)

    #create phases
    phases= [
        Phase('test', loader, grad=False)
    ]
    #model 
    model=LNN(label_mngr.nr_classes(), model_params ).to("cuda")

    predictions_list=[]
    scores_list=[]

    while loader.is_loader_thread_alive():

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)

            # pbar = tqdm(total=phase.loader.nr_samples())
            pbar = tqdm(total=100000)
            # while ( phase.samples_processed_this_epoch < phase.loader.nr_samples()):
            while loader.is_loader_thread_alive():
                
                if(phase.loader.has_data()): 
                    cloud=phase.loader.get_cloud()
                    cloud.m_label_mngr=label_mngr

                    is_training = phase.grad

                    #forward
                    with torch.set_grad_enabled(is_training):
                        cb.before_forward_pass(lattice=lattice) #sets the appropriate sigma for the lattice
                        positions, values, target = prepare_cloud(cloud, model_params) #prepares the cloud for pytorch, returning tensors alredy in cuda
                        TIME_START("forward")
                        pred_logsoftmax, pred_raw =model(lattice, positions, values)
                        TIME_END("forward")


                        #debug 
                        #l_pred=pred_logsoftmax.detach().argmax(axis=1).cpu().numpy()
                        #nr_zeros= l_pred ==0
                        #nr_zeros=nr_zeros.sum()
                        #print("nr_zeros is ", nr_zeros)
                      

                        #if its the first time we do a forward on the model we need to load here the checkpoint
                        if first_time:
                            first_time=False
                            #TODO load checkpoint
                            # now that all the parameters are created we can fill them with a model from a file
                            model.load_state_dict(torch.load(eval_params.checkpoint_path()))
                            #need to rerun forward with the new parameters to get an accurate prediction
                            pred_logsoftmax, pred_raw =model(lattice, positions, values)


                        cb.after_forward_pass(pred_softmax=pred_logsoftmax, target=target, cloud=cloud, loss=0, loss_dice=0, phase=phase, lr=0) #visualizes the prediction 
                        pbar.update(1)

                        #show and accumulate the cloud
                        if(eval_params.with_viewer()):
                            mesh_pred=cloud.clone()
                            l_pred=pred_logsoftmax.detach().argmax(axis=1).cpu().numpy()
                            mesh_pred.L_pred=l_pred
                            mesh_pred.m_vis.m_point_size=4
                            mesh_pred.m_vis.set_color_semanticpred()
                            # Scene.show(mesh_pred, "mesh_pred_"+str(phase.iter_nr) )
                            Scene.show(mesh_pred, "mesh_pred" )

                            # #get only the points that correspond to person
                            # person_idx=label_mngr.label2idx("person")
                            # mask_static=l_pred!=person_idx #is a vector of 1 for the point that are NOT person
                            # mesh_dyn=mesh_pred.clone()
                            # new_V=mesh_dyn.V.copy()
                            # new_V[mask_static]=0
                            # mesh_dyn.V=new_V
                            # Scene.show(mesh_dyn, "mesh_dyn" )





                # if phase.loader.is_finished():
                #     pbar.close()
                #     cb.epoch_ended(phase=phase, model=model, save_checkpoint=False, checkpoint_path="" ) 
                #     cb.phase_ended(phase=phase) 
                #     return


                if eval_params.with_viewer():
                    view.update()


def main():
    run()



if __name__ == "__main__":
     main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()'
