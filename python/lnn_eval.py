#!/usr/bin/env python3.6

import torch

import sys
import os
import numpy as np
from tqdm import tqdm

from easypbr  import *
from dataloaders import *
from lattice.lattice_py import LatticePy
from lattice.models import *

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.state_callback import *
from callbacks.phase import *

#semantickitt on infcuda2


config_file="lnn_eval_semantic_kitti.cfg"

torch.manual_seed(0)

# #initialize the parameters used for training
eval_params=EvalParams.create(config_file) 
model_params=ModelParams.create(config_file)    

def write_prediction(pred_softmax, cloud, pred_path):
    mesh_pred=cloud.clone()
    l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy()
    mesh_pred.color_from_label_indices(l_pred)
    mesh_pred.L_pred=l_pred
    mesh_pred.save_to_file(pred_path)
    
def write_gt(cloud, gt_path):
    mesh_gt=cloud.clone()
    mesh_gt.color_from_label_indices(cloud.L_gt)
    mesh_gt.save_to_file(gt_path)


def create_loader(dataset_name, config_file):
    if(dataset_name=="semantickitti"):
        loader=DataLoaderSemanticKitti(config_file)
    elif dataset_name=="shapenet":
        loader=DataLoaderShapeNetPartSeg(config_file)
    elif dataset_name=="scannet":
        loader=DataLoaderScanNet(config_file)
    else:
        err="Datset name not recognized. It is " + dataset_name
        sys.exit(err)

    return loader


def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if eval_params.with_viewer():
        view=Viewer.create(config_path)

    first_time=True

    #torch stuff 
    lattice=LatticePy()
    lattice.create(config_path, "splated_lattice")

    cb = CallbacksGroup([
        # LatticeSigmaCallback() #TODO
        ViewerCallback(),
        # VisdomCallback(),
        StateCallback() #changes the iter nr epoch nr,
    ])
    #create loaders
    loader_test=create_loader(eval_params.dataset_name(), config_path)
    loader_test.set_mode_test()
    if isinstance(loader_test, DataLoaderSemanticKitti):
        loader_test.set_sequence("all") #for smenantic kitti in case the train one only trains on only one sequence we still want to test on all
    if isinstance(loader_test, DataLoaderScanNet):
        loader_test.set_mode_validation() #scannet doesnt have a ground truth for the test set so we use the validation set
    loader_test.start()
    #create phases
    phases= [
        Phase('test', loader_test, grad=False)
    ]
    #model 
    model=LNN_skippy_efficient(loader_test.label_mngr().nr_classes(), model_params, False, False).to("cuda")

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)

            pbar = tqdm(total=phase.loader.nr_samples())
            while ( phase.samples_processed_this_epoch < phase.loader.nr_samples()):
                
                if(phase.loader.has_data()): 
                    cloud=phase.loader.get_cloud()

                    is_training = phase.grad

                    #forward
                    with torch.set_grad_enabled(is_training):
                        cb.before_forward_pass(lattice=lattice) #sets the appropriate sigma for the lattice
                        positions, values, target = model.prepare_cloud(cloud) #prepares the cloud for pytorch, returning tensors alredy in cuda
                        pred_softmax, pred_raw, delta_weight_error_sum=model(lattice, positions, values)
                      

                        #if its the first time we do a forward on the model we need to load here the checkpoint
                        if first_time:
                            first_time=False
                            #TODO load checkpoint
                            # now that all the parameters are created we can fill them with a model from a file
                            model.load_state_dict(torch.load(eval_params.checkpoint_path()))
                            #need to rerun forward with the new parameters to get an accurate prediction
                            pred_softmax, pred_raw, delta_weight_error_sum=model(lattice, positions, values)

                        cb.after_forward_pass(pred_softmax=pred_softmax, target=target, cloud=cloud, loss=0, phase=phase, lr=0) #visualizes the prediction 
                        pbar.update(1)

                        if eval_params.do_write_predictions():
                            # full path in which we save the cloud depends on the data loader. If it's semantic kitti we save also with the sequence, if it's scannet
                            cloud_path_full=cloud.m_disk_path
                            # cloud_path=os.path.join(os.path.dirname(cloud_path), "../../")
                            basename=os.path.splitext(os.path.basename(cloud_path_full))[0]
                            cloud_path_base=os.path.abspath(os.path.join(os.path.dirname(cloud_path_full), "../../"))
                            cloud_path_head=os.path.relpath( cloud_path_full, cloud_path_base  )
                            # print("cloud_path_head is ", cloud_path_head)
                            # print("basename is ", basename)
                            path_before_file=os.path.join(eval_params.output_predictions_path(), os.path.dirname(cloud_path_head))
                            os.makedirs(path_before_file, exist_ok=True)
                            to_save_path=os.path.join(path_before_file, basename )
                            print("saving in ", to_save_path)
                            pred_path=to_save_path+"_pred.ply"
                            gt_path=to_save_path+"_gt.ply"
                            # print("writing prediction to ", pred_path)
                            write_prediction(pred_softmax, cloud, pred_path)
                            write_gt(cloud, gt_path)

                if phase.loader.is_finished():
                    pbar.close()
                    if not is_training: #we reduce the learning rate when the test iou plateus
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                    cb.epoch_ended(phase=phase, model=model, save_checkpoint=False, checkpoint_path="" ) 
                    cb.phase_ended(phase=phase) 


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
