#!/usr/bin/env python3.6

import torch

import sys
import os
import numpy as np
from tqdm import tqdm
import time

from easypbr  import *
from dataloaders import *
from latticenet  import TrainParams
from latticenet  import ModelParams
from latticenet  import EvalParams
# from latticenet_py.lattice.lattice_py import LatticePy
from latticenet_py.lattice.diceloss import GeneralizedSoftDiceLoss
from latticenet_py.lattice.lovasz_loss import LovaszSoftmax
from latticenet_py.lattice.models import *

from latticenet_py.callbacks.callback import *
from latticenet_py.callbacks.viewer_callback import *
from latticenet_py.callbacks.visdom_callback import *
# from latticenet_py.callbacks.tensorboard_callback import *
from latticenet_py.callbacks.state_callback import *
from latticenet_py.callbacks.phase import *


config_file="ln_train_shapenet_example.cfg"
# config_file="lnn_train_shapenet.cfg"
# config_file="lnn_train_semantic_kitti.cfg"
# config_file="lnn_train_scannet.cfg"

torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.benchmark = True
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)

# #initialize the parameters used for training
train_params=TrainParams.create(config_path)    
model_params=ModelParams.create(config_path)    


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

def sanity_check(lattice):
    nr_positions=lattice.lattice.positions().shape[0]

    print("lattice has nr of vertices", lattice.nr_lattice_vertices() , " and nr positions is ", nr_positions)
    if(lattice.nr_lattice_vertices()<100):
        print(colored("-------------------------------", 'yellow'))
        print(colored("The number of lattice vertice is bellow 100. This could be a sign of a bug like too big of a sigma set in the config file. If you believe this is correct, feel free to ignore this.", 'yellow'))
        print(colored("-------------------------------", 'yellow'))

    if(lattice.nr_lattice_vertices()>nr_positions):
        print(colored("-------------------------------", 'yellow'))
        print(colored("The number of lattice vertice is higher than the number of positions. This could be a sign of a bug like too small of a sigma set in the config file. If you believe this is correct, feel free to ignore this.", 'yellow'))
        print(colored("-------------------------------", 'yellow'))

    #sanity check that we don't have way too many points splatting into the same vertex 
    mean_points_per_vertex, max_points_per_vertex, *rest = lattice.compute_nr_points_per_lattice_vertex()
    if(max_points_per_vertex>nr_positions*0.1):
        print(colored("-------------------------------", 'yellow'))
        print(colored("More then 10% of the positions in the cloud are splatted onto one single lattice vertex. This could be a sign of a bug like too big of a sigma set in the config file. If you believe this is correct, feel free to ignore this.", 'yellow'))
        print(colored("-------------------------------", 'yellow'))

    if(lattice.nr_lattice_vertices()>lattice.capacity()*0.25):
        print(colored("-------------------------------", 'yellow'))
        print(colored("The number of lattice vertice is close to the maximum capacity of the hashmap. Please increase the capacity of the lattice in the config file.", 'yellow'))
        print(colored("-------------------------------", 'yellow'))



def run():
    if train_params.with_viewer():
        view=Viewer.create(config_path)

    first_time=True

    experiment_name="s_10tryingback"

    #torch stuff 
    lattice=Lattice.create(config_path, "lattice")

    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    # if(train_params.with_tensorboard()):
        # cb_list.append(TensorboardCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)


    #create loaders
    loader_train=create_loader(train_params.dataset_name(), config_path)
    loader_train.set_mode_train()
    loader_train.start()
    loader_test=create_loader(train_params.dataset_name(), config_path)
    loader_test.set_mode_validation()
    if isinstance(loader_test, DataLoaderSemanticKitti):
        loader_test.set_sequence("all") #for smenantic kitti in case the train one only trains on only one sequence we still want to test on all
    if isinstance(loader_test, DataLoaderScanNet):
        loader_test.set_mode_validation() #scannet doesnt have a ground truth for the test set so we use the validation set
    loader_test.start()
    time.sleep(5) #wait a bit so that we actually have some data from the loaders
    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        Phase('test', loader_test, grad=False)
    ]
    #model 
    model=LNN(loader_train.label_mngr().nr_classes(), model_params).to("cuda")
    #create loss function
    #loss_fn=GeneralizedSoftDiceLoss(ignore_index=loader_train.label_mngr().get_idx_unlabeled() ) 
    loss_fn=LovaszSoftmax(ignore_index=loader_train.label_mngr().get_idx_unlabeled())
    #class_weights_tensor=model.compute_class_weights(loader_train.label_mngr().class_frequencies(), loader_train.label_mngr().get_idx_unlabeled())
    secondary_fn=torch.nn.NLLLoss(ignore_index=loader_train.label_mngr().get_idx_unlabeled() )  #combination of nll and dice  https://arxiv.org/pdf/1809.10486.pdf
    scheduler=None

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
                        positions, values, target = prepare_cloud(cloud, model_params) #prepares the cloud for pytorch, returning tensors alredy in cuda

                        TIME_START("forward")
                        pred_logsoftmax, pred_raw =model(lattice, positions, values)
                        TIME_END("forward")
                        loss_dice = 0.5*loss_fn(pred_logsoftmax, target)
                        loss_ce = 0.5*secondary_fn(pred_logsoftmax, target)
                        loss = loss_dice+loss_ce

                        #model.summary()

                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            optimizer=torch.optim.AdamW(model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay(), amsgrad=True)
                            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            if train_params.dataset_name()=="semantickitti":
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)

                            #sanity check that the lattice has enough vertices
                            # sanity_check(lattice)


                        cb.after_forward_pass(pred_softmax=pred_logsoftmax, target=target, cloud=cloud, loss=loss.item(), loss_dice=loss_dice.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        pbar.update(1)

                    #backward
                    if is_training:
                        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / phase.loader.nr_samples() )
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        loss.backward()

                        # model.summary()
                        # exit()

                        cb.after_backward_pass()
                        optimizer.step()

                    # Profiler.print_all_stats()

                if phase.loader.is_finished():
                    pbar.close()
                    if not is_training: #we reduce the learning rate when the test iou plateus
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                    cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
                    cb.phase_ended(phase=phase) 


                if train_params.with_viewer():
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
    # tracer.run('main()')
