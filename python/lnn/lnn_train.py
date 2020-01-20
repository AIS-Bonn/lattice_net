#!/usr/bin/env python3.6

import torch
from torch.autograd import Function
from torch import Tensor

import sys
import os
import numpy as np

from easypbr  import *
from dataloaders import *
from lattice_py import LatticePy
from lattice_funcs import * #for the timing macros
import visdom
import torchnet
from lr_finder import LRFinder
from scores import Scores
from model_ctx import ModelCtx
from vis import Vis
from diceloss import GeneralizedSoftDiceLoss

from callback import *
from viewer_callback import *
from scores_callback import *
from state_callback import *
from phase import *
from models import *


config_file="lnn_train_shapenet.cfg"

torch.manual_seed(0)

# #initialize the parameters used for training
train_params=TrainParams.create(config_file)    
model_params=ModelParams.create(config_file)    


def create_loader(dataset_name, config_file):
    # if(dataset_name=="semantickitti"):
    #     loader=DataLoaderSemanticKitti(config_file)
    # elif dataset_name=="shapenet":
    #     loader=DataLoaderShapeNetPartSeg(config_file)
    # elif dataset_name=="toyexample":
    #     loader=DataLoaderToyExample(config_file)
    # elif dataset_name=="stanford":
    #     loader=DataLoaderStanfordIndoor(config_file)
    # elif dataset_name=="scannet":
    #     loader=DataLoaderScanNet(config_file)
    # else:
    #     err="Datset name not recognized. It is " + dataset_name
    #     sys.exit(err)

    loader=DataLoaderShapeNetPartSeg(config_file)

    return loader


def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../config', config_file)
    if train_params.with_viewer():
        view=Viewer.create(config_path)

    first_time=True

    #torch stuff 
    lattice=LatticePy()
    lattice.create(config_path, "splated_lattice")

    cb = CallbacksGroup([
        # LatticeSigmaCallback() #TODO
        ViewerCallback(),
        ScoresCallback(),
        StateCallback() #changes the iter nr epoch nr,
    ])
    #create loaders
    loader_train=create_loader(train_params.dataset_name(), config_path)
    loader_train.set_mode_train()
    loader_train.start()
    loader_test=create_loader(train_params.dataset_name(), config_path)
    loader_test.set_mode_test()
    loader_test.start()
    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        Phase('test', loader_test, grad=False)
    ]
    phase_idx=0
    phase=phases[phase_idx]
    #model 
    model=LNN_skippy_efficient(loader_train.label_mngr().nr_classes(), model_params, False, False).to("cuda")
    model.train(phase.grad) #turns the train on or off depending if hte phase requires gradients or not
    #create loss function
    loss_fn=GeneralizedSoftDiceLoss(ignore_index=loader_train.label_mngr().get_idx_unlabeled() ) 
    secondary_fn=torch.nn.NLLLoss(ignore_index=loader_train.label_mngr().get_idx_unlabeled())  #combination of nll and dice  https://arxiv.org/pdf/1809.10486.pdf

    while True:

        for phase in phases:
            cb.phase_started(phase=phase)
            model.train(phase.grad)

            while ( phase.samples_processed_this_epoch < phase.loader.nr_samples()):

                if(phase.loader.has_data()): 
                    cloud=phase.loader.get_cloud()

                    is_training = phase.grad

                    #forward
                    with torch.set_grad_enabled(is_training):
                        cb.before_forward_pass(lattice=lattice) #sets the appropriate sigma for the lattice
                        positions, values, target = model.prepare_cloud(cloud) #prepares the cloud for pytorch, returning tensors alredy in cuda
                        pred_softmax, pred_raw, delta_weight_error_sum=model(lattice, positions, values)
                        loss = loss_fn(pred_softmax, target)
                        loss += secondary_fn(pred_softmax, target)
                        loss += 0.1*delta_weight_error_sum
                        # loss /=train_params.batch_size() #TODO we only support batchsize of 1 at the moment

                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            optimizer=torch.optim.AdamW(model.parameters(), lr=train_params.base_lr(), weight_decay=train_params.weight_decay(), amsgrad=True)
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2, 2)

                        cb.after_forward_pass(pred_softmax=pred_softmax, cloud=cloud, loss=loss, phase=phase, lr=scheduler.get_lr()) #visualizes the prediction 

                    #backward
                    if is_training:
                        scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / phase.loader.nr_samples() )
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        loss.backward()
                        cb.after_backward_pass()
                        optimizer.step()



                if phase.loader.is_finished():
                    cb.epoch_ended(phase=phase) 
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
