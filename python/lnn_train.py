#!/usr/bin/env python3.6

import torch

import sys
import os
import numpy as np
from tqdm import tqdm
import time

from easypbr  import *
from dataloaders import *
from lattice.lattice_py import LatticePy
from lattice.diceloss import GeneralizedSoftDiceLoss
from lattice.lovasz_loss import LovaszSoftmax
from lattice.models import *

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.state_callback import *
from callbacks.phase import *

from optimizers.over9000.radam import *
# from optimizers.pytorch_optimizer.torch_optimizer.adabound import *
# from optimizers.pytorch_optimizer.torch_optimizer.adamod import *


# config_file="lnn_train_shapenet.cfg"
config_file="lnn_train_semantic_kitti.cfg"
# config_file="lnn_train_scannet.cfg"

torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)

# #initialize the parameters used for training
train_params=TrainParams.create(config_file)    
model_params=ModelParams.create(config_file)    


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
    if train_params.with_viewer():
        view=Viewer.create(config_path)

    first_time=True

    #torch stuff 
    lattice=LatticePy()
    lattice.create(config_path, "splated_lattice")

    cb = CallbacksGroup([
        # LatticeSigmaCallback() #TODO
        # ViewerCallback(),
        # VisdomCallback(),
        StateCallback() #changes the iter nr epoch nr,
    ])
    #create loaders
    loader_train=create_loader(train_params.dataset_name(), config_path)
    loader_train.set_mode_train()
    loader_train.start()
    loader_test=create_loader(train_params.dataset_name(), config_path)
    loader_test.set_mode_test()
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
    model=LNN(loader_train.label_mngr().nr_classes(), model_params, False, False).to("cuda")
    #create loss function
    # loss_fn=GeneralizedSoftDiceLoss(ignore_index=loader_train.label_mngr().get_idx_unlabeled() ) 
    loss_fn=LovaszSoftmax(ignore_index=loader_train.label_mngr().get_idx_unlabeled())
    class_weights_tensor=model.compute_class_weights(loader_train.label_mngr().class_frequencies(), loader_train.label_mngr().get_idx_unlabeled())
    secondary_fn=torch.nn.NLLLoss(ignore_index=loader_train.label_mngr().get_idx_unlabeled(), weights=class_weights_tensor)  #combination of nll and dice  https://arxiv.org/pdf/1809.10486.pdf

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
                        TIME_START("forward")
                        pred_logsoftmax, pred_raw, delta_weight_error_sum=model(lattice, positions, values)
                        TIME_END("forward")
                        # loss_dice = 0.3*loss_fn(pred_logsoftmax, target) #seems to work quite good with 0.3 of dice and 0.7 of CE. Trying now to lovasz
                        # loss_dice = 0.5*loss_fn(pred_logsoftmax, target)
                        loss_dice = 0.5*loss_fn(pred_logsoftmax, target)
                        # loss_dice = 0.0
                        #print("pred_softmax has shape ", pred_softmax.shape, "target is ", target.shape)
                        loss_ce = 0.5*secondary_fn(pred_logsoftmax, target)
                        # loss_ce = 0.0
                        loss = loss_dice+loss_ce
                        # loss += 0.1*delta_weight_error_sum #TODO is not clear how much it improves iou if at all
                        # loss /=train_params.batch_size() #TODO we only support batchsize of 1 at the moment

                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            # optimizer=torch.optim.AdamW(model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay(), amsgrad=True)
                            optimizer=RAdam(model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay())
                            # optimizer=torch.optim.SGD(model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay(), momentum=0.9, nesterov=True)
                            # optimizer=AdaBound(model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay()) #starts as adam, becomes sgd epoch 175, reduced lr twice, got train iou of 82.5 and test iou of 74.5
                            # optimizer=AdaMod(model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay()) #does warmup like radam but all along the training after  epoch 70, reaches train iou of 83.3 and test iou of 74.8 and converges a lot faster than the adabound
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

                        cb.after_forward_pass(pred_softmax=pred_logsoftmax, target=target, cloud=cloud, loss=loss.item(), loss_dice=loss_dice.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        pbar.update(1)

                    #backward
                    if is_training:
                        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / phase.loader.nr_samples() )
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        loss.backward()
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
                    # if not phase.grad:


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
