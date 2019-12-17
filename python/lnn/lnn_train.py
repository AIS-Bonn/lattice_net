#!/usr/bin/env python3.6

import torch
from torch.autograd import Function
from torch import Tensor

import sys
import os
import numpy as np
# http://wiki.ros.org/Packages#Client_Library_Support
# import rospkg
# rospack = rospkg.RosPack()
# sf_src_path=rospack.get_path('surfel_renderer')
# sf_build_path=os.path.abspath(sf_src_path + "/../../build/surfel_renderer")
# sys.path.append(sf_build_path) #contains the modules of pycom

# from DataLoaderTest  import *
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

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=50000)
# torch.set_printoptions(profile="full")
# torch.autograd.set_detect_anomaly(True)

# config_file="lnn_train_semantic_kitti.cfg"
# config_file="lnn_train_semantic_kitti_bg5.cfg"
config_file="lnn_train_shapenet.cfg"
# config_file="lnn_train_shapenet_bg5.cfg"
# config_file="lnn_train_stanford.cfg"
# config_file="lnn_train_stanford_bg5.cfg"
#config_file="lnn_train_scannet.cfg"
# config_file="lnn_train_scannet_bg5.cfg"



torch.manual_seed(0)


node_name="train_lnn"
vis = visdom.Visdom()
port=8097
logger_loss_acumm = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_loss_acumm'}, port=port, env=node_name)
logger_train_loss = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_train_loss'}, port=port, env=node_name)
logger_test_loss = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_test_loss'}, port=port, env=node_name)
logger_lr = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_lr'}, port=port, env=node_name)

#initialize the parameters used for training
train_params=TrainParams.create(config_file)    
dataset_name=train_params.dataset_name()
with_viewer=train_params.with_viewer()
with_debug_output=train_params.with_debug_output()
with_error_checking=train_params.with_error_checking()
with_viewer=train_params.with_viewer()
batch_size=train_params.batch_size()
learning_rate=train_params.lr()
base_lr=train_params.base_lr()
weight_decay=train_params.weight_decay()
nr_epochs_per_half_cycle=train_params.nr_epochs_per_half_cycle()
exponential_gamma=train_params.exponential_gamma()
max_training_epochs=train_params.max_training_epochs()
# jitter_xyz=train_params.jitter_xyz()
# jitter_rotation=train_params.jitter_rotation()
# jitter_stretch=train_params.jitter_stretch()
# random_subsample_percentage=train_params.random_subsample_percentage()
# random_noise_stddev=train_params.random_noise_stddev()
save_chekpoint=train_params.save_checkpoint()
checkpoint_path=train_params.checkpoint_path() #where to save models after each epoch if their avg iou is the best until now


model_params=ModelParams.create(config_file)    




#applies some random translations, rotations and scaling to the cloud
# def jitter(cloud):
#     #the order of this operations matter
#     cloud.random_subsample(random_subsample_percentage)
#     cloud.random_stretch(jitter_stretch)
#     cloud.random_noise(random_noise_stddev)
#     cloud.random_rotation(jitter_rotation)
#     cloud.random_translation(jitter_xyz)
#     return cloud


def show_predicted_cloud(pred_softmax, cloud):
    # TIME_START("show_pred")
    mesh_pred=cloud.clone()
    l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy()
    mesh_pred.L_pred=l_pred
    mesh_pred.m_vis.m_point_size=4
    mesh_pred.m_vis.set_color_semanticpred()
    # mesh_pred.move_in_z(cloud.get_scale()) 
    Scene.show(mesh_pred, "mesh_pred")
    # TIME_END("show_pred")

def show_difference_cloud(pred_softmax, cloud):
    mesh_pred=cloud.clone()
    l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy() #(nr_points, )
    l_gt=cloud.L_gt  #(nr_points,1)
    l_pred=np.expand_dims(l_pred,1) #(nr_points,1)

    diff=l_pred!=cloud.L_gt
    diff_repeated=np.repeat(diff, 3,1) #repeat 3 times on axis 1 so to obtain a (nr_points,3)

    mesh_pred.C=diff_repeated
    mesh_pred.m_vis.m_point_size=4
    mesh_pred.m_vis.set_color_pervertcolor()
    # mesh_pred.move_in_z(-cloud.get_scale()) #good for shapenetpartseg
    # mesh_pred.move_in_z(-2.0) #good for shapenetpartseg
    Scene.show(mesh_pred, "mesh_diff")

def show_confidence_cloud(pred_softmax, cloud):
    mesh_pred=cloud.clone()
    l_pred_confidence,_=pred_softmax.detach().exp().max(axis=1)
    l_pred_confidence=l_pred_confidence.cpu().numpy() #(nr_points, )
    l_pred_confidence=np.expand_dims(l_pred_confidence,1) #(nr_points,1)
    l_pred_confidence=np.repeat(l_pred_confidence, 3,1) #repeat 3 times on axis 1 so to obtain a (nr_points,3)

    mesh_pred.C=l_pred_confidence
    mesh_pred.m_vis.m_point_size=4
    mesh_pred.m_vis.set_color_pervertcolor()
    # mesh_pred.move_in_z(-cloud.get_scale()) #good for shapenetpartseg
    # mesh_pred.move_in_z(-1.0) #good for shapenetpartseg
    Scene.show(mesh_pred, "mesh_confidence")


def show_pca_of_features_cloud(per_point_features, cloud):
    mesh=cloud.clone()


    ## http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch
    X=per_point_features.detach().squeeze(0).cpu()#we switch to cpu because svd for gpu needs magma: No CUDA implementation of 'gesdd'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/) at /opt/pytorch/aten/src/THC/generic/THCTensorMathMagma.cu:191
    k=3
    print("x is ", X.shape)
    X_mean = torch.mean(X,0)
    print("x_mean is ", X_mean.shape)
    X = X - X_mean.expand_as(X)

    U,S,V = torch.svd(torch.t(X)) 
    C = torch.mm(X,U[:,:k])
    print("C has shape ", C.shape)
    print("C min and max is ", C.min(), " ", C.max() )
    C-=C.min()
    C/=C.max()
    print("after normalization C min and max is ", C.min(), " ", C.max() )


    mesh.C=C.detach().cpu().numpy()
    mesh.m_vis.m_point_size=10
    mesh.m_vis.set_color_pervertcolor()
    mesh.move_in_z(-2*cloud.get_scale()) #good for shapenetpartseg
    Scene.show(mesh, "mesh_pca")


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
    

#depending on the object loaded by shapenet we need a differnt sigma  in order to get around 20 points per simplex
def set_appropriate_sigma(lattice_to_splat, object_name):
    # airplane, bag, cap, car, chair, earphone, guitar, knife, lamp, motorbike, mug, pistol, rocket, skateboard, table
    if object_name=="airplane":
        lattice_to_splat.lattice.set_sigma(0.035)
    elif object_name=="bag":
        lattice_to_splat.lattice.set_sigma(0.06)
    elif object_name=="cap":
        lattice_to_splat.lattice.set_sigma(0.06)
    elif object_name=="car":
        lattice_to_splat.lattice.set_sigma(0.06)
    elif object_name=="chair":
        lattice_to_splat.lattice.set_sigma(0.055)
    elif object_name=="earphone":
        lattice_to_splat.lattice.set_sigma(0.045)
    elif object_name=="guitar":
        lattice_to_splat.lattice.set_sigma(0.035)
    elif object_name=="knife":
        lattice_to_splat.lattice.set_sigma(0.03)
    elif object_name=="lamp":
        lattice_to_splat.lattice.set_sigma(0.04)
    elif object_name=="laptop":
        lattice_to_splat.lattice.set_sigma(0.06)
    elif object_name=="motorbike":
        lattice_to_splat.lattice.set_sigma(0.045)
    elif object_name=="mug":
        lattice_to_splat.lattice.set_sigma(0.06)
    elif object_name=="pistol":
        lattice_to_splat.lattice.set_sigma(0.04)
    elif object_name=="rocket":
        lattice_to_splat.lattice.set_sigma(0.04)
    elif object_name=="skateboard":
        lattice_to_splat.lattice.set_sigma(0.04)
    elif object_name=="table":
        lattice_to_splat.lattice.set_sigma(0.055)
    else:
        err=object_name+" is not a known object"
        sys.exit(err)



def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../config', config_file)
    if with_viewer:
        view=Viewer.create(config_path)
    loader_train=create_loader(dataset_name, config_path)
    loader_train.set_mode_train()
    # loader_train.set_mode_validation() #DEBUG, this is set now to be in val mode
    # if isinstance(loader_train, DataLoaderSemanticKitti):
        # loader_train.set_sequence("all") #for smenantic kitti
    loader_train.start()

    # loader_test=DataLoaderShapeNetPartSeg(config_file)
    loader_test=create_loader(dataset_name, config_path)
    loader_test.set_mode_test()
    # loader_test.set_mode_train() #DEBUG, this is set now to be in train mode
    # if isinstance(loader_test, DataLoaderSemanticKitti):
        # loader_test.set_sequence("all") #for smenantic kitti
    # if isinstance(loader_test, DataLoaderScanNet):
        # loader_test.set_mode_validation()
    loader_test.start()
    vis=Vis()

    first_time=True
    iter_train_nr=0 #nr of batches processed
    iter_test_nr=0
    samples_training_processed=0 #nr of samples processed (we cna have multiple samples per batch)
    samples_test_processed=0
    epoch_nr=0
    loss_acum_per_epoch = 0.0

    #torch stuff 
    loss_fn = torch.nn.NLLLoss(ignore_index=0) #TODO make it dynamic depending on the labelmngr.get_background_idx
    lattice_to_splat=LatticePy()
    lattice_to_splat.create(config_path, "splated_lattice")
    model_ctx=ModelCtx(base_lr, learning_rate, weight_decay, batch_size, nr_epochs_per_half_cycle, exponential_gamma, model_params, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
    mode="train" #this will switch between train and eval as we finish each epoch
    loader=loader_train #pointer to a loader, we start with the train one
    scores=Scores(node_name, port)
    best_avg_iou=-1

    shapenet_objects=["airplane", "bag", "cap", "car", "chair", "earphone", "guitar", "knife", "lamp", "laptop", "motorbike", "mug", "pistol", "rocket", "skateboard", "table"]
    shapenet_curent_object_idx=0

    # lr_finder=LRFinder(model_ctx, loader_train, loader_test, do_eval=True)
    # lr_finder=LRFinder(model_ctx, loader_train, loader_test, do_eval=False)
    # lr_finder.range_test(config_file, batch_size, start_lr=0.0001, end_lr=0.01, num_steps=100, smooth_f=0.0, diverge_th=15)
    # lr_finder.range_test(config_file, batch_size, start_lr=0.002, end_lr=0.4, num_steps=100, smooth_f=0.0, diverge_th=15)
    #for airplane 1.1e-5 and 0.005
    # for knife is 1.5e-5 and 0.05
    # for knife 5u and 0.002
    # for knife with batch size 8 we ave baselr 2e-6 and max_lr is 0.015
    # for motorbike with batchsize 8 we have 2e-6 and max lr is 0.003
    # for rocket with batchzie of 8 we have 2e-6 and max_lr 0.015
    # for bag with batchsize 8 we have 2e-6 and maxlr 0.02 (with the new resnet block not max_lr should be 0.01)
    #motobike densenet batsize 2
    # return

    # nr_cloud=0
    # max_idx=-1
    # min_idx=100

    while True:
        if with_viewer:
            view.update()

        # if(nr_cloud>42):
        # if(nr_cloud>100):
            # continue


        if(loader.has_data()): 
            cloud=loader.get_cloud()
            print("\n\n\n")
            print("got cloud")
            print("label uindx of the cloud is ", cloud.m_label_mngr.get_idx_unlabeled() )
            # print("scale of the cloud is ", cloud.get_scale())


            if with_viewer:
                cloud.m_vis.m_point_size=4
                # Scene.show(cloud,"cloud")

            #set the lattice sigmas on shapenet depending on the object 
            if isinstance(loader, DataLoaderShapeNetPartSeg):
                set_appropriate_sigma(lattice_to_splat, loader.get_object_name())


            # max_idx= np.maximum(max_idx, cloud.L_gt.max())
            # min_idx= np.minimum(min_idx, cloud.L_gt.min())

            # print("min max idx ", min_idx, max_idx)
            # print("nr of cloud is ", nr_cloud)

            # #get how many points we have for each label
            # nr_classes=cloud.m_label_mngr.nr_classes()
            # if nr_cloud==0:
            #     points_per_label=[0] * nr_classes
            # for i in range(nr_classes):
            #     nr_points_for_class=(cloud.L_gt==i).sum()
            #     points_per_label[i]+=nr_points_for_class

            # print(points_per_label)


            # nr_cloud+=1
        
            # print("max index for l_gt is ", cloud.L_gt.max())

            # print("nr classes is ", cloud.m_label_mngr.nr_classes())

            positions, values, target = model_ctx.prepare_cloud(cloud) #prepares the cloud for pytorch, returning tensors alredy in cuda
            pred_softmax, pred_raw, delta_weight_error_sum=model_ctx.forward(lattice_to_splat, positions, values, mode, cloud.m_label_mngr.nr_classes(), loader_train.nr_samples() )
            # Profiler.print_all_stats()
            if with_viewer:
                show_predicted_cloud(pred_softmax, cloud)
                # show_difference_cloud(pred_softmax, cloud)
                # show_confidence_cloud(pred_softmax, cloud)
                # show_pca_of_features_cloud(model_ctx.per_point_features(), cloud)
            if(mode=="train"):
                loss_per_batch, finished_batch=model_ctx.backward(pred_softmax, pred_raw, target, delta_weight_error_sum, cloud.m_label_mngr.get_idx_unlabeled(), cloud.m_label_mngr.class_frequencies(),  samples_training_processed )
                samples_training_processed+=1
                if finished_batch:
                    # logger_train_loss.log(iter_train_nr, loss_per_batch, name='loss_per_batch_train')
                    vis.log(iter_train_nr, loss_per_batch, "loss_train", "loss_per_batch_train")
                    logger_lr.log(iter_train_nr, model_ctx.get_lr() , name='lr')
                    loss_acum_per_epoch += loss_per_batch
                    iter_train_nr+=1
            elif(mode=="eval"):
                loss_per_batch, finished_batch=model_ctx.loss(pred_softmax, pred_raw, target, cloud.m_label_mngr.get_idx_unlabeled(), cloud.m_label_mngr.class_frequencies(), samples_test_processed )
                samples_test_processed+=1
                if finished_batch:
                    vis.log(iter_test_nr, loss_per_batch, "loss_test", "loss_per_batch_test")
                    iter_test_nr+=1
                scores.accumulate_scores(pred_softmax, target, cloud.m_label_mngr.get_idx_unlabeled() )

            print("model has nr of params ", model_ctx.nr_parameters())

            # model_ctx.summary()
            # model_ctx.print_params()
            print("nr_conv layers is ", model_ctx.nr_convolutional_layers() )


            # sys.exit("debug gather backwards")

            # lattice_to_splat.compute_nr_points_per_lattice_vertex()

        if loader_train.is_finished() and loader==loader_train:
            print("epoch of training is finished")
            #log stuff
            # logger_loss_acumm.log(epoch_nr, loss_acum_per_epoch , name='loss_acum_per_epoch')

            #reset
            loader_train.reset()
            if model_ctx.model is not None:
                if(isinstance(model_ctx.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) ):
                    model_ctx.scheduler.step(loss_acum_per_epoch) #for reduce on plateu
                model_ctx.model.eval() #switch to test mode
            # model_ctx.set_batch_norm(False)
            # sys.exit("done")

            # for child in model.children():
            #     for ii in range(len(child)):
            #         if type(child[ii])==nn.BatchNorm2d:
            #             child[ii].track_running_stats = False

            mode="eval"
            torch.set_grad_enabled(False)
            loader=loader_test 
            #increment bookkeeping stuff
            epoch_nr+=1
            loss_acum_per_epoch=0.0
            model_ctx.loss_per_batch=0 #each sample, this loss gets incremented, now that we start testing, we set it back to zero

     

        if loader_test.is_finished() and loader==loader_test:
            print("epoch of testing is finished")
            loader_test.reset()

            # #save the model if it's the best until now
            # if(scores.avg_class_iou()>best_avg_iou):
            #     best_avg_iou=scores.avg_class_iou()
            #     model_name="model_e_"+str(epoch_nr)+".pt"
            #     out_path=os.path.join(checkpoint_path, model_name)
            #     torch.save(model_ctx.model.state_dict(), out_path)

            #save after each epoch
            if save_chekpoint and model_ctx.model is not None:
                model_name="model_e_"+str(epoch_nr)+"_"+str( scores.avg_class_iou() )+".pt"
                info_txt_name="model_e_"+str(epoch_nr)+"_info"+".csv"
                out_model_path=os.path.join(checkpoint_path, model_name)
                out_info_path=os.path.join(checkpoint_path, info_txt_name)
                torch.save(model_ctx.model.state_dict(), out_model_path)
                scores.write_iou_to_csv(out_info_path)

            #print results
            scores.update_best()
            scores.show(epoch_nr)
            scores.start_fresh_eval() #reset the nr of samples used for computing averages and so on to 0 so that we start accumulating anew on the next epoch

            #switch to training mode
            if model_ctx.model is not None:
                model_ctx.model.train()
            # model_ctx.set_batch_norm(True)
            mode="train"
            torch.set_grad_enabled(True)
            loader=loader_train

            #bookkeeping things
            model_ctx.loss_per_batch=0 #each sample, this loss gets incremented, now that we start training, we set it back to zero

            #finsihed testing
            if (epoch_nr>=max_training_epochs and max_training_epochs>0):
                if isinstance(loader, DataLoaderShapeNetPartSeg) and shapenet_curent_object_idx<len(shapenet_objects):
                    #if we are doing Shapenet then we need to move to the new object and start the epoch nr and model from anew

                    print("finished object of shapenet")
                    first_time=True
                    iter_train_nr=0 #nr of batches processed
                    iter_test_nr=0
                    samples_training_processed=0 #nr of samples processed (we cna have multiple samples per batch)
                    samples_test_processed=0
                    epoch_nr=0
                    loss_acum_per_epoch = 0.0

                    # TODO update lattice sigmas
                    # will be done automaticaly at the beggining

                    #save the best iou 
                    current_object=shapenet_objects[shapenet_curent_object_idx]
                    best_txt_name="best_"+current_object + "_e_" + str(epoch_nr) +".csv"
                    out_best_path=os.path.join(checkpoint_path, best_txt_name)
                    scores.write_best_iou_to_csv(out_best_path)


                    #TODO reset model to initial weights by doing model_ctx.clear()
                    model_ctx.clear()
                    scores.clear()

                    #TODO update the loader label_mngr for the new object, should be done with only just setting loader.start()
                    shapenet_curent_object_idx+=1
                    next_object=shapenet_objects[shapenet_curent_object_idx]
                    print("switching to object", next_object )
                    loader_train.set_object_name(next_object)
                    loader_test.set_object_name(next_object)
                    time.sleep(5) #HACK sleep for a bit to give the loaders time to load something. Otherwise we will check if the loader is finished and the anwer will be true
                else:
                    #we are in any other datset, we get the best iou and the best iou per class and we store it to file
                    print ("finished training")
                    return # we are done training

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
