#!/usr/bin/env python3.6

import torch
from torch.autograd import Function
from torch import Tensor

import sys
import os
# http://wiki.ros.org/Packages#Client_Library_Support
import rospkg
rospack = rospkg.RosPack()
sf_src_path=rospack.get_path('surfel_renderer')
sf_build_path=os.path.abspath(sf_src_path + "/../../build/surfel_renderer")
sys.path.append(sf_build_path) #contains the modules of pycom

from DataLoaderTest  import *
from lattice_py import LatticePy
from lattice_funcs import * #for the timing macros
import visdom
import torchnet
from lr_finder import LRFinder
from scores import Scores
from model_ctx import ModelCtx

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=50000)
# torch.set_printoptions(profile="full")

# config_file="lnn_train_semantic_kitti.cfg"
# config_file="lnn_eval_semantic_kitti_bg5.cfg"
config_file="lnn_eval_single_mesh.cfg"
# config_file="lnn_eval_scannet_bg5.cfg"


torch.manual_seed(0)

node_name="eval_lnn"
vis = visdom.Visdom()
port=8097
logger_test_loss = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_test_loss'}, port=port, env=node_name)

#initialize the parameters used for evaluating
eval_params=EvalParams.create(config_file)    
dataset_name=eval_params.dataset_name()
with_viewer=eval_params.with_viewer()
with_viewer=eval_params.with_viewer()
checkpoint_path=eval_params.checkpoint_path()
do_write_predictions=eval_params.do_write_predictions()
output_predictions_path=eval_params.output_predictions_path()
#rest of params needed by the net to will not be used since we are not training
batch_size=1
learning_rate=0
base_lr=0
weight_decay=0
nr_epochs_per_half_cycle=0
exponential_gamma=0.0
max_training_epochs=0
with_debug_output=False
with_error_checking=False

model_params=ModelParams.create(config_file)    

def show_predicted_cloud(pred_softmax, cloud):
    mesh_pred=cloud.clone()
    l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy()
    mesh_pred.L_pred=l_pred
    mesh_pred.m_vis.m_point_size=4
    mesh_pred.m_vis.set_color_semanticpred()
    mesh_pred.move_in_z(cloud.get_scale()) #good for shapenetpartseg
    Scene.show(mesh_pred, "mesh_pred")


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
    elif dataset_name=="toyexample":
        loader=DataLoaderToyExample(config_file)
    elif dataset_name=="stanford":
        loader=DataLoaderStanfordIndoor(config_file)
    elif dataset_name=="scannet":
        loader=DataLoaderScanNet(config_file)
    else:
        err="Datset name not recognized. It is " + dataset_name
        sys.exit(err)

    return loader


def run():
    if with_viewer:
        view=Viewer(config_file)

    loader_test=create_loader(dataset_name, config_file)
    loader_test.start()

    first_time=True
    iter_test_nr=0 #nr of batches processed
    samples_test_processed=0

    #torch stuff 
    lattice_to_splat=LatticePy()
    lattice_to_splat.create(config_file, "splated_lattice") #IMPORTANT THAT the size of the lattice in this file is the same as the size of the lattice that was used during training
    model_ctx=ModelCtx(base_lr, learning_rate, weight_decay, batch_size, nr_epochs_per_half_cycle, exponential_gamma, model_params, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
    mode="eval" #this will switch between train and eval as we finish each epoch

    torch.set_grad_enabled(False)
    scores=Scores(node_name, port)

    write_scannet_predictions=False

    cloud=MeshCore()
    cloud.load_from_file("/home/local/staff/rosu/data/ais_kitchen/kitchen_meshes/kitchen_2105.ply")
    cloud.random_subsample(0.7)
    cloud.rotate_x_axis(1.0)
    cloud.C=cloud.C/255
    cloud.C=cloud.C+0.2 # for some reason this yields better results
    # nr_classes=21
    #use the dataloader to get the labelmngr
    while True:
        if(loader_test.has_data()): 
            cloud_with_label_mngr=loader_test.get_cloud()
            cloud.m_label_mngr=cloud_with_label_mngr.m_label_mngr
            break

    # for sigma in np.arange(0.04, 0.1, 0.01):   
        # lattice_to_splat.lattice.set_sigma(sigma)
    # sigma=0.07
    # lattice_to_splat.lattice.set_sigma(sigma)

    positions, values, target = model_ctx.prepare_cloud(cloud) #prepares the cloud for pytorch, returning tensors alredy in cuda
    pred_softmax, pred_raw, delta_weight_error_sum=model_ctx.forward(lattice_to_splat, positions, values, mode, cloud.m_label_mngr.nr_classes(), 1 )

    
    if first_time:
        first_time=False
        #now that all the parameters are created we can fill them with a model from a file
        model_ctx.model.load_state_dict(torch.load(checkpoint_path))
        #need to rerun forward with the new parameters to get an accurate prediction
        pred_softmax, pred_raw, delta_weight_error_sum=model_ctx.forward(lattice_to_splat, positions, values, mode, cloud.m_label_mngr.nr_classes(), 1 )

    lattice_to_splat.compute_nr_points_per_lattice_vertex()
    print("max color is ", cloud.C.max() )


    if with_viewer:
        show_predicted_cloud(pred_softmax, cloud)

    if do_write_predictions:
        pred_path=os.path.join(output_predictions_path, str(samples_test_processed)+"_2105_pred.ply" )
        print("writing prediction to ", pred_path)
        write_prediction(pred_softmax, cloud, pred_path)
        # write_gt(cloud, gt_path)


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
