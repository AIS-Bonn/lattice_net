#!/usr/bin/env python3.6

import torch
from torch.autograd import Function
from torch import Tensor

import sys
import os
import numpy as np
# http://wiki.ros.org/Packages#Client_Library_Support
import rospkg
rospack = rospkg.RosPack()
sf_src_path=rospack.get_path('surfel_renderer')
sf_build_path=os.path.abspath(sf_src_path + "/../../build/surfel_renderer")
sys.path.append(sf_build_path) #contains the modules of pycom

from DataLoaderTest  import *
from vis import Vis

# config_file="lnn_train_semantic_kitti.cfg"
# config_file="lnn_train_semantic_kitti_bg5.cfg"
# config_file="lnn_train_shapenet.cfg"
config_file="lnn_train_stanford.cfg"
# config_file="lnn_train_stanford_bg5.cfg"
# config_file="lnn_train_scannet.cfg"
# config_file="lnn_train_scannet_bg5.cfg"

train_params=TrainParams.create(config_file)    
dataset_name=train_params.dataset_name()
with_viewer=train_params.with_viewer()


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
    loader=create_loader(dataset_name, config_file)
    loader.start()


    vis=Vis()


    min_nr_points=999999999
    max_nr_points=-999999999
    summed_nr_points=0
    nr_processed_clouds=0


    while True:
        if with_viewer:
            view.update()


        if(loader.has_data()): 
            cloud=loader.get_cloud()
            print("\n\n\n")
            print("got cloud")

            # if nr_processed_clouds==2:
                # continue

            if with_viewer:
                cloud.m_vis.m_point_size=4
                # cloud.m_vis.set_color_pervertcolor()
                Scene.show(cloud,"cloud")

            print("got cloud with nr_points", cloud.V.shape[0])
            cur_nr_points=cloud.V.shape[0]
            summed_nr_points+=cur_nr_points
            max_nr_points=np.maximum(cur_nr_points, max_nr_points)
            min_nr_points=np.minimum(cur_nr_points, min_nr_points)
            nr_processed_clouds+=1


            #print results
            print("avg nr of points", summed_nr_points/nr_processed_clouds)
            print("min nr of points", min_nr_points)
            print("max nr of points", max_nr_points)



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