#!/usr/bin/env python3.6

import torch
from torch.autograd import Function
from torch import Tensor

import sys
import os
import numpy as np
import time
# http://wiki.ros.org/Packages#Client_Library_Support
import rospkg
rospack = rospkg.RosPack()
sf_src_path=rospack.get_path('surfel_renderer')
sf_build_path=os.path.abspath(sf_src_path + "/../../build/surfel_renderer")
sys.path.append(sf_build_path) #contains the modules of pycom

from DataLoaderTest  import *
from vis import Vis

config_file="lnn_make_teaser_img.cfg"

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
    # loader.set_mode_test()
    loader.start()


    # view.m_camera.from_string("-0.0229652 -0.0329689    5.88696   0.153809  -0.457114 -0.0806017 0.872292 5.88928 30 0.5 500") #for moto train 9
    view.m_camera.from_string("-0.0229652 -0.0329689    5.88696   0.153809  -0.457114 -0.0806017 0.872292 5.88928 30 0.5 500") #for moto train 10

    nr_processed_clouds=-1
    chosen_cloud=9 # train mode: nice scooter
    # chosen_cloud=10 #train mode: nice looking motorbike
    # chosen_cloud=41

    while True:
        if with_viewer:
            view.update()


        if(loader.has_data()): 
            cloud=loader.get_cloud()
            print("\n\n\n")
            print("got cloud")

            nr_processed_clouds+=1

            if nr_processed_clouds!=chosen_cloud:
                continue
            # time.sleep(2) 


            if with_viewer:
                cloud.m_vis.m_point_size=9
                # cloud.m_vis.set_color_pervertcolor()
                Scene.show(cloud,"cloud")

            print("showing cloud ", nr_processed_clouds)




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