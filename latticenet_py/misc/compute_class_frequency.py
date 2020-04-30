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


config_file="compute_class_frequency.cfg"




def run():
    view=Viewer(config_file)
    # loader=DataLoaderSemanticKitti(config_file)
    loader=DataLoaderShapeNetPartSeg(config_file)
    loader.start()
    nr_total_points=0
    nr_points_of_class=None
    nr_classes=0


    while True:
        view.update()

        if(loader.has_data()): 
            print("got cloud")
            cloud=loader.get_cloud()
            cloud.m_vis.m_point_size=4
            Scene.show(cloud,"cloud")

            nr_classes=cloud.m_label_mngr.nr_classes()
            if nr_points_of_class is None:
                nr_points_of_class=np.zeros(nr_classes)
            nr_total_points+=cloud.V.shape[0]
            # print("adding to total nr of points", cloud.V.shape[0], " updated is ", nr_total_points )
            for i in range(nr_classes):
                nr_points_of_class[i]+=(cloud.L_gt==i).sum()
                # print("adding for class ", i, " nr of points ", (cloud.L_gt==i).sum(), " updated nr is now ", nr_points_of_class[i]  )

        if loader.is_finished():
            print("frequencies are:")
            for i in range(nr_classes):
                print(nr_points_of_class[i]/nr_total_points)
            # return


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