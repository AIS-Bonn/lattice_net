#!/usr/bin/env python3

import os
import numpy as np
import sys
try:
  import torch
except ImportError:
    pass
from easypbr  import *
from dataloaders import *


config_file="lnn_check_lattice_size.cfg"

config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../config', config_file)
view=Viewer.create(config_path) #first because it needs to init context

loader=DataLoaderScanNet(config_path)
loader.start()

nr_points_in_radius=[]

while True:
    if(loader.has_data()): 
        cloud=loader.get_cloud()
        Scene.show(cloud, "cloud")

        random_point=cloud.V[1,:]
        # print("random point is ", random_point)
        nr_points=cloud.radius_search(random_point, 0.05)

        nr_points_in_radius.append(nr_points)
        print("mean_nr_points: ", np.mean(nr_points_in_radius))

        
    view.update()