#!/usr/bin/env python3.6


import os
import numpy as np
import sys
try:
  import torch
except ImportError:
    pass
from easypbr  import *
from dataloaders import *


config_file="lnn_compare_semantic_kitti.cfg"
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../config', config_file)
view=Viewer.create(config_path) #first because it needs to init context

loader=DataLoaderSemanticKitti(config_path)
loader.start()

arr=np.load("/home/rosu/Downloads/prediction.npz")
data=arr["arr_0"]
print("data has shape ", data.shape)

mesh=Mesh()
mesh.V=data[:,0:3]
print("V is ", mesh.V.shape)
mesh.L_pred=data[:,3:4]
print("pred is ", mesh.L_pred.shape)


while True:

    if(loader.has_data()): 
        cloud=loader.get_cloud()

        mesh.m_label_mngr=cloud.m_label_mngr
        mesh.m_vis.m_show_points=True
        Scene.show(mesh,"mesh")

    view.update()