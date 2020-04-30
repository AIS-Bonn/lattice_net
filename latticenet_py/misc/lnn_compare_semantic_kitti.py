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


# mesh=Mesh("/media/rosu/Data/data/semantic_kitti/predictions/final_7_amsgrad_iou/18_pred.ply")
# max_dist=-1
# for i in range(mesh.V.shape[0]):
#     point=mesh.V[i,:]
#     dist=np.linalg.norm(point)
#     if dist>max_dist:
#         max_dist=dist
#     print("max_dist is " , max_dist)

while True:

    if(loader.has_data()): 
        cloud=loader.get_cloud()
        # Scene.show(cloud, "gt")

        #nice 
        seq="19"
        cloud_name="004597"
        # seq="18"
        # cloud_name="001523"

        point_size=12


        #load gt
        # gt=Mesh("/media/rosu/Data/data/semantic_kitti/predictions/after_icra_experiments_fixed_deform_none/test/19/000535_gt.ply")
        gt=Mesh("/media/rosu/Data/data/semantic_kitti/predictions/after_icra_experiments_fixed_deform_none/test/"+seq+"/"+cloud_name+"_gt.ply")
        gt.m_vis.m_point_size=point_size
        Scene.show(gt, "gt")


        #load my prediction
        # my_pred=Mesh("/media/rosu/Data/data/semantic_kitti/predictions/final_7_amsgrad_iou/18_pred.ply")
        # my_pred=Mesh("/media/rosu/Data/data/semantic_kitti/predictions/after_icra_experiments_fixed_deform_none/test/19/000535_pred.ply")
        # my_pred_file="/media/rosu/Data/data/semantic_kitti/predictions/after_icra_experiments_fixed_deform_none/test/19/000535.label"
        my_pred_file="/media/rosu/Data/data/semantic_kitti/predictions/after_icra_experiments_fixed_deform_none/test/"+seq+"/"+cloud_name+".label"
        my_pred=gt.clone()
        my_pred.m_vis.m_point_size=point_size
        f = open(my_pred_file, "r")
        a = np.loadtxt(my_pred_file)
        my_pred.m_label_mngr=cloud.m_label_mngr
        my_pred.L_pred=a
        my_pred.m_vis.set_color_semanticpred()
        Scene.show(my_pred, "my_pred")



        #load also the prediction from splatnet 
        # splatnet_pred_file="/media/rosu/Data/data/semantic_kitti/predictions_from_related_work/splatnet_semantic_kitti_single_frame_final_predictions_11_21/15/predictions/001700.label"
        splatnet_pred_file="/media/rosu/Data/data/semantic_kitti/predictions_from_related_work/splatnet_semantic_kitti_single_frame_final_predictions_11_21/"+seq+"/predictions/"+cloud_name+".label"
        cloud_splatnet=my_pred.clone()
        cloud_splatnet.m_vis.m_point_size=point_size
        f = open(splatnet_pred_file, "r")
        a = np.fromfile(f, dtype=np.uint32)
        cloud_splatnet.L_pred=a
        cloud_splatnet.m_vis.set_color_semanticpred()
        Scene.show(cloud_splatnet, "splatnet")

        #load also the prediction from tangentconv
        # tangentconv_pred_file="/media/rosu/Data/data/semantic_kitti/predictions_from_related_work/tangent_conv_semantic_kitti_single_frame_final_predictions_11_21/15/001700.label"
        tangentconv_pred_file="/media/rosu/Data/data/semantic_kitti/predictions_from_related_work/tangent_conv_semantic_kitti_single_frame_final_predictions_11_21/"+seq+"/"+cloud_name+".label"
        cloud_tangentconv=my_pred.clone()
        cloud_tangentconv.m_vis.m_point_size=point_size
        f = open(tangentconv_pred_file, "r")
        a = np.fromfile(f, dtype=np.uint32)
        cloud_tangentconv.L_pred=a
        cloud_tangentconv.m_vis.set_color_semanticpred()
        Scene.show(cloud_tangentconv, "tangentconv")

        view.m_camera.from_string("3.18081 306.493  215.98  -0.464654 0.00651987 0.00342135 0.885462       0 -6.3551       0 30 0.3 7830.87")



    view.update()