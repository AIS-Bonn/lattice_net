#!/usr/bin/env python3.6

import torch
from torch.autograd import Function
from torch import Tensor

import sys
import os
import numpy as np
from matplotlib import pyplot as plt 
# http://wiki.ros.org/Packages#Client_Library_Support
import rospkg
rospack = rospkg.RosPack()
sf_src_path=rospack.get_path('surfel_renderer')
sf_build_path=os.path.abspath(sf_src_path + "/../../build/surfel_renderer")
sys.path.append(sf_build_path) #contains the modules of pycom

from DataLoaderTest  import *
from lattice_py import LatticePy
import visdom
import torchnet
from lr_finder import LRFinder
from scores import Scores
from model_ctx import ModelCtx
from lattice_funcs import *

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=50000)
# torch.set_printoptions(profile="full")
# torch.autograd.set_detect_anomaly(True)

config_file="lnn_train_semantic_kitti.cfg"
#config_file="lnn_train_semantic_kitti_bg5.cfg"
# config_file="lnn_train_shapenet.cfg"
with_viewer=True
desired_mean_points_per_vertex=20
sigma_stepsize=0.01



torch.manual_seed(0)


node_name="lnn"
vis = visdom.Visdom()
port=8097
logger_mean_points = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_mean_points'}, port=port, env='scale_finder_'+node_name)
logger_max_points = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_max_points'}, port=port, env='scale_finder_'+node_name)
logger_median_points = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_median_points'}, port=port, env='scale_finder_'+node_name)
logger_sigma = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_sigma'}, port=port, env='scale_finder_'+node_name)

# loader=DataLoaderSemanticKitti(config_file)



def run():
    if with_viewer:
        view=Viewer(config_file)
    # loader=DataLoaderShapeNetPartSeg(config_file)
    loader=DataLoaderSemanticKitti(config_file)
    loader.start()

    
    lattice_py=LatticePy()
    lattice_py.create(config_file, "splated_lattice")

    iter_nr=0
    samples_nr=0
    loader_is_reseted=False
    mean_points_per_epoch_accum=0
    max_points_per_epoch_accum=0
    median_points_per_epoch_accum=0

    while True:
        if with_viewer:
            view.update()


        if(loader.has_data()): 
            loader_is_reseted=False
            cloud=loader.get_cloud()
            print("\n\n\n")
            print("got cloud")
            print("scale of the cloud is ", cloud.get_scale())


            #get positions and values 
            positions=cloud.V
            positions_tensor=torch.from_numpy(positions).unsqueeze(0).float().to("cuda")
            print("prearing cloud with possitions tensor of shape", positions_tensor.shape)
            values_tensor=torch.ones(1, positions_tensor.shape[1], 1) #not really necessary but at the moment I have no way of passing an empty value array


            #splat and check how many point we have per lattice vertex
            lattice_py.begin_splat()
            lattice_py.splat_standalone(positions_tensor, values_tensor, with_homogeneous_coord=False)
            # lattice_py.splat_standalone(positions_tensor, values_tensor, with_homogeneous_coord=False)
            # lattice_py.splat_standalone(positions_tensor, values_tensor, with_homogeneous_coord=False)
            indices=lattice_py.splatting_indices()


            #more debug shit
            torch.set_printoptions(profile="full")
            # print("indices is ", lattice_py.splatting_indices() )
            torch.set_printoptions(profile="default")

            #more debug. If I unique the splatting_indices I should get a vector in the same size as the nr of lattice vertices
            indices_unique=torch.unique(indices,dim=0)
            print("indices has shape, ", indices.shape)
            print("indices_unique has shape ", indices_unique.shape)





            mean_points, max_points, min_points, median_points, nr_points_per_lattice_vertex =lattice_py.compute_nr_points_per_lattice_vertex() 
            mean_points=mean_points.item()
            max_points=max_points.item()
            median_points=median_points.item()
            mean_points_per_epoch_accum+=mean_points
            max_points_per_epoch_accum+=max_points
            median_points_per_epoch_accum+=median_points


            nr_of_vertices_with_one_point= (nr_points_per_lattice_vertex==1).sum().item()
            print("nr of vertices with only one point in them is ", nr_of_vertices_with_one_point)

            #get the number of position that didnt splat onto a vertex
            pos_dim=positions_tensor.shape[2]
            #indices has size n_positions * (m_pos_dim+1) if a position was not splatted at all then the whole row will sum to -(m_pos_dim+1)
            indices=indices.view(positions_tensor.shape[1], pos_dim+1)
            indices_min,_=indices.min(1) # this will yield a vector of nr_positions, if at that position any one of the m_pos_dim+1 vertices we couldnt splat into the min will give us a -1
            nr_not_splatted=(indices_min<0).sum().item()
            print("nr of position is ", positions_tensor.shape[1], " of which nr of not splatted is ", nr_not_splatted)
            #color the points that are not splatted at all
            not_splatted_at_all=(indices_min<0)*1 # a vector of nr_positions
            print("not splatted_at_all", not_splatted_at_all)
            not_splatted_at_all=not_splatted_at_all.unsqueeze(1) # nr_positionsx1
            print("not splatted at all should be " , (not_splatted_at_all>0).sum() )
            not_splatted_at_all=not_splatted_at_all.cpu().numpy()
            C=np.repeat(not_splatted_at_all, 3,1) #repeat 3 times on axis 1 so to obtain a (nr_points,3)
            cloud.C=C
            cloud.m_vis.set_color_pervertcolor()

            # #try out the torch unique (takes waaay to long)
            # TIME_START("unique")
            # # keys=torch.randint(0, 5, (123782*4,3)) # (nr_positions x nr_vertices_per_simplex) x key_dim(pos_dim)
            # keys=torch.randint(0, 5, (18000,3)) # (nr_positions x nr_vertices_per_simplex) x key_dim(pos_dim)
            # print("keys is ", keys.shape) 
            # keys_unique=torch.unique(keys,dim=0)
            # print("keys_unique is ", keys_unique.shape)
            # TIME_END("unique")




            # plt.hist(nr_points_per_lattice_vertex.cpu().numpy(), bins = np.arange(max_points) ) 
            # plt.title("histogram") 
            # plt.draw()
            # plt.pause(0.001)
            # plt.clf()
             

            # if mean_points>desired_mean_points_per_vertex:
            #     lattice_py.lattice.increase_sigmas(-sigma_stepsize)
            # else:
            #     lattice_py.lattice.increase_sigmas(sigma_stepsize)
            # sigma=lattice_py.lattice.sigmas_tensor()[0].item()
            if with_viewer:
                cloud.m_vis.m_point_size=4
                Scene.show(cloud,"cloud")
            
            iter_nr+=1
            samples_nr+=1

        if loader.is_finished():
            if loader_is_reseted==False: #to avoid resetting multiple times while we are in this loop
                loader.reset()


                #decide on weathre we increase or decrease the sigma
                mean_points=mean_points_per_epoch_accum/samples_nr
                max_points=max_points_per_epoch_accum/samples_nr
                median_points=median_points_per_epoch_accum/samples_nr
                if mean_points>desired_mean_points_per_vertex:
                    lattice_py.lattice.increase_sigmas(-sigma_stepsize)
                else:
                    lattice_py.lattice.increase_sigmas(sigma_stepsize)
                sigma=lattice_py.lattice.sigmas_tensor()[0].item()
            
                logger_mean_points.log(iter_nr, mean_points, name='mean_points')
                logger_max_points.log(iter_nr, max_points, name='max_points')
                logger_median_points.log(iter_nr, median_points, name='median_points')
                logger_sigma.log(iter_nr, sigma, name='sigma')


                samples_nr=0
                loader_is_reseted=True
                mean_points_per_epoch_accum=0
                max_points_per_epoch_accum=0
                median_points_per_epoch_accum=0

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