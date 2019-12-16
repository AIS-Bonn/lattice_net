#!/usr/bin/env python3.6

import torch
from torch.autograd import Function
# from torch.autograd import gradcheck
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
from lattice_modules import *
from models import *
from gradcheck_custom import *

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=50000)
# torch.set_printoptions(profile="full")
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)

config_file="lnn_grad_check.cfg"
with_viewer=True


def prepare_cloud(cloud):
    positions=cloud.V
    positions_tensor=torch.from_numpy(positions).unsqueeze(0).float().to("cuda")
    # positions_tensor=positions_tensor[:,:,0:2].clone() #get only the first 2 column because I want to debug some stuff with the coarsening of the lattice
    print("prearing cloud with possitions tensor of shape", positions_tensor.shape)
    values_tensor=torch.ones(1, positions_tensor.shape[1], 1) #not really necessary but at the moment I have no way of passing an empty value array
    # values_tensor=positions_tensor #usualyl makes the things worse... it converges faster to a small loss but not as small as just setting the values to one
    target=cloud.L_gt
    target_tensor=torch.from_numpy(target).long().squeeze(1).to("cuda").squeeze(0)
    # print("maximum class idx is ", target_tensor.max() )

    return positions_tensor, values_tensor, target_tensor


def run_one_coarsen(lv, ls):
    nr_filters=1
    filter_extent=ls.lattice.get_filter_extent(1)
    val_full_dim=ls.lattice.val_full_dim()
    weight = torch.randn( filter_extent * val_full_dim, nr_filters).to("cuda")  #works for ConvIm2RowLattice
    weight.requires_grad=True

    use_center_vertex_from_lattice_neighbours=True
    lv_coarse, ls_coarse= CoarsenLattice.apply( lv,ls, weight, use_center_vertex_from_lattice_neighbours, False, False) 
    return lv_coarse, ls_coarse



def check_slice(lv, ls, positions):
    slice_lattice=SliceLatticeModule(with_homogeneous_coord=False, with_debug_output=True, with_error_checking=False)
    gradcheck(slice_lattice, (lv,ls, positions), eps=1e-5) #diff_norm of 0.0062

def check_conv(lv, ls):
    nr_filters=1
    filter_extent=ls.lattice.get_filter_extent(1)
    val_full_dim=ls.lattice.val_full_dim()
    weight = torch.randn( filter_extent * val_full_dim, nr_filters).to("cuda")  #works for ConvIm2RowLattice
    weight.requires_grad=True
    # gradcheck(ConvIm2RowLattice.apply, (lv,ls, weight, 1, False, None, None, False, False, False), eps=1e-5, atol=1e-2, rtol=1e-2, raise_exception=False)
    gradcheck(ConvIm2RowLattice.apply, (lv,ls, weight, 1, False, None, None, False, False, False), eps=1e-5) #diff_norm of around 0.075 for both the lattice values and the filter_bank

def check_coarsen(lv, ls):
    nr_filters=1
    filter_extent=ls.lattice.get_filter_extent(1)
    val_full_dim=ls.lattice.val_full_dim()
    weight = torch.randn( filter_extent * val_full_dim, nr_filters).to("cuda")  #works for ConvIm2RowLattice
    weight.requires_grad=True

    use_center_vertex_from_lattice_neighbours=True
    gradcheck(CoarsenLattice.apply, (lv,ls, weight, use_center_vertex_from_lattice_neighbours, False, False), eps=1e-5) 

def check_finefy(lv,ls):
    lv_coarse, ls_coarse=run_one_coarsen(lv,ls)


    nr_filters=1
    filter_extent=ls_coarse.lattice.get_filter_extent(1)
    val_full_dim=ls_coarse.lattice.val_full_dim()
    weight = torch.randn( filter_extent * val_full_dim, nr_filters).to("cuda")  #works for ConvIm2RowLattice
    weight.requires_grad=True

    use_center_vertex_from_lattice_neighbours=True
    gradcheck(FinefyLattice.apply, ( lv_coarse, ls_coarse, ls, weight, use_center_vertex_from_lattice_neighbours, True, False ), eps=1e-5) 

def check_slice_classify(lv,ls,positions):

    nr_positions=positions.shape[1]
    pos_dim=positions.shape[2]
    val_full_dim=lv.shape[1]
    print("check slice classify: lv has shape ", lv.shape)

    nr_classes=2
    delta_weights = torch.zeros(1, nr_positions, pos_dim+1).to("cuda") 
    delta_weights.requires_grad=True
    linear_clasify=torch.nn.Linear(val_full_dim, nr_classes, bias=True).to("cuda") 

    gradcheck(SliceClassifyLattice.apply, ( lv, ls, positions, delta_weights, linear_clasify.weight, linear_clasify.bias, nr_classes, True, False  ), eps=1e-5) 

def check_gather(lv,ls,positions):
    gradcheck(GatherLattice.apply, ( lv, ls, positions, True, False  ), eps=1e-5) 

def check_gather_elevated(lv,ls):
    lv_coarse, ls_coarse = run_one_coarsen(lv, ls)
    gradcheck(GatherElevatedLattice.apply, ( ls_coarse, lv, ls, True, False  ), eps=1e-5) 









def run():
    if with_viewer:
        view=Viewer(config_file)
    loader=DataLoaderToyExample(config_file)
    loader.start()

    
    lattice_py=LatticePy()
    lattice_py.create(config_file, "splated_lattice")


    while True:
        if with_viewer:
            view.update()


        if(loader.has_data()): 
            loader_is_reseted=False
            cloud=loader.get_cloud()
            cloud.remove_vertices_at_zero() #we need to remove them otherwise we will have a position at zero which will splat with weight 1.0 onto one vertex and 0.0 in all the others. Can lead to numerical issues
            print("\n\n\n")
            print("got cloud")
            if with_viewer:
                cloud.m_vis.m_point_size=4
                Scene.show(cloud,"cloud")

            #get positions and values 
            positions, values, target=prepare_cloud(cloud)


            # #splat and check how many point we have per lattice vertex
            # lattice_py.begin_splat()
            # lattice_py.splat_standalone(positions_tensor, values_tensor, with_homogeneous_coord=False)
            # print("after splatting lattice_py has values", lattice_py.values()) 
            # print("after splatting indices is ", lattice_py.splatting_indices())
            # print("after splatting weights is ", lattice_py.splatting_weights())


            # #since those values are created with the splat operation which also adds the homogeneous coordinate for which I don't really care about, we replace the values with something else
            # vals=torch.randn(lattice_py.nr_lattice_vertices(), 1).to("cuda")
            # lattice_py.set_values(vals)
            # lattice_py.set_val_dim(vals.shape[1])
            # lattice_py.set_val_full_dim(vals.shape[1])

            # #double check that we managed to splat all positions
            # indices=lattice_py.splatting_indices()
            # nr_invalid_splats=(indices==-1).sum()
            # if nr_invalid_splats>0:
            #     sys.exit("Why are they invalid splats ", nr_invalid_splats)


            # #prepare for gradients
            # lv=lattice_py.values()
            # # lv.requires_grad=True
            # ls=lattice_py

            # #check the gradient of slicing
            # slice_lattice=SliceLatticeModule(with_homogeneous_coord=False, with_debug_output=True, with_error_checking=False)
            # gradcheck(slice_lattice, (lv,ls, positions_tensor), eps=1e-5)


            #check the gradient of conv
            # conv=ConvLatticeModule(nr_filters=1, neighbourhood_size= 1, dilation=1, with_homogeneous_coord=False, with_debug_output=False, with_error_checking=False)
            # gradcheck(conv, (lv,ls, None, None), eps=1e-5)

            # #check the gradient of conv using the function
            # nr_filters=1
            # filter_extent=lattice_py.lattice.get_filter_extent(1)
            # val_full_dim=lattice_py.lattice.val_full_dim()
            # weight = torch.randn( filter_extent * val_full_dim, nr_filters).to("cuda")  #works for ConvIm2RowLattice
            # weight.requires_grad=True
            # gradcheck(ConvIm2RowLattice.apply, (lv,ls, weight, 1, False, None, None, False, False, False), eps=1e-5, atol=1e-2, rtol=1e-2)


            #check grad of a whole model (doesnt work because it would return the gradient of only the values_tensor and that is not what we want)
            # model=LNN_gradcheck(2, with_debug_output=False, with_error_checking=False)
            # values_tensor.requires_grad=True
            # gradcheck(model, (ls,positions_tensor, values_tensor), eps=1e-5, atol=1e-2, rtol=1e-2)

            #create a lattice by distributing values
            distribute=DistributeLatticeModule(with_debug_output=False, with_error_checking=False) 
            point_net=PointNetDenseModule( growth_rate=1, nr_layers=1, nr_outputs_last_layer=3, with_debug_output=False, with_error_checking=False) 
            distributed, indices=distribute(lattice_py, positions, values)
            lv, ls=point_net(lattice_py, distributed, indices)
            # lv.fill_(0)


            # check_slice(lv, ls, positions)
            # check_conv(lv, ls)
            # check_coarsen(lv, ls) 
            # check_finefy(lv, ls) 
            # check_slice_classify(lv, ls, positions)
            # check_gather(lv, ls, positions)
            check_gather_elevated(lv, ls)

            print("FINISHED GRAD CHECK")

            # run_one_coarsen(lv,ls)
            return

            
      
            

        if loader.is_finished():
            if loader_is_reseted==False: #to avoid resetting multiple times while we are in this loop
                loader.reset()

               

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