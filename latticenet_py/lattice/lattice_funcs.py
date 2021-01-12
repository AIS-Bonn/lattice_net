import torch
from torch.autograd import Function
from torch import Tensor

import sys
from easypbr  import Profiler
from latticenet  import HashTable
from latticenet  import Lattice
import numpy as np
import time
import math
import torch_scatter
from latticenet_py.lattice.lattice_wrapper import LatticeWrapper

#Just to have something close to the macros we have in c++
def profiler_start(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.start(name)
def profiler_end(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.end(name)
TIME_START = lambda name: profiler_start(name)
TIME_END = lambda name: profiler_end(name)


        

class SplatLattice(Function):
    @staticmethod
    def forward(ctx, lattice, positions, values):

        lattice.begin_splat()
        splatting_indices, splatting_weights=lattice.splat_standalone(positions, values )

        return lattice.values(), LatticeWrapper.wrap(lattice), splatting_indices, splatting_weights #Pytorch functions requires all the return values to be torch.Variables so we wrap the lattice into one


    @staticmethod
    def backward(ctx, grad_lattice_values, grad_lattice_structure):
              
        return None, None, None

       



class DistributeLattice(Function):
    @staticmethod
    def forward(ctx, lattice, positions, values, experiment):

        lattice.begin_splat()
        distributed, splatting_indices, splatting_weights = lattice.distribute(positions, values)


        #subsctract mean from the positions so we have something like a local laplacian as a feature
        experiments_that_imply_no_mean_substraction=["pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat"]
        # indices=lattice_py.splatting_indices()
        pos_dim=positions.shape[1]
        distributed_positions=distributed[:,:pos_dim] #get the first 3 columns, the ones corresponding only to the xyz positions

        indices_long=splatting_indices.long()

        #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long<0]=0

        if experiment in experiments_that_imply_no_mean_substraction:
            # print("not performing mean substraction as the experiment is ", experiment)
            pass
        else:
            mean_positions = torch_scatter.scatter_mean(distributed_positions, indices_long, dim=0 )
            # mean_positions[0,:]=0 #the first lattice vertex corresponds to the invalid points, the ones that had an index of -1. We set it to 0 so it doesnt affect the prediction or the batchnorm
            index = torch.tensor([0]).to("cuda")
            mean_positions=torch.index_fill(mean_positions, dim=0, index=index, value=0) 
            #by setting the first row of mean_positions to 0 it means that all the point that splat onto vertex zero will have a wrong mean. We will set those distributed_mean_substracted to also zero later
            #the distributed means now has shape nr_positions x pos_dim but we want to substract each distributed position (shape  (nr_positions x m_pos_dim+1) x pos_dim   ) with its corresponding mean. We can do a index_select with splatting indices to get the means
            distributed_mean_positions=torch.index_select(mean_positions, 0, indices_long)
            distributed[:,:pos_dim]=distributed_positions-distributed_mean_positions

        #we have to set the positions that ended up in an invalid vertes or the zero one because it's also considered invalid, to zero
        positions_that_splat_onto_vertex_zero_or_are_invalid=indices_long==0
        positions_that_splat_onto_vertex_zero_or_are_invalid=positions_that_splat_onto_vertex_zero_or_are_invalid.unsqueeze(1)


   
        # distributed.masked_fill_(positions_that_splat_onto_vertex_zero_or_are_invalid, 0)
        distributed=distributed.masked_fill(positions_that_splat_onto_vertex_zero_or_are_invalid, 0)


        return distributed, splatting_indices, splatting_weights


    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None, None


class ConvIm2RowLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values, lattice, filter_bank, dilation):
       
        lattice.set_values(lattice_values)
        # if(lattice_neighbours_structure is not None):
            # lattice_neighbours_structure.set_values(lattice_neighbours_values)

        convolved_lattice=lattice.convolve_im2row_standalone(filter_bank, dilation, lattice, False)
        

        # values=convolved_lattice.values()


        ctx.save_for_backward(filter_bank, lattice_values ) 
        ctx.lattice=lattice
        # ctx.lattice_neighbours_structure=lattice_neighbours_structure
        ctx.filter_extent=int(filter_bank.shape[0]/lattice_values.shape[1])
        ctx.nr_filters= int(filter_bank.shape[1])#i hope it doesnt leak any memory
        ctx.dilation=dilation
        # if lattice_neighbours_structure!=None:
            # ctx.val_dim= lattice_neighbours_structure.val_dim()
        # else: 
        ctx.val_dim= lattice.val_dim()

        # help(convolved_lattice_py)
        # help(torch.autograd.Variable)

        return convolved_lattice.values(), LatticeWrapper.wrap(convolved_lattice) #Pytorch functions requires all the return values to be torch.Variables so we wrap the lattice into one

    @staticmethod
    def backward(ctx, grad_lattice_values, grad_lattice_structure):
        


        lattice=ctx.lattice
        # lattice_neighbours_structure=ctx.lattice_neighbours_structure
        filter_extent=ctx.filter_extent
        nr_filters=ctx.nr_filters
        dilation=ctx.dilation
        val_dim=ctx.val_dim
        filter_bank, lattice_values =ctx.saved_tensors
        filter_extent=int(filter_bank.shape[0]/val_dim)

        #reconstruct lattice_rowified 
        lattice.set_values(lattice_values)
        # if(lattice_neighbours_structure is not None):
            # lattice_neighbours_structure.set_values(lattice_neighbours_values)
        lattice_rowified= lattice.im2row(lattice, filter_extent, dilation, False)



        grad_filter=lattice_rowified.transpose(0,1).mm(grad_lattice_values) 



        # #attempt 4 for grad_lattice
        filter_bank_backwards=filter_bank.transpose(0,1) # creates a nr_filters x filter_extent * val_fim  
        filter_bank_backwards=filter_bank_backwards.view(nr_filters,filter_extent,val_dim) # nr_filters x filter_extent x val_fim  
        filter_bank_backwards=filter_bank_backwards.transpose(0,1).contiguous()  #makes it filter_extent x nr_filters x val_fim   #TODO the contigous may noy be needed because the reshape does may do a contigous if needed or may also just return a view, both work
        filter_bank_backwards=filter_bank_backwards.reshape(filter_extent*nr_filters, val_dim)
        lattice.set_values(grad_lattice_values)
        grad_lattice=lattice.convolve_im2row_standalone(filter_bank_backwards, dilation, lattice, True)
        grad_lattice_values=grad_lattice.values()



        ctx.lattice=0 #release this object so it doesnt leak
        # ctx.lattice_neighbours_structure=0

        return grad_lattice_values, None, grad_filter,  None, None, None, None, None, None, None


class CoarsenLattice(Function):
    @staticmethod
    def forward(ctx, lattice_fine_values, lattice_fine_structure, filter_bank):
        lattice_fine_structure.set_values(lattice_fine_values)

        #create a structure for the coarse lattice, the values of the coarse vertices will be zero
        positions=lattice_fine_structure.positions()

        # print("fine lattice has keys", lattice_fine_structure.keys()) 
        #coarsened_lattice_py=lattice_fine_structure.create_coarse_verts()
        # print("lattice fine structure has indices", lattice_fine_structure.splatting_indices())
        coarsened_lattice=lattice_fine_structure.create_coarse_verts_naive(positions)
      



        #convolve at this lattice vertices with the neighbours from lattice_fine
        dilation=1
        convolved_lattice=coarsened_lattice.convolve_im2row_standalone(filter_bank, dilation, lattice_fine_structure, False)
     

        
        ctx.save_for_backward(filter_bank, lattice_fine_values ) 
        ctx.coarsened_lattice=coarsened_lattice
        ctx.lattice_fine_structure=lattice_fine_structure
        ctx.filter_extent=int(filter_bank.shape[0]/lattice_fine_values.shape[1])
        ctx.nr_filters= int(filter_bank.shape[1])#i hope it doesnt leak any memory
        ctx.dilation=dilation
        ctx.val_dim= lattice_fine_structure.val_dim()

     

        return convolved_lattice.values(), LatticeWrapper.wrap(convolved_lattice) #Pytorch functions requires all the return values to be torch.Variables so we wrap the lattice into one

    @staticmethod
    def backward(ctx, grad_lattice_values, grad_lattice_structure):
        


        coarsened_lattice=ctx.coarsened_lattice
        lattice_fine_structure=ctx.lattice_fine_structure
        filter_extent=ctx.filter_extent
        nr_filters=ctx.nr_filters
        dilation=ctx.dilation
        val_dim=ctx.val_dim
  
        filter_bank, lattice_fine_values =ctx.saved_tensors
        filter_extent=int(filter_bank.shape[0]/val_dim)

        #reconstruct lattice_rowified 
        lattice_fine_structure.set_values(lattice_fine_values)
        lattice_rowified= coarsened_lattice.im2row(lattice_fine_structure, filter_extent, dilation, False)


        grad_filter=lattice_rowified.transpose(0,1).mm(grad_lattice_values) 

        filter_bank_backwards=filter_bank.transpose(0,1) # creates a nr_filters x filter_extent * val_fim  
        filter_bank_backwards=filter_bank_backwards.view(nr_filters,filter_extent,val_dim) # nr_filters x filter_extent x val_fim  
        filter_bank_backwards=filter_bank_backwards.transpose(0,1).contiguous()  #makes it filter_extent x nr_filters x val_fim   #TODO the contigous may noy be needed because the reshape does may do a contigous if needed or may also just return a view, both work
        filter_bank_backwards=filter_bank_backwards.reshape(filter_extent*nr_filters, val_dim)
        coarsened_lattice.set_values(grad_lattice_values)
        # lattice_fine_structure.set_val_dim(nr_filters) #setting val full dim to nr of filters because we will convolve the values of grad_lattice values and those have a row of size nr_filters
        #one hast o convolve at the fine positions, having the neighbour as the coarse ones because they are the ones with the errors
        grad_lattice_py=lattice_fine_structure.convolve_im2row_standalone(filter_bank_backwards, dilation,  coarsened_lattice, True)
        grad_lattice=grad_lattice_py.values()

     
        #release
        ctx.coarsened_lattice=0
        ctx.lattice_fine_structure=0



        return grad_lattice, None, grad_filter, None, None, None #THe good one
       

class FinefyLattice(Function):
    @staticmethod
    def forward(ctx, lattice_coarse_values, lattice_coarse_structure, lattice_fine_structure,  filter_bank ):
        lattice_coarse_structure.set_values(lattice_coarse_values)
        # lattice_fine_structure.set_val_dim(lattice_coarse_structure.val_dim())


        dilation=1
        convolved_lattice=lattice_fine_structure.convolve_im2row_standalone(filter_bank, dilation,lattice_coarse_structure, False)
       
        # values=convolved_lattice_py.values()
        # convolved_lattice_py.set_values(values)

        
        ctx.save_for_backward(filter_bank, lattice_coarse_values) 
        ctx.lattice_fine_structure=convolved_lattice
        ctx.lattice_coarse_structure=lattice_coarse_structure
        ctx.filter_extent=int(filter_bank.shape[0]/lattice_coarse_values.shape[1])
        ctx.nr_filters= int(filter_bank.shape[1])#i hope it doesnt leak any memory
        ctx.dilation=dilation
        ctx.val_dim= lattice_coarse_structure.val_dim()


        return convolved_lattice.values(), LatticeWrapper.wrap(convolved_lattice) #Pytorch functions requires all the return values to be torch.Variables so we wrap the lattice into one

    @staticmethod
    def backward(ctx, grad_lattice_values, grad_lattice_structure):

        # coarsened_lattice_py=ctx.coarsened_lattice_py
        lattice_fine_structure=ctx.lattice_fine_structure
        lattice_coarse_structure=ctx.lattice_coarse_structure
        filter_extent=ctx.filter_extent
        nr_filters=ctx.nr_filters
        dilation=ctx.dilation
        val_dim=ctx.val_dim
        filter_bank, lattice_coarse_values =ctx.saved_tensors
        filter_extent=int(filter_bank.shape[0]/val_dim)

        #reconstruct lattice_rowified 
        lattice_coarse_structure.set_values(lattice_coarse_values)
        lattice_rowified= lattice_fine_structure.im2row(lattice_coarse_structure, filter_extent, dilation, False)


        grad_filter=lattice_rowified.transpose(0,1).mm(grad_lattice_values) 

        filter_bank_backwards=filter_bank.transpose(0,1) # creates a nr_filters x filter_extent * val_fim  
        filter_bank_backwards=filter_bank_backwards.view(nr_filters,filter_extent,val_dim) # nr_filters x filter_extent x val_fim  
        filter_bank_backwards=filter_bank_backwards.transpose(0,1).contiguous()  #makes it filter_extent x nr_filters x val_fim   #TODO the contigous may noy be needed because the reshape does may do a contigous if needed or may also just return a view, both work
        filter_bank_backwards=filter_bank_backwards.reshape(filter_extent*nr_filters, val_dim)
        # print("finefy backwards: saved for backwards a coarsened lattice py with nr of keys", coarsened_lattice_py.nr_lattice_vertices())
        lattice_fine_structure.set_values(grad_lattice_values)
        # lattice_coarse_structure.set_val_dim(lattice_fine_structure.val_dim()) #setting val full dim to nr of filters because we will convolve the values of grad_lattice values and those have a row of size nr_filters
        #one hast o convolve at the fine positions, having the neighbour as the coarse ones because they are the ones with the errors
        grad_lattice_py=lattice_coarse_structure.convolve_im2row_standalone(filter_bank_backwards, dilation,  lattice_fine_structure, True)
        grad_lattice=grad_lattice_py.values()


        #release
        ctx.lattice_coarse_structure=0
        ctx.lattice_fine_structure=0


        return grad_lattice, None, None, grad_filter, None, None, None #THe good one




class SliceLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values, lattice_structure, positions, splatting_indices=None, splatting_weights=None):



        #attempt 2
        lattice_structure.set_values(lattice_values)
        # lattice_structure.set_val_dim(lattice_values.shape[1])

        if splatting_indices==None and splatting_weights==None:
            sliced_values, splatting_indices, splatting_weights=lattice_structure.slice_standalone_no_precomputation(positions )
        else: 
            sliced_values=lattice_structure.slice_standalone_with_precomputation(positions, splatting_indices, splatting_weights  )

        ctx.save_for_backward(positions, sliced_values, splatting_indices, splatting_weights )
        ctx.lattice_structure = lattice_structure


        return sliced_values



       
    @staticmethod
    def backward(ctx, grad_sliced_values):
        

      
        positions, sliced_values_hom, splatting_indices, splatting_weights =ctx.saved_tensors
        lattice_structure = ctx.lattice_structure

        # lattice_py.set_splatting_indices(splatting_indices)
        # lattice_py.set_splatting_weights(splatting_weights)


 
        if(lattice_structure.val_dim() is not grad_sliced_values.shape[1]):
            sys.exit("for some reason the values stored in the lattice are not the same dimension as the gradient. What?")
        # lattice_py.set_val_dim(grad_sliced_values.shape[1])
        lattice_structure.slice_backwards_standalone_with_precomputation_no_homogeneous(positions, grad_sliced_values, splatting_indices, splatting_weights) 
        lattice_values=lattice_structure.values() #we get a pointer to the values so they don't dissapear when we realease the lettice
       
        ctx.lattice_structure=0 # release the pointer to this so it gets cleaned up
       


        return lattice_values, None, None, None, None, None, None


class SliceClassifyLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values, lattice_structure, positions, delta_weights, linear_clasify_weight, linear_clasify_bias, nr_classes, splatting_indices, splatting_weights):


        lattice_structure.set_values(lattice_values)
        # lattice_structure.set_val_dim(lattice_values.shape[1])

        initial_values=lattice_values #needed fo the backwards pass TODO maybe the clone is not needed?


        class_logits=lattice_structure.slice_classify_with_precomputation(positions, delta_weights, linear_clasify_weight, linear_clasify_bias, nr_classes, splatting_indices, splatting_weights)


        ctx.save_for_backward(positions, initial_values, delta_weights, linear_clasify_weight, linear_clasify_bias, splatting_indices, splatting_weights )
        ctx.lattice_structure = lattice_structure
        ctx.val_dim=lattice_values.shape[1]
        ctx.nr_classes=nr_classes


        return class_logits

       
    @staticmethod
    def backward(ctx, grad_class_logits):
        
        positions, initial_values, delta_weights, linear_clasify_weight, linear_clasify_bias, splatting_indices, splatting_weights =ctx.saved_tensors
        lattice_py = ctx.lattice_structure
        val_dim=ctx.val_dim
        nr_classes=ctx.nr_classes

        # lattice_py.set_val_dim(val_dim)


        #create some tensors to host the gradient wrt to lattice_values, delta_weights, linear_weight and linear_bias
        grad_lattice_values=torch.zeros_like( lattice_py.values() )
        grad_delta_weights=torch.zeros_like( delta_weights )
        grad_linear_clasify_weight=torch.zeros_like(linear_clasify_weight)
        grad_linear_clasify_bias=torch.zeros_like(linear_clasify_bias)


        lattice_py.slice_classify_backwards_with_precomputation(grad_class_logits, positions, initial_values, delta_weights, linear_clasify_weight, linear_clasify_bias, nr_classes,
                                                                                    grad_lattice_values, grad_delta_weights, grad_linear_clasify_weight, grad_linear_clasify_bias,
                                                                                    splatting_indices, splatting_weights) 

       
        ctx.lattice_structure=0 # release the pointer to this so it gets cleaned up


        return grad_lattice_values, None, None, grad_delta_weights, grad_linear_clasify_weight, grad_linear_clasify_bias, None, None, None

class GatherLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values, lattice_structure, positions, splatting_indices, splatting_weights):

        lattice_structure.set_values(lattice_values)
        # lattice_structure.set_val_dim(lattice_values.shape[1])


        gathered_values=lattice_structure.gather_standalone_with_precomputation(positions, splatting_indices, splatting_weights)

    
        ctx.save_for_backward(positions, splatting_indices, splatting_weights)
        ctx.lattice_structure = lattice_structure
        ctx.val_dim=lattice_values.shape[1]


        return gathered_values

       
    @staticmethod
    def backward(ctx, grad_sliced_values):
        
        positions,splatting_indices, splatting_weights =ctx.saved_tensors
        lattice_py = ctx.lattice_structure
        val_dim=ctx.val_dim


        # lattice_py.set_val_dim(val_dim)
        lattice_py.gather_backwards_standalone_with_precomputation(positions, grad_sliced_values, splatting_indices, splatting_weights) 
        lattice_values=lattice_py.values() #we get a pointer to the values so they don't dissapear when we realease the lettice
      

        ctx.lattice_structure=0 # release the pointer to this so it gets cleaned up

        return lattice_values, None, None, None, None, None



# class PCA(Function):
#     @staticmethod
#     def forward(ctx, sv, with_debug_output, with_error_checking): #sv corresponds to the slices values, it has dimensions N x nr_positions x val_full_dim

#         # http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch/


#         X=sv.detach().squeeze(0).cpu()#we switch to cpu because svd for gpu needs magma: No CUDA implementation of 'gesdd'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/) at /opt/pytorch/aten/src/THC/generic/THCTensorMathMagma.cu:191
#         k=3
#         print("x is ", X.shape)
#         X_mean = torch.mean(X,0)
#         print("x_mean is ", X_mean.shape)
#         X = X - X_mean.expand_as(X)

#         U,S,V = torch.svd(torch.t(X)) 
#         C = torch.mm(X,U[:,:k])
#         print("C has shape ", C.shape)
#         print("C min and max is ", C.min(), " ", C.max() )
#         C-=C.min()
#         C/=C.max()
#         print("after normalization C min and max is ", C.min(), " ", C.max() )

#         return C


        
       
