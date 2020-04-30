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
from latticenet_py.lattice.lattice_py import LatticePy
# from inplace_abn import InPlaceABN

#Just to have something close to the macros we have in c++
def profiler_start(name):
    # print("profiler start form python ", name, " is profiling gpu is ", Profiler.is_profiling_gpu() )
    torch.cuda.synchronize()
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.start(name)
def profiler_end(name):
    torch.cuda.synchronize()
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.end(name)
TIME_START = lambda name: profiler_start(name)
TIME_END = lambda name: profiler_end(name)


class CreateVerts(Function):
    @staticmethod
    def forward(ctx, lattice_py, positions, with_debug_output, with_error_checking):

        lattice_py.just_create_verts(positions)

        return lattice_py
      

    @staticmethod
    def backward(ctx, grad_lattice_structure):
        return None
        

class SplatLattice(Function):
    @staticmethod
    def forward(ctx, lattice_py, positions, values, with_homogeneous_coord):
        # lattice=Lattice.create(config_file)
        # lattice=LatticePy()
        # lattice.create(config_file, "splated_lattice")
        lattice_py.begin_splat()
        lattice_py.splat_standalone(positions, values, with_homogeneous_coord)

        # lattice=Inher()
        # return *(lattice.lattice.to_tensors() )

        return lattice_py.values(), lattice_py

        # dummy_tensor = torch.zeros(1)
        # return dummy_tensor, lattice
        # ctx.pyscalar = pyscalar
        # ctx.save_for_backward(tensor1, tensor2)
        # return tensor1 + pyscalar * tensor2 + tensor1 * tensor2
        # return 

    @staticmethod
    def backward(ctx, grad_lattice_values, grad_lattice_structure):
        # print("backward")
        # raise NotImplementedError
        # return grad_output, grad_output
        
        return None, None, None

        pass
        # self.assertFalse(torch.is_grad_enabled())
        # t1, t2 = ctx.saved_tensors
        # return (grad_output + grad_output * t2, None,
                # grad_output * ctx.pyscalar + grad_output * t1)


#class CorsenLattice(Function):
#    @staticmethod
#    def forward(ctx, lattice_fine_values, lattice_fine_structure_py, nr_filters, neighbourhood_size, dilation, with_homogeneous_coord):
#        coarse_lattice_py=lattice_fine_structure_py.coarsen() #resturns a corsen lattice structure with new keys but with empty values (all values are zeros)
#
#
#        return coarse_lattice_py.values(), coarse_lattice_py
#
#    @staticmethod
#    def backward(ctx, grad_coarse_lattice_values, grad_coarse_lattice_structure):
#        
#        return None, None, None
#


class DistributeLattice(Function):
    @staticmethod
    # def forward(ctx, lattice_py, positions, values, dummy_weight):
    def forward(ctx, lattice_py, positions, values, experiment, with_debug_output, with_error_checking):
        # lattice_distributed=lattice_py.clone_lattice()
        # lattice_distributed.begin_splat()
        # lattice_distributed.distribute(positions, values)

        lattice_py.begin_splat()
        lattice_py.distribute(positions, values)
        lattice_py.set_values(lattice_py.distributed())

        #store the positions that created this lattice
        lattice_py.set_positions(positions)




        #subsctract mean from the positions so we have something like a local laplacian as a feature
        distributed=lattice_py.distributed() 

        experiments_that_imply_no_mean_substraction=["pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat"]
        indices=lattice_py.splatting_indices()
        distributed_positions=distributed[:,:3] #get the first 3 columns, the ones corresponding only to the xyz positions

        indices_long=indices.long()

        #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long<0]=0

        if experiment in experiments_that_imply_no_mean_substraction:
            # print("not performing mean substraction as the experiment is ", experiment)
            pass
        else:
            mean_positions = torch_scatter.scatter_mean(distributed_positions, indices_long, dim=0 )
            mean_positions[0,:]=0 #the first lattice vertex corresponds to the invalid points, the ones that had an index of -1. We set it to 0 so it doesnt affect the prediction or the batchnorm
            #by setting the first row of mean_positions to 0 it means that all the point that splat onto vertex zero will have a wrong mean. We will set those distributed_mean_substracted to also zero later
            #the distributed means now has shape nr_positions x pos_dim but we want to substract each distributed position (shape  (nr_positions x m_pos_dim+1) x pos_dim   ) with its corresponding mean. We can do a index_select with splatting indices to get the means
            distributed_mean_positions=torch.index_select(mean_positions, 0, indices_long)
            distributed[:,:3]=distributed_positions-distributed_mean_positions

        #we have to set the positions that ended up in an invalid vertes or the zero one because it's also considered invalid, to zero
        positions_that_splat_onto_vertex_zero_or_are_invalid=indices_long==0
        positions_that_splat_onto_vertex_zero_or_are_invalid=positions_that_splat_onto_vertex_zero_or_are_invalid.unsqueeze(1)
        # diRtributed.masked_fill_(positions_that_splat_onto_vertex_zero_or_are_invalid, 0)

        #DEBUG make distributed into only the positions
        # distributed=distributed[:,:3].clone()


        ##concat also the mean positions/ ADDING IT actually makes it worse because then the local features change a lot if the object is just translated a bit
        #distributed=torch.cat((distributed, distributed_mean_positions),1)

        #debug just add another column with zeros
        # zeros=torch.zeros(distributed.shape[0], 1).to("cuda")
        # distributed=torch.cat((distributed,zeros),1)

   
        distributed.masked_fill_(positions_that_splat_onto_vertex_zero_or_are_invalid, 0)

        # print("distributed is", distributed)

        return distributed, lattice_py.splatting_indices()


    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None, None


class BlurLattice(Function):
    @staticmethod
    def forward(ctx, lattice_py):
        blurred_lattice_py=lattice_py.blur_standalone()
        return blurred_lattice_py

    @staticmethod
    def backward(ctx, grad_output):
        pass


class ConvIm2RowLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values, lattice_py, filter_bank, dilation, with_homogeneous_coord, lattice_neighbours_values=None, lattice_neighbours_structure=None, use_center_vertex_from_lattice_neighbours=False, with_debug_output=True, with_error_checking=True):
        #in the case we have lattice_neighbours, this one will be the coarse one and the lattice neighbours will be the finer one

        if with_debug_output:
            print("lattice neigbhour values is ", lattice_neighbours_values)
        # print("lattice values is ", lattice_values)
        # print("lattice neighbour values required grad is ", lattice_neighbours_values.requires_grad)
        # print("lattice values required grad is ", lattice_values.requires_grad)

        lattice_py.set_values(lattice_values)
        if(lattice_neighbours_structure is not None):
            lattice_neighbours_structure.set_values(lattice_neighbours_values)

        # print("inside ConvIm2RowLattice input lattice values is ", lattice_values.shape)
        # TIME_START("convolution_itself")
        convolved_lattice_py=lattice_py.convolve_im2row_standalone(filter_bank, dilation, with_homogeneous_coord, lattice_neighbours_structure, use_center_vertex_from_lattice_neighbours)
        # TIME_END("convolution_itself")
        # print("conv forwards: convolved lattice has values ", convolved_lattice_py.values())
        if with_error_checking:
            nr_zeros=(convolved_lattice_py.values()==0).sum().item()
            print("conv forwards: convolved lattice has nr of values which are zero  ", nr_zeros  )
            if(nr_zeros>10):
                sys.exit("something went wrong. We have way to many zeros after doing the convolution")
            nr_zero_rows_rowified= ( lattice_py.lattice_rowified().sum(1)==0 ).sum().item()
            print("conv forwards: lattice rowified has nr of rows which are zero  ", nr_zero_rows_rowified  )
            if(nr_zero_rows_rowified>0):
                sys.exit("Why are they vertices that have no neigbhours")

        values=convolved_lattice_py.values()
        # print("inside ConvIm2RowLattice output lattice values is ", values.shape)
        # values+=bias
        convolved_lattice_py.set_values(values)



        # #debug why there are so many vertices that do not have any neighours
        # rowified=lattice_py.lattice_rowified().clone()
        # sum_rowified=rowified.sum(1)
        # print("sum_rowified is ", sum_rowified)
        # print("sum_rowified ha shape ", sum_rowified.shape)


        # if(lattice_neighbours_structure is not None):
            # print("lattice rowified for the coarse and fine graph is ", lattice_py.lattice_rowified() )
        #save stuff
        # ctx.save_for_backward(filter_bank, bias, lattice_py.lattice_rowified() ) 
        # print("saving for backwards convolved lattice which has nr filled", convolved_lattice_py.lattice.m_hash_table.m_nr_filled_tensor)
        # print("saving for backwards, filter_bank of shape", filter_bank.shape, "lattice rowified ", lattice_py.lattice_rowified().shape , "lattice_neighbours_values" )
        # ctx.save_for_backward(filter_bank, lattice_py.lattice_rowified(), lattice_neighbours_values ) 
        ctx.save_for_backward(filter_bank, lattice_values, lattice_neighbours_values ) 
        ctx.lattice_py=lattice_py
        # ctx.convolved_lattice_py=convolved_lattice_py #seems like its a bad idea to store an output because in the backwards pass this object might not be in the same state as we expected it to
        ctx.lattice_neighbours_structure=lattice_neighbours_structure
        ctx.use_center_vertex_from_lattice_neighbours=use_center_vertex_from_lattice_neighbours
        ctx.with_homogeneous_coord=with_homogeneous_coord
        ctx.filter_extent=int(filter_bank.shape[0]/lattice_values.shape[1])
        ctx.nr_filters= int(filter_bank.shape[1])#i hope it doesnt leak any memory
        ctx.dilation=dilation
        ctx.val_full_dim= lattice_py.lattice.val_full_dim()
        ctx.with_homogeneous_coord=with_homogeneous_coord
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking

        return values, convolved_lattice_py

    @staticmethod
    def backward(ctx, grad_lattice_values, grad_lattice_structure):
        
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking

        if with_debug_output:
            print("backwards conv")
        #the forward pass was done with lattice_in_rows*filter_bank, afterwards the homogeneous coordinate was just copied so the gradient for that will be 0
        #out=lattice_in_rows*filter_bank
        # this is WRONG https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product
        # this is good http://cs231n.stanford.edu/vecDerivs.pdf
        #dout_d_lattice_in_rows=filter_bank.tranposed()
        #dout_dfilterbank = dlatticerows.tranpose()

        #atemp2 derived by me
        #out=lattice_in_rows*filter_bank
        #dout_d_filter=lattice_in_rows

        # lattice_py=ctx.lattice.py 
        #grad_convolved_lattice has size hash_table_capacity x nr_filers+1
        #lattice_rowified has size  m_hash_table_capacity x filter_extent*(m_val_dim+1)
        #grad_filter_bank should have size filter_extent * (val_dim+1) x self.nr_filters 
        # so on possibility is to do im2row.tranpose.mm(grad_convolved_lattice.slice(1, 0, nr_filters) this will give me a matrix that would fit the filter bank
        # grad_filter_bank=grad_convolved_lattice.mm(im2col)


        lattice_py=ctx.lattice_py 
        # convolved_lattice_py=ctx.convolved_lattice_py 
        # print("got from backwards convolved lattice which has nr filled", convolved_lattice_py.lattice.m_hash_table.m_nr_filled_tensor)
        lattice_neighbours_structure=ctx.lattice_neighbours_structure
        use_center_vertex_from_lattice_neighbours=ctx.use_center_vertex_from_lattice_neighbours
        with_homogeneous_coord=ctx.with_homogeneous_coord
        filter_extent=ctx.filter_extent
        nr_filters=ctx.nr_filters
        dilation=ctx.dilation
        val_full_dim=ctx.val_full_dim

        # lattice_rowified=lattice_py.lattice_rowified()   
        filter_bank, lattice_values, lattice_neighbours_values =ctx.saved_tensors
        filter_extent=int(filter_bank.shape[0]/val_full_dim)

        #reconstruct lattice_rowified 
        lattice_py.set_values(lattice_values)
        if(lattice_neighbours_structure is not None):
            lattice_neighbours_structure.set_values(lattice_neighbours_values)
        lattice_rowified= lattice_py.im2row(filter_extent, lattice_neighbours_structure, dilation, use_center_vertex_from_lattice_neighbours, False)

        filter_bank_transposed=filter_bank.transpose(0,1) 
        if(ctx.with_homogeneous_coord):
            #grad with respect to the filte bank
            grad_filter=filter_bank_transposed.mm(grad_lattice_values[:, 0:nr_filters ]  ) #we ignore the last column of the grad_convolved lattice because that corresponds to the homogeneous coordinate and this layer just acts a pass through for it. Intrisically we can thing of the homogeneous coordinate as getting convolved with another filter that is zero everywhere except for the homogenous coordinate of the centre pixel where it has value of 1. This filter would never get optimized so it mearly acts a passthrough therefore we just ignore it

            #grad with respect to the input lattice
            #grad_input_lattice should have size m_hash_table_capacity x (m_val_dim+1)
            #filter_bank has size filter_extent*(m_val_dim+1) x nr_filters
            #they seems that they do filter_bank.mm(grad_convolved_lattice) and then a col2im
            #one posibiliy would be grad_convolved_lattice.slice(1, 0, nr_filters) .mm(filter_bank_tranpose_)
            grad_lattice_rowified=grad_lattice_values[: , 0: nr_filters].mm(filter_bank_transposed ) #this will have shape m_hash_table_capacity x (filter_extent*(m_val_dim+1))
            grad_lattice= grad_lattice_rowified[:, -val_full_dim: ] #rowim is just getting the last columns of the grad_lattice_rowified because those values correspond to the center lattice vertex
            # print("grad lattic has shape: ", grad_lattice.shape)
        else:
            grad_filter=lattice_rowified.transpose(0,1).mm(grad_lattice_values) 

            #grad with respect to the input lattice
            #grad_input_lattice should have size m_hash_table_capacity x (m_val_dim+1)
            #filter_bank has size filter_extent*(m_val_dim+1) x nr_filters
            #they seems that they do filter_bank.mm(grad_convolved_lattice) and then a col2im
            #one posibiliy would be grad_convolved_lattice.slice(1, 0, nr_filters) .mm(filter_bank_tranpose_)
            # filter_bank, bias =ctx.saved_tensors
            # filter_bank_transposed=filter_bank.transpose(0,1) 
            # grad_lattice_rowified=grad_lattice_values.mm(filter_bank_transposed ) #this will have shape m_hash_table_capacity x (filter_extent*(m_val_dim+1))
            # grad_lattice_prev= grad_lattice_rowified[:, -val_full_dim: ] #rowim is just getting the last columns of the grad_lattice_rowified because those values correspond to the center lattice vertex

            ##attempt2 for grad_lattice
            # filter_bank_slice=filter_bank[-val_full_dim:, :].transpose(0,1) #we get the filter weights what will affect the center vertex, these correspond to the last val_ful_dim rows of the fitler bank
            # grad_lattice=grad_lattice_values.mm(filter_bank_slice ) 


            #attempt 3 for grad_lattice 
            # N x nr_filters  *  nr_filters x val x filter_extent = N x val x filter_extent  And then do a row2im
            # print("lattice rowified is ", lattice_rowified)
            # torch.save(lattice_rowified, "lattice_rowified.pt")
            # grad_lattice_rowified=grad_lattice_values.mm(filter_bank_transposed) 
            # grad_lattice=lattice_py.row2im(grad_lattice_rowified, dilation, filter_extent, nr_filters, lattice_neighbours_structure, use_center_vertex)


            # #attempt 4 for grad_lattice
            filter_bank_backwards=filter_bank.transpose(0,1) # creates a nr_filters x filter_extent * val_fim  
            filter_bank_backwards=filter_bank_backwards.view(nr_filters,filter_extent,val_full_dim) # nr_filters x filter_extent x val_fim  
            filter_bank_backwards=filter_bank_backwards.transpose(0,1).contiguous()  #makes it filter_extent x nr_filters x val_fim   #TODO the contigous may noy be needed because the reshape does may do a contigous if needed or may also just return a view, both work
            filter_bank_backwards=filter_bank_backwards.reshape(filter_extent*nr_filters, val_full_dim)
            lattice_py.set_values(grad_lattice_values)
            grad_lattice_py=lattice_py.convolve_im2row_standalone(filter_bank_backwards, dilation, with_homogeneous_coord, lattice_neighbours_structure, use_center_vertex_from_lattice_neighbours, True)
            grad_lattice=grad_lattice_py.values()

            # #how to tranpose the filter 
            # filter_bank_test=torch.arange(1,29)
            # filter_bank_test=filter_bank_test.view(14,2) #filter_extent * val_fim  x nr_filters
            # print("filter_bank_test is ", filter_bank_test)
            # filter_bank_test=filter_bank_test.transpose(0,1)  #nr_filters x filter_extent * val_fim  
            # filter_bank_test=filter_bank_test.view(2,7,2) #nr_filters x filter_extent x val_fim  
            # print("filter_bank_test is ", filter_bank_test)
            # # filter_bank_test=filter_bank_test.transpose(0,1).contiguous()  #makes it filter_extent x nr_filters x val_fim  
            # filter_bank_test=filter_bank_test.transpose(0,1)  #makes it filter_extent x nr_filters x val_fim  
            # filter_bank_test=filter_bank_test.reshape(14,2)
            # print("filter_bank_test is ", filter_bank_test)

            # diff=grad_lattice-grad_lattice_2
            # diff_val=diff.norm()
            # print("diff is ", diff_val)
            

            # print("grad_lattice_rowified has shape ", grad_lattice_rowified.shape)
            # print("grad_lattice_prev has shape ", grad_lattice_prev.shape)
            # diff=grad_lattice-grad_lattice_prev
            # diff=torch.abs(diff)
            # print("diff is " , diff.sum())


        #grad wrt to bias
        # grad_bias = grad_lattice_values.sum(0)

        #debug 
        if with_debug_output:
            print("grad_lattice has norm: ", grad_lattice.norm())
            # print("grad_lattice_2 has norm: ", grad_lattice_2.norm())
            print("grad_filter has norm: ", grad_filter.norm())
            # print("grad_bias has norm: ", grad_bias.norm())


        ctx.lattice_py=0 #release this object so it doesnt leak
        ctx.lattice_neighbours_structure=0
        # print("WE ARE NOT RELEASING THE LATTICE BECAUSE WE ARE RUNNING GRAD_CHECK")

        # return None, grad_filter
        if(lattice_neighbours_values is None):
            return grad_lattice, None, grad_filter,  None, None, None, None, None, None, None
        else:
            return grad_lattice, None, grad_filter,  None, None, None, None, None, None, None
            # return grad_lattice, None, grad_filter, grad_bias, None, None, grad_lattice, None

        # dummy = torch.zeros(4000,64)
        # dummy=dummy.to("cuda")
        # return grad_lattice, None, grad_filter, grad_bias, None, None, dummy, None

class CoarsenLattice(Function):
    @staticmethod
    def forward(ctx, lattice_fine_values, lattice_fine_structure, filter_bank, use_center_vertex_from_lattice_neighbours, with_debug_output, with_error_checking):
        if with_debug_output:
            print("coarsening forward")
            print("coarseing forward got lattice fine values of norm", lattice_fine_values.norm())
        lattice_fine_structure.set_values(lattice_fine_values)
        if with_debug_output:
            print("setting values to the fine structure, the values have norm", lattice_fine_structure.values().norm() )

        #create a structure for the coarse lattice, the values of the coarse vertices will be zero
        positions=lattice_fine_structure.positions()
        if with_debug_output:
            print("coarsening in a naive, way getting the positions that created the previous lattice which are of shape", positions.shape)

        # print("fine lattice has keys", lattice_fine_structure.keys()) 
        #coarsened_lattice_py=lattice_fine_structure.create_coarse_verts()
        # print("lattice fine structure has indices", lattice_fine_structure.splatting_indices())
        coarsened_lattice_py=lattice_fine_structure.create_coarse_verts_naive(positions)
        # print("coarsened_lattice_py has indices", coarsened_lattice_py.splatting_indices())
        # print("coarsened has keys", coarsened_lattice_py.keys()) 


        if with_debug_output:
            print("after creating the coarse_verts the fine structure has values with norm", lattice_fine_structure.values().norm() )

        #convolve at this lattice vertices with the neighbours from lattice_fine
        # TIME_START("convolution_itself")
        dilation=1
        with_homogeneous_coord=False
        convolved_lattice_py=coarsened_lattice_py.convolve_im2row_standalone(filter_bank, dilation, with_homogeneous_coord, lattice_fine_structure, use_center_vertex_from_lattice_neighbours)
        # TIME_END("convolution_itself")
        # print("coarsening forwards: convolved lattice has values ", convolved_lattice_py.values())
        if with_error_checking:
            nr_zeros=(convolved_lattice_py.values()==0).sum().item()
            vals_debug=convolved_lattice_py.values().clone()
            vals_summed_debug=vals_debug.sum(1)
            nr_zeros_rows=(vals_summed_debug==0).sum().item()
            print("coarsening forwards: convolved lattice has nr of values which are zero  ", nr_zeros  )
            print("coarsening forwards: convolved lattice has nr of rows which are zero  ", nr_zeros_rows  )
            nr_zero_rows_rowified= ( coarsened_lattice_py.lattice_rowified().sum(1)==0 ).sum().item()
            # print("summing the lattice rowified along the rows is ", coarsened_lattice_py.lattice_rowified().sum(1)  )
            print("coarsening forwards: lattice rowified has nr of rows which are zero  ", nr_zero_rows_rowified  )
            if(nr_zero_rows_rowified>100):
                sys.exit("Why are they so many vertices that have no neigbhours")


        # if(nr_zeros>10):
            # sys.exit("something went wrong. We have way to many zeros after doing the convolution")
        values=convolved_lattice_py.values()
        # values+=bias
        convolved_lattice_py.set_values(values)


        #for debug
        # print("coarsening made a lattice with lattice_rowified", coarsened_lattice_py.lattice_rowified())

        if with_debug_output:
            print("saving coarsened_lattice_rowified of norm ", coarsened_lattice_py.lattice_rowified().norm() )
            print("saving coarsened_lattice_rowified of shape ", coarsened_lattice_py.lattice_rowified().shape )
        # ctx.save_for_backward(filter_bank, coarsened_lattice_py.lattice_rowified() ) 
        ctx.save_for_backward(filter_bank, lattice_fine_values ) 
        ctx.coarsened_lattice_py=coarsened_lattice_py
        ctx.lattice_fine_structure=lattice_fine_structure
        ctx.use_center_vertex_from_lattice_neighbours=use_center_vertex_from_lattice_neighbours
        ctx.with_homogeneous_coord=with_homogeneous_coord
        ctx.filter_extent=int(filter_bank.shape[0]/lattice_fine_values.shape[1])
        ctx.nr_filters= int(filter_bank.shape[1])#i hope it doesnt leak any memory
        ctx.dilation=dilation
        ctx.val_full_dim= lattice_fine_structure.lattice.val_full_dim()
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking


       
        # ctx.save_for_backward(filter_bank, bias, lattice_py.lattice_rowified().clone(), lattice_fine_values ) 
        # ctx.lattice_py=lattice_py
        # ctx.nr_filters= filter_bank.shape[1]#i hope it doesnt leak any memory
        # ctx.val_full_dim= lattice_py.lattice.val_full_dim()
        # ctx.with_homogeneous_coord=with_homogeneous_coord

        return values, convolved_lattice_py

    @staticmethod
    def backward(ctx, grad_lattice_values, grad_lattice_structure):
        
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking

        if with_debug_output:
            print("coarsening backwards")
            print("coarsening backwards input grad lattice values has size ", grad_lattice_values.shape)

        coarsened_lattice_py=ctx.coarsened_lattice_py
        lattice_fine_structure=ctx.lattice_fine_structure
        use_center_vertex_from_lattice_neighbours=ctx.use_center_vertex_from_lattice_neighbours
        with_homogeneous_coord=ctx.with_homogeneous_coord
        filter_extent=ctx.filter_extent
        nr_filters=ctx.nr_filters
        dilation=ctx.dilation
        val_full_dim=ctx.val_full_dim
  
        filter_bank, lattice_fine_values =ctx.saved_tensors
        filter_extent=int(filter_bank.shape[0]/val_full_dim)

        #reconstruct lattice_rowified 
        lattice_fine_structure.set_values(lattice_fine_values)
        lattice_rowified= coarsened_lattice_py.im2row(filter_extent, lattice_fine_structure, dilation, use_center_vertex_from_lattice_neighbours, False)

        if with_debug_output:
            print("got from saved_tensors coarsened_lattice_rowified of norm ", lattice_rowified.norm() )

        # return grad_lattice_values, grad_lattice_structure
        # return None, None, None, None, None
        grad_filter=lattice_rowified.transpose(0,1).mm(grad_lattice_values) 

        filter_bank_backwards=filter_bank.transpose(0,1) # creates a nr_filters x filter_extent * val_fim  
        filter_bank_backwards=filter_bank_backwards.view(nr_filters,filter_extent,val_full_dim) # nr_filters x filter_extent x val_fim  
        filter_bank_backwards=filter_bank_backwards.transpose(0,1).contiguous()  #makes it filter_extent x nr_filters x val_fim   #TODO the contigous may noy be needed because the reshape does may do a contigous if needed or may also just return a view, both work
        filter_bank_backwards=filter_bank_backwards.reshape(filter_extent*nr_filters, val_full_dim)
        if with_debug_output:
            print("coarsened backwards: saved for backwards a coarsened lattice py with nr of keys", coarsened_lattice_py.nr_lattice_vertices())
        coarsened_lattice_py.set_values(grad_lattice_values)
        if with_debug_output:
            print("before setting the val_full_dim, fine structure has val_full_dim set to ", lattice_fine_structure.val_full_dim())
        lattice_fine_structure.set_val_full_dim(nr_filters) #setting val full dim to nr of filters because we will convolve the values of grad_lattice values and those have a row of size nr_filters
        if with_debug_output:
            print("after setting the val_full_dim, fine structure has val_full_dim set to ", lattice_fine_structure.val_full_dim())
        #one hast o convolve at the fine positions, having the neighbour as the coarse ones because they are the ones with the errors
        grad_lattice_py=lattice_fine_structure.convolve_im2row_standalone(filter_bank_backwards, dilation, with_homogeneous_coord, coarsened_lattice_py, use_center_vertex_from_lattice_neighbours, True)
        grad_lattice=grad_lattice_py.values()

        # grad_bias = grad_lattice_values.sum(0)


        #debug why the gradients are so shitty
        # fine_rowified=lattice_fine_structure.lattice_rowified().clone()
        # fine_sum_rowified=fine_rowified.sum(1)
        # print("fine_sum_rowified is ", fine_sum_rowified)
        # print("fine_sum_rowified ha shape ", fine_sum_rowified)


        #release
        ctx.coarsened_lattice_py=0
        ctx.lattice_fine_structure=0
        # print("WE are not releasing the coarse lattice")

        #debug 
        if with_debug_output:
            print("grad_lattice has norm: ", grad_lattice.norm())
            print("grad_filter has norm: ", grad_filter.norm())
            # print("grad_bias has norm: ", grad_bias.norm())
            print("grad_lattice has shape: ", grad_lattice.shape)
            print("grad_filter has shape: ", grad_filter.shape)
            # print("grad_bias has shape: ", grad_bias.shape )

        # sys.exit("exit after the coarsening backwards")

        return grad_lattice, None, grad_filter, None, None, None #THe good one
        # return None, None, grad_filter, grad_bias, None #this sems somewhat fine, the loss is still jiggly but not that much as with just passing grad_lattice and not affecting the filter and the bias
        # return grad_lattice, None, None, None, None #still make sthe train lost jiggly

class CoarsenAndReturnLatticeRowified(Function):
    @staticmethod
    def forward(ctx, lattice_fine_values, lattice_fine_structure, use_center_vertex_from_lattice_neighbours, with_debug_output, with_error_checking):
        if with_debug_output:
            print("coarsening and returning lattice rowified forward")
            print("coarseing and returning lattice rowified forward got lattice fine values of norm", lattice_fine_values.norm())
        lattice_fine_structure.set_values(lattice_fine_values)
        if with_debug_output:
            print("setting values to the fine structure, the values have norm", lattice_fine_structure.values().norm() )

        #create a structure for the coarse lattice, the values of the coarse vertices will be zero
        positions=lattice_fine_structure.positions()
        if with_debug_output:
            print("coarsening in a naive, way getting the positions that created the previous lattice which are of shape", positions.shape)

        # print("fine lattice has keys", lattice_fine_structure.keys()) 
        # coarsened_lattice_py=lattice_fine_structure.create_coarse_verts()
        coarsened_lattice_py=lattice_fine_structure.create_coarse_verts_naive(positions)

        if with_debug_output:
            print("after creating the coarse_verts the fine structure has values with norm", lattice_fine_structure.values().norm() )

        print("COARSEN AND RETURN LATTICE ROWIFIED, lattice fine values has val full dim of shape ", lattice_fine_values.shape[1]) 


        val_full_dim=lattice_fine_values.shape[1]
        filter_extent=coarsened_lattice_py.lattice.get_filter_extent(1)
        nr_vertices=coarsened_lattice_py.nr_lattice_vertices()
        coarse_lattice_rowified=coarsened_lattice_py.im2row(filter_extent=filter_extent, lattice_neighbours=lattice_fine_structure,  dilation=1, use_center_vertex_from_lattice_neighbours=True, flip_neighbours=False)

        #debug, check with a test of row2im that all the values are in the positions we expect them to be in
        # grad_lattice=lattice_fine_structure.lattice.row2im(coarse_lattice_rowified, 1, filter_extent, val_full_dim, coarsened_lattice_py.lattice, use_center_vertex_from_lattice_neighbours, True)
        
      
        ctx.coarsened_lattice_py=coarsened_lattice_py
        ctx.lattice_fine_structure=lattice_fine_structure
        ctx.use_center_vertex_from_lattice_neighbours=use_center_vertex_from_lattice_neighbours
        ctx.val_full_dim= lattice_fine_structure.lattice.val_full_dim()
        ctx.filter_extent=filter_extent
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking
       

        return coarse_lattice_rowified, coarsened_lattice_py

    @staticmethod
    def backward(ctx, grad_lattice_rowified, grad_lattice_structure):
        
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking

        if with_debug_output:
            print("CoarsenAndReturnLatticeRowified backwards")
            print("CoarsenAndReturnLatticeRowified backwards input grad lattice values has size ", grad_lattice_rowified.shape)

        coarsened_lattice_py=ctx.coarsened_lattice_py
        lattice_fine_structure=ctx.lattice_fine_structure
        use_center_vertex_from_lattice_neighbours=ctx.use_center_vertex_from_lattice_neighbours
        val_full_dim=ctx.val_full_dim
        filter_extent=ctx.filter_extent
  




        values=lattice_fine_structure.values()
        if (values.shape[1] is not val_full_dim):
            # print("COARSEN AND RETURN LATTICE ROWIFIED BACKWARD, lattice fine values has val full dim of shape ", values.shape[1]) 
            # sys.exit("During row2im we are accumulating the gradient into the values of the current fine lattice. It should have size nr_vertices x val_full_dim, wher val full dim. But apartenyl it's not")
            values=torch.zeros( (values.shape[0], val_full_dim), device="cuda" )
        else:
            values=values.fill_(0)
        lattice_fine_structure.set_values(values)

        grad_lattice=lattice_fine_structure.lattice.row2im(grad_lattice_rowified, 1, filter_extent, val_full_dim, coarsened_lattice_py.lattice, use_center_vertex_from_lattice_neighbours, False)

 
        #release
        ctx.coarsened_lattice_py=0
        ctx.lattice_fine_structure=0

        #debug 
        if with_debug_output:
            print("grad_lattice has norm: ", grad_lattice.norm())
            print("grad_lattice has shape: ", grad_lattice.shape)

        # sys.exit("exit after the coarsening backwards")

        return grad_lattice, None, None, None, None #THe good one

class FinefyLattice(Function):
    @staticmethod
    def forward(ctx, lattice_coarse_values, lattice_coarse_structure, lattice_fine_structure,  filter_bank, use_center_vertex_from_lattice_neighbours, with_debug_output, with_error_checking):
        if with_debug_output:
            print("finefy forward")
        lattice_coarse_structure.set_values(lattice_coarse_values)
        lattice_fine_structure.set_val_full_dim(lattice_coarse_structure.val_full_dim())


        #convolve at this lattice vertices with the neighbours from lattice_fine
        # TIME_START("convolution_itself")
        dilation=1
        with_homogeneous_coord=False
        convolved_lattice_py=lattice_fine_structure.convolve_im2row_standalone(filter_bank, dilation, with_homogeneous_coord, lattice_coarse_structure, use_center_vertex_from_lattice_neighbours)
        # TIME_END("convolution_itself")
        # print("coarsening forwards: convolved lattice has values ", convolved_lattice_py.values())
        # nr_zeros=(convolved_lattice_py.values()==0).sum().item()
        # vals_debug=convolved_lattice_py.values().clone()
        # vals_summed_debug=vals_debug.sum(1)
        # nr_zeros_rows=(vals_summed_debug==0).sum().item()
        # print("coarsening forwards: convolved lattice has nr of values which are zero  ", nr_zeros  )
        # print("coarsening forwards: convolved lattice has nr of rows which are zero  ", nr_zeros_rows  )
        # nr_zero_rows_rowified= ( coarsened_lattice_py.lattice_rowified().sum(1)==0 ).sum().item()
        # print("summing the lattice rowified along the rows is ", coarsened_lattice_py.lattice_rowified().sum(1)  )
        # print("coarsening forwards: lattice rowified has nr of rows which are zero  ", nr_zero_rows_rowified  )
        # if(nr_zero_rows_rowified>0):
            # sys.exit("Why are they vertices that have no neigbhours")


        # if(nr_zeros>10):
            # sys.exit("something went wrong. We have way to many zeros after doing the convolution")
        values=convolved_lattice_py.values()
        # values+=bias
        convolved_lattice_py.set_values(values)

        # print("saving coarsened_lattice_rowified of norm ", coarsened_lattice_py.lattice_rowified().norm() )
        # print("saving coarsened_lattice_rowified of shape ", coarsened_lattice_py.lattice_rowified().shape )
        # ctx.save_for_backward(filter_bank, lattice_fine_structure.lattice_rowified() ) 
        ctx.save_for_backward(filter_bank, lattice_coarse_values) 
        # ctx.coarsened_lattice_py=coarsened_lattice_py
        ctx.lattice_fine_structure=lattice_fine_structure
        ctx.lattice_coarse_structure=lattice_coarse_structure
        ctx.use_center_vertex_from_lattice_neighbours=use_center_vertex_from_lattice_neighbours
        ctx.with_homogeneous_coord=with_homogeneous_coord
        ctx.filter_extent=int(filter_bank.shape[0]/lattice_coarse_values.shape[1])
        ctx.nr_filters= int(filter_bank.shape[1])#i hope it doesnt leak any memory
        ctx.dilation=dilation
        ctx.val_full_dim= lattice_coarse_structure.lattice.val_full_dim()
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking

       
        # ctx.save_for_backward(filter_bank, bias, lattice_py.lattice_rowified().clone(), lattice_fine_values ) 
        # ctx.lattice_py=lattice_py
        # ctx.nr_filters= filter_bank.shape[1]#i hope it doesnt leak any memory
        # ctx.val_full_dim= lattice_py.lattice.val_full_dim()
        # ctx.with_homogeneous_coord=with_homogeneous_coord

        return values, convolved_lattice_py

    @staticmethod
    def backward(ctx, grad_lattice_values, grad_lattice_structure):
        # print("finefy backwards")
        # print("finefy backwards input grad lattice values has size ", grad_lattice_values.shape)

        # coarsened_lattice_py=ctx.coarsened_lattice_py
        lattice_fine_structure=ctx.lattice_fine_structure
        lattice_coarse_structure=ctx.lattice_coarse_structure
        use_center_vertex_from_lattice_neighbours=ctx.use_center_vertex_from_lattice_neighbours
        with_homogeneous_coord=ctx.with_homogeneous_coord
        filter_extent=ctx.filter_extent
        nr_filters=ctx.nr_filters
        dilation=ctx.dilation
        val_full_dim=ctx.val_full_dim
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking
        filter_bank, lattice_coarse_values =ctx.saved_tensors
        filter_extent=int(filter_bank.shape[0]/val_full_dim)

        #reconstruct lattice_rowified 
        lattice_coarse_structure.set_values(lattice_coarse_values)
        lattice_rowified= lattice_fine_structure.im2row(filter_extent, lattice_coarse_structure, dilation, use_center_vertex_from_lattice_neighbours, False)

        if with_debug_output:
            print("got from saved_tensors coarsened_lattice_rowified of norm ", lattice_rowified.norm() )

        # return grad_lattice_values, grad_lattice_structure
        # return None, None, None, None, None
        grad_filter=lattice_rowified.transpose(0,1).mm(grad_lattice_values) 

        filter_bank_backwards=filter_bank.transpose(0,1) # creates a nr_filters x filter_extent * val_fim  
        filter_bank_backwards=filter_bank_backwards.view(nr_filters,filter_extent,val_full_dim) # nr_filters x filter_extent x val_fim  
        filter_bank_backwards=filter_bank_backwards.transpose(0,1).contiguous()  #makes it filter_extent x nr_filters x val_fim   #TODO the contigous may noy be needed because the reshape does may do a contigous if needed or may also just return a view, both work
        filter_bank_backwards=filter_bank_backwards.reshape(filter_extent*nr_filters, val_full_dim)
        # print("finefy backwards: saved for backwards a coarsened lattice py with nr of keys", coarsened_lattice_py.nr_lattice_vertices())
        lattice_fine_structure.set_values(grad_lattice_values)
        if with_debug_output:
            print("before setting the val_full_dim, fine structure has val_full_dim set to ", lattice_fine_structure.val_full_dim())
        lattice_coarse_structure.set_val_full_dim(lattice_fine_structure.val_full_dim()) #setting val full dim to nr of filters because we will convolve the values of grad_lattice values and those have a row of size nr_filters
        if with_debug_output:
            print("after setting the val_full_dim, fine structure has val_full_dim set to ", lattice_fine_structure.val_full_dim())
        #one hast o convolve at the fine positions, having the neighbour as the coarse ones because they are the ones with the errors
        grad_lattice_py=lattice_coarse_structure.convolve_im2row_standalone(filter_bank_backwards, dilation, with_homogeneous_coord, lattice_fine_structure, use_center_vertex_from_lattice_neighbours, True)
        grad_lattice=grad_lattice_py.values()

        # grad_bias = grad_lattice_values.sum(0)


        #debug why the gradients are so shitty
        # fine_rowified=lattice_fine_structure.lattice_rowified().clone()
        # fine_sum_rowified=fine_rowified.sum(1)
        # print("fine_sum_rowified is ", fine_sum_rowified)
        # print("fine_sum_rowified ha shape ", fine_sum_rowified)


        #release
        ctx.lattice_coarse_structure=0
        ctx.lattice_fine_structure=0
        # print("FINEFY BACKWARD WE ARE NOT RELEASING THE LATTICE BECAUSE WE ARE RUNNING GRAD_CHECK")

        #debug 
        if with_debug_output:
            print("grad_lattice has norm: ", grad_lattice.norm())
            print("grad_filter has norm: ", grad_filter.norm())
            # print("grad_bias has norm: ", grad_bias.norm())
            print("grad_lattice has shape: ", grad_lattice.shape)
            print("grad_filter has shape: ", grad_filter.shape)
            # print("grad_bias has shape: ", grad_bias.shape )


        return grad_lattice, None, None, grad_filter, None, None, None #THe good one
        # return None, None, grad_filter, grad_bias, None #this sems somewhat fine, the loss is still jiggly but not that much as with just passing grad_lattice and not affecting the filter and the bias
        # return grad_lattice, None, None, None, None #still make sthe train lost jiggly



# class CoarsenMaxLattice(Function):
#     @staticmethod
#     def forward(ctx, lattice_fine_values, lattice_fine_structure):
#         lattice_fine_structure.set_values(lattice_fine_values)

#         #create a structure for the coarse lattice, the values of the coarse vertices will be zero
#         positions=lattice_fine_structure.positions()
#         print("coarsening in a naive, way getting the positions that created the previous lattice which are of shape", positions.shape)
#         coarsened_lattice_py=lattice_fine_structure.create_coarse_verts()
#         # coarsened_lattice_py=lattice_fine_structure.create_coarse_verts_naive(positions)
#         print("after creating the coarse_verts the fine structure has values with norm", lattice_fine_structure.values().norm() )


#         #coarsened lattice_im2row(use_center_vertex_from_lattineighbous)
#         #view the lattice rowified as a 3D tensor
#             #m_lattice_rowified=torch::zeros({nr_vertices, filter_extent*m_val_full_dim });
#             #lattice_rowified=lattice_rowified.view(nr_vertices, filter_extent, val_full_dim)
#         #transpose so that the values at the same index are adyacent in memory
#             # lattice_rowified=lattice_rowified.tranpose(1,2) #this will make it have shape nr_vertices x val_full_dim x filter_extent
#         #argmax over the values
#             # lattice_max, argmax=lattice_rowified.max(2) 
#             #argmax will have shape of nr_vertices x filter_extent x val_full_dim
#         #the max will give us the values of the coarsened lattice and the argmax sayd for each value we got the index of the neigbhour that create it
#             # values=lattice_max #shape is nr_vertices x val_full_dim


#         #backwards pass
#         #we get a gradient g of nr_verts x val_full_dim
#         #This need to backpropagate for each vertex into the ones that created the max value
        
#         use_center_vertex_from_lattice_neighbours=True
#         filter_extent=coarsened_lattice_py.lattice.get_filter_extent(1) 
#         lattice_rowified=coarsened_lattice_py.im2row(filter_extent=filter_extent, lattice_neighbours=lattice_fine_structure)
#         # print("lattice_rowified is ", lattice_rowified)
#         print("lattice rowified has shape", lattice_rowified.shape)
#         nr_zero_rows_rowified= ( lattice_rowified.sum(1)==0 ).sum().item()
#         print("coarsening_max forwards: lattice rowified has nr of rows which are zero  ", nr_zero_rows_rowified  )

#         nr_vertices=lattice_rowified.shape[0]
#         val_full_dim=lattice_fine_structure.val_full_dim()
#         lattice_rowified=lattice_rowified.view(nr_vertices, filter_extent, val_full_dim)
#         print("lattice rowified has shape", lattice_rowified.shape)
#         max_values, argmax=lattice_rowified.max(1)
#         print("max_values has shape", max_values.shape)
#         print("argmax has shape", argmax.shape) #nr_vertices x m_val_full_dim . Contains on each row the index of the neigbhour that maxed into that value
#         # print("argmax is",argmax)
#         # print("max_values is",max_values)

#         coarsened_lattice_py.set_values(max_values)
        
#         ctx.save_for_backward(lattice_rowified.clone(), argmax) 
#         ctx.coarsened_lattice_py=coarsened_lattice_py 
#         ctx.lattice_fine_structure=lattice_fine_structure 
#         ctx.filter_extent=filter_extent
#         ctx.val_full_dim=val_full_dim
        

#         return max_values, coarsened_lattice_py

#     @staticmethod
#     def backward(ctx, grad_lattice_values, grad_lattice_structure):
        
#         # grad_lattice_values_rowified.reshape like(lattice_rowified)
#         # grad_lattice_values_rowified=grad_lattice_values_rowified.tranpose(1,2) #now it has shape nr_vertices x filter_extent x val_full_dim just like the argmax
#         # #multiply elementwise with the argmax effectivelly setting to 
#         # grad_lattice_values_rowified[argmax] = grad_lattice_values
#         # values=row2im(grad_lattice_values_rowified)


#         coarsened_lattice_py=ctx.coarsened_lattice_py
#         lattice_fine_structure=ctx.lattice_fine_structure
#         lattice_rowified, argmax =ctx.saved_tensors
#         filter_extent=ctx.filter_extent
#         val_full_dim=ctx.val_full_dim

#         grad_lattice_values_rowified=torch.zeros_like(lattice_rowified)
#         print("grad_lattice_values_rowified has shape " , grad_lattice_values_rowified.shape)
#         # grad_lattice_values_rowified=grad_lattice_values_rowified.transpose(1,2)
#         # mask = torch.arange(lattice_rowified.size(1)).reshape(1, 1, -1) == argmax.unsqueeze(2)
#         # https://stackoverflow.com/questions/54057112/indexing-the-max-elements-in-a-multidimensional-tensor-in-pytorch
#         possible_indices = torch.arange(lattice_rowified.size(1)).reshape(1, -1, 1) 
#         print("possible_indices size is ", possible_indices.shape)
#         argmax_unsqueezed=argmax.unsqueeze(1)
#         print("argmax_unsqueezed size is ", argmax_unsqueezed.shape)
#         mask = possible_indices.cuda() == argmax_unsqueezed.cuda()
#         print("mask size is ", mask.shape)
#         grad_lattice_values_rowified[mask] = grad_lattice_values.flatten()
#         print("grad_lattice_values_rowified norm is ", grad_lattice_values_rowified.norm())
#         print("grad_lattice_values_rowified", grad_lattice_values_rowified)


#         # grad_lattice=lattice_fine_structue.row2im(grad_lattice_values_rowified, 1, filter_extent, val_full_dim, lattice_neighbours_structure, use_center_vertex, I dont think it matters if we flip)

#         ctx.coarsened_lattice_py=0
#         ctx.lattice_fine_structure=0


#         return None, None
#         # return grad_lattice, None#THe good one
#         # return None, None, grad_filter, grad_bias, None #this sems somewhat fine, the loss is still jiggly but not that much as with just passing grad_lattice and not affecting the filter and the bias
#         # return grad_lattice, None, None, None, None #still make sthe train lost jiggly

         
            




#following this veyr good explanation https://deepnotes.io/maxpool
# class MaxPoolLattice(Function):
#     @staticmethod
#     def forward(ctx, lattice_py):

#         lattice_rowified =lattice_py.lattice_rowified() # size of hash_table_capacity x (filter_extent*(val_dim+1) ) where the val_dim is the fastest changing dimensions, then comes filter_extent and finally hash_table_capacity which is the slowest
#         lattice_rowified.reshape()

#         return lattice_py

#     @staticmethod
#     def backward(ctx, grad_downsampled_lattice):
        
#         return None

#multiscale should be added through dilated convolution because they are better for semantic segmentation and easier to implement in a lattice
#https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/




class SliceLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values, lattice_structure, positions, with_homogeneous_coord, with_debug_output, with_error_checking):

        # sliced_values=lattice_py.slice_standalone_no_precomputation(positions, with_homogeneous_coord)
        # # import ipdb; ipdb.set_trace()

        # ctx.save_for_backward(positions, lattice_py.sliced_values_hom() )
        # ctx.lattice_py = lattice_py 
        # ctx.with_homogeneous_coord=with_homogeneous_coord

        # #sliced_values has no grad_fn
        # # sliced_values.grad_fn=self.backward

        # return sliced_values
        if with_debug_output:
            print("slice forward received a lattice structure with values ", lattice_structure.values())
            print("slice forward received a lattice structure with nr of vertices", lattice_structure.nr_lattice_vertices())
            print("slice forward received a lattice structure with values of shape", lattice_structure.values().shape)


        #attempt 2
        lattice_structure.set_values(lattice_values)
        # lattice_structure.set_val_dim(dim_of_sliced_values)
        lattice_structure.set_val_dim(lattice_values.shape[1])
        lattice_structure.set_val_full_dim(lattice_values.shape[1])

        sliced_values=lattice_structure.slice_standalone_no_precomputation(positions, with_homogeneous_coord)
        # import ipdb; ipdb.set_trace()

        ctx.save_for_backward(positions, lattice_structure.sliced_values_hom(), lattice_structure.splatting_indices(), lattice_structure.splatting_weights() )
        ctx.lattice_structure = lattice_structure
        ctx.with_homogeneous_coord=with_homogeneous_coord
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking


        #sanity check that we sliced enough values
        if with_error_checking:
            print("after slicing we recreate the splatting indices and they should all be valid")
            indices=lattice_structure.splatting_indices()
            indices=indices.detach()
            indices=indices[lattice_structure.nr_lattice_vertices()*lattice_structure.pos_dim()]
            nr_invalid=(indices==-1).sum()
            print("nr invalid is ", nr_invalid)
            if(nr_invalid.item()>500):
                sys.exit("there are too many positions which end up in empty space...")


        return sliced_values



       
    @staticmethod
    def backward(ctx, grad_sliced_values):
        
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking

        if with_debug_output:
            print("slice backwards")
            print("input slice backwards is grad_sliced_values ", grad_sliced_values)
            print("input slice backwards is grad_sliced_values which has shape", grad_sliced_values.shape)
            print("input slice backwards is grad_sliced_values which has norm", grad_sliced_values.norm())


        #attempt2
        #INPUT 
        #grad_sliced_values is the gradient of the loss wrt to the sliced_values
        #dL_dsliced_values
        #OUTPUT
        #must return dL_dlattice,  dL_dpositions is not really necessary as we will not optimize for the positions
        #dL_dlattice must have the same size as the values of the lattice
        #LOCAL
        #local gradient: dsliced_values_dlattice (gigantic matrix in size sliced_values.numel x d_lattice.values.numel )
        #No need to compute the local gradient here because it's a gigantic matrix, we cna get directly the dL_dlattice by splatting

        positions, sliced_values_hom, splatting_indices, splatting_weights =ctx.saved_tensors
        lattice_py = ctx.lattice_structure

        lattice_py.set_splatting_indices(splatting_indices)
        lattice_py.set_splatting_weights(splatting_weights)


        # values_forward_vertices=lattice_py.lattice.svalues().clone()
        # print("values_forward vertice is ", values_forward_vertices)
        # lattice_py.begin_splat_modify_only_values()
        if with_debug_output:
            print("slice backwards has a lattice_py with indices", lattice_py.splatting_indices())
            print("slice backwards has a lattice_py with weights", lattice_py.splatting_weights())
        #divide by the homogeneous coordinate
        # grad_sliced_values/=lattice_py.lattice.values()[:, 3:4]
        # lattice_py.splat_standalone(positions, grad_sliced_values)
        # print("in slice backwards, the indices are", lattice_py.splatting_indices())
        # print("grad_sliced_values is", grad_sliced_values)
        # print("grad_sliced_values sum is", grad_sliced_values.sum())
        # print("grad_sliced_values abs sum is", torch.abs(grad_sliced_values).sum())
        # print("grad_sliced_values max is", grad_sliced_values.max())
        # print("grad_sliced_values has shape", grad_sliced_values.shape)
    # torch::Tensor slice_backwards_standalone_with_precomputation(torch::Tensor& positions_raw, const torch::Tensor& values_forward_vertices, const torch::Tensor& grad_sliced_values);
        if with_debug_output:
            print("slice backwards with_homogeneous coord",ctx.with_homogeneous_coord)
        if(ctx.with_homogeneous_coord):
            lattice_py.lattice.slice_backwards_standalone_with_precomputation(positions, sliced_values_hom, grad_sliced_values)
        else:
            # print("splatting in the backward step")
            # print("splatting in the backward step val dim is", lattice_py.val_dim())
            # print("splatting in the backward step val_full dim is", lattice_py.val_full_dim())
            # print("lattice_py. has nr vertices", lattice_py.nr_lattice_vertices() )
            # print("lattice values has shape", lattice_py.values().shape )
            # lattice_py.lattice.splat_standalone(positions, grad_sliced_values, False)
            if(lattice_py.val_full_dim() is not grad_sliced_values.shape[2]):
                sys.exit("for some reason the values stored in the lattice are not the same dimension as the gradient. What?")
            lattice_py.set_val_dim(grad_sliced_values.shape[2])
            lattice_py.set_val_full_dim(grad_sliced_values.shape[2])
            # print("lattice has val dim set to ", lattice_py.val_dim())
            # print("lattice has val_full dim set to ", lattice_py.val_full_dim())
            # lattice_py.lattice.splat_standalone(positions, grad_sliced_values, False) 
            lattice_py.lattice.slice_backwards_standalone_with_precomputation_no_homogeneous(positions, grad_sliced_values) 
        # lattice_values=lattice_py.values().clone() #we get a pointer to the values so they don't dissapear when we realease the lettice
        lattice_values=lattice_py.values() #we get a pointer to the values so they don't dissapear when we realease the lettice
        # lattice_values=lattice_py.latt #we get a pointer to the values so they don't dissapear when we realease the lettice
        # print("lattice values for the gradient after splatting backwards are", lattice_values)
        # print("lattice values for the gradient after splatting backwards have max", lattice_values.max())
        # print("lattice values for the gradient after splatting backwards have sum", lattice_values.sum())
        # print("lattice values for the gradient after splatting backwards have shape", lattice_values.shape)
        # print("lattice values after graddient splatting backwards have non zero", (lattice_values==0).sum() )

        # print("after slicing backwards the lattice values is ", lattice_values)

        #make a new lattice and dont use the one used in the ctx
        # lattice_grad=LatticePy()
        # lattice_grad.create(config_file,"lattice_grad")
        # lattice_grad.begin_splat()
        # lattice_grad.splat_standalone(positions, grad_sliced_values)

        # lattice_values=lattice_py.lattice.values().clone() #we get a pointer to the values so they don't dissapear when we realease the lettice
        # lattice_values=lattice_grad.lattice.values().clone() #we get a pointer to the values so they don't dissapear when we realease the lettice
        # lattice_new=lattice_py

        # lattice_new=LatticePy()
        # lattice_new.lattice.m_name="new_lattice"
        # lattice_new.lattice=ctx.lattice_py.lattice
        ctx.lattice_structure=0 # release the pointer to this so it gets cleaned up
        # print("SLICING BACKWARDS DOESNT RELEASE THE LATTICE AT THE MOMENT BEUCASE I AM DOING GRAD_CHECK")

        #attempt 3 after talking with jan
        #grad sliced values is the grad of the loss wrt to the sliced_values
        #one needs to compute the gradient of a certain sliced value wrt to the values in the vertices
        #the gradient of loss wrt to the sliced value will get multiplied by the gradient of the sliced value with rest to the vertex values
        #then each vertex will acumulate the gradients

        # print("SLICE backwards, returning grad lattice_values", lattice_values)
        if with_debug_output:
            print("SLICE backwards, returning grad lattice_values ", lattice_values)
            print("SLICE backwards, returning grad lattice_values of norm", lattice_values.norm())
            print("SLICE backwards, returning grad lattice_values of sum", lattice_values.sum())
            print("SLICE backwards, returning grad lattice_values of shape", lattice_values.shape)

        # return lattice_new, None
        return lattice_values, None, None, None, None, None, None
        # return torch.rand_like(lattice_values), None


class SliceElevatedVertsLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values_to_slice_from, lattice_structure_to_slice_from, lattice_structure_elevated_verts, with_debug_output, with_error_checking):

        #attempt 2
        lattice_structure_to_slice_from.set_values(lattice_values_to_slice_from)

        sliced_lattice=lattice_structure_elevated_verts.slice_elevated_verts(lattice_structure_to_slice_from)


        ctx.save_for_backward( lattice_structure_elevated_verts.splatting_indices().clone(), lattice_structure_elevated_verts.splatting_weights().clone() )
        ctx.lattice_structure_to_slice_from = lattice_structure_to_slice_from
        ctx.lattice_structure_elevated_verts = lattice_structure_elevated_verts
        ctx.nr_verts_to_slice_from=lattice_structure_to_slice_from.nr_lattice_vertices()
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking

        return sliced_lattice.values(), sliced_lattice



       
    @staticmethod
    def backward(ctx, grad_sliced_values, grad_lattice_structure):

        splatting_indices, splatting_weights =ctx.saved_tensors
        lattice_structure_to_slice_from = ctx.lattice_structure_to_slice_from
        lattice_structure_elevated_verts = ctx.lattice_structure_elevated_verts
        nr_verts_to_slice_from=ctx.nr_verts_to_slice_from
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking

        lattice_structure_elevated_verts.set_splatting_indices(splatting_indices)
        lattice_structure_elevated_verts.set_splatting_weights(splatting_weights)

      
        # lattice_structure_to_slice_from.begin_splat_modify_only_values()

        lattice_structure_to_slice_from.set_val_dim(grad_sliced_values.shape[1])
        lattice_structure_to_slice_from.set_val_full_dim(grad_sliced_values.shape[1])
        lattice_structure_elevated_verts.set_val_dim(grad_sliced_values.shape[1])
        lattice_structure_elevated_verts.set_val_full_dim(grad_sliced_values.shape[1])

        lattice_structure_elevated_verts.lattice.slice_backwards_elevated_verts_with_precomputation(lattice_structure_to_slice_from.lattice, grad_sliced_values, nr_verts_to_slice_from) 
        lattice_values=lattice_structure_to_slice_from.values().clone() #we get a pointer to the values so they don't dissapear when we realease the lettice
      
        ctx.lattice_structure_to_slice_from=0 # release the pointer to this so it gets cleaned up
        ctx.lattice_structure_elevated_verts=0 # release the pointer to this so it gets cleaned up

        if with_debug_output:
            print("SLICE backwards, returning grad lattice_values of norm", lattice_values.norm())
            print("SLICE backwards, returning grad lattice_values of sum", lattice_values.sum())

        # return lattice_new, None
        return lattice_values, None, None, None, None
        # return torch.rand_like(lattice_values), None

class SliceClassifyLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values, lattice_structure, positions, delta_weights, linear_clasify_weight, linear_clasify_bias, nr_classes, with_debug_output, with_error_checking):


        if with_debug_output:
            print("slice clasify forward received a lattice structure with values ", lattice_structure.values())
            print("slice clasify forward received a lattice structure with nr of vertices", lattice_structure.nr_lattice_vertices())
            print("slice clasify forward received a lattice structure with values of shape", lattice_structure.values().shape)
        
        # print("linear_clasify_weight has shape ", linear_clasify_weight.shape)
        # sys.exit("stop for debugging")

        #IN GRAD CHECK WE SET THE VALUES TO ALL ZEROS SO THEY SHOULD BE SO

        #attempt 2
        lattice_structure.set_values(lattice_values)
        lattice_structure.set_val_dim(lattice_values.shape[1])
        lattice_structure.set_val_full_dim(lattice_values.shape[1])

        initial_values=lattice_values #needed fo the backwards pass TODO maybe the clone is not needed?

        # print("before slice_no_computation indices are ", lattice_structure.splatting_indices())

        class_logits=lattice_structure.lattice.slice_classify_with_precomputation(positions, delta_weights, linear_clasify_weight, linear_clasify_bias, nr_classes)

        # print("after slice_no_computation indices are ", lattice_structure.splatting_indices())

        # print("slice indices is ", lattice_structure.splatting_indices())
        # ctx.save_for_backward(positions, initial_values, lattice_structure.splatting_indices(), lattice_structure.splatting_weights(), delta_weights, linear_clasify_weight, linear_clasify_bias )
        ctx.save_for_backward(positions, initial_values, delta_weights, linear_clasify_weight, linear_clasify_bias )
        ctx.lattice_structure = lattice_structure
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking
        ctx.val_full_dim=lattice_values.shape[1]
        ctx.nr_classes=nr_classes


        #sanity check that we sliced enough values
        if with_error_checking:
            print("after slicing we recreate the splatting indices and they should all be valid")
            indices=lattice_structure.splatting_indices()
            indices=indices.detach()
            indices=indices[lattice_structure.nr_lattice_vertices()*lattice_structure.pos_dim()]
            nr_invalid=(indices==-1).sum()
            print("nr invalid is ", nr_invalid)
            if(nr_invalid.item()>500):
                sys.exit("there are too many positions which end up in empty space...")


        return class_logits



       
    @staticmethod
    def backward(ctx, grad_class_logits):
        
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking

        if with_debug_output:
            print("slice clasify backwards")
            print("input slice clasify backwards is grad_class_logits ", grad_class_logits)
            print("input slice clasify backwards is grad_class_logits which has shape", grad_class_logits.shape)
            print("input slice clasify backwards is grad_class_logits which has norm", grad_class_logits.norm())


        # positions, initial_values, splatting_indices, splatting_weights, delta_weights, linear_clasify_weight, linear_clasify_bias =ctx.saved_tensors
        positions, initial_values, delta_weights, linear_clasify_weight, linear_clasify_bias =ctx.saved_tensors
        lattice_py = ctx.lattice_structure
        val_full_dim=ctx.val_full_dim
        nr_classes=ctx.nr_classes

        # lattice_py.set_splatting_indices(splatting_indices)
        # lattice_py.set_splatting_weights(splatting_weights)
        lattice_py.set_val_full_dim(val_full_dim)


        # if with_debug_output:
            # print("slice backwards has a lattice_py with indices", lattice_py.splatting_indices())
            # print("slice backwards has a lattice_py with weights", lattice_py.splatting_weights())

        #create some tensors to host the gradient wrt to lattice_values, delta_weights, linear_weight and linear_bias
        grad_lattice_values=torch.zeros_like( lattice_py.values() )
        grad_delta_weights=torch.zeros_like( delta_weights )
        grad_linear_clasify_weight=torch.zeros_like(linear_clasify_weight)
        grad_linear_clasify_bias=torch.zeros_like(linear_clasify_bias)

        # print("grad delta weight has shape ",grad_delta_weights.shape)
        # print("intiial values is ",initial_values)

        lattice_py.lattice.slice_classify_backwards_with_precomputation(grad_class_logits, positions, initial_values, delta_weights, linear_clasify_weight, linear_clasify_bias, nr_classes,
                                                                                    grad_lattice_values, grad_delta_weights, grad_linear_clasify_weight, grad_linear_clasify_bias) 
        # grad_lattice_values=lattice_py.values().clone() #we get a pointer to the values so they don't dissapear when we realease the lettice

        #set the values to zero, because it contains the gradient and we don't want it to pollute another possible slice
        # lattice_py.set_values(initial_values)

        # print("after backwards the lattice values are ", lattice_py.values())
        # print("after backwards the initial values are ", initial_values )

       
        ctx.lattice_structure=0 # release the pointer to this so it gets cleaned up
        # print("SLICING CLASIFY BACKWARDS DOESNT RELEASE THE LATTICE AT THE MOMENT BEUCASE I AM DOING GRAD_CHECK")

        
        if with_debug_output:
            # print("SLICE backwards, returning grad lattice_values ", lattice_values)
            print("SLICE backwards, returning grad lattice_values of norm", grad_lattice_values.norm())
            # print("SLICE backwards, returning grad lattice_values of shape", grad_lattice_values.shape)
            print("SLICE backwards, returning grad_delta_weights of norm", grad_delta_weights.norm())
            # print("SLICE backwards, returning grad delta_weights of shape", grad_delta_weights.shape)
            print("SLICE backwards, returning grad_linear_clasify_weight of norm", grad_linear_clasify_weight.norm())
            # print("SLICE backwards, returning grad linear_clasify_weight of shape", grad_linear_clasify_weight.shape)
            # print("SLICE backwards, returning grad_linear_clasify_weight", grad_linear_clasify_weight )
            print("SLICE backwards, returnign", grad_lattice_values )

        # sys.exit("stop for debugging")

        return grad_lattice_values, None, None, grad_delta_weights, grad_linear_clasify_weight, grad_linear_clasify_bias, None, None, None

class GatherLattice(Function):
    @staticmethod
    def forward(ctx, lattice_values, lattice_structure, positions,  with_debug_output, with_error_checking):
       
        # return sliced_values
        if with_debug_output:
            print("gather forward received a lattice structure with nr of vertices", lattice_structure.nr_lattice_vertices())
            print("gather forward received a lattice structure with values of shape", lattice_structure.values().shape)


        #attempt 2
        lattice_structure.set_values(lattice_values)
        lattice_structure.set_val_dim(lattice_values.shape[1])
        lattice_structure.set_val_full_dim(lattice_values.shape[1])


        # gathered_values=lattice_structure.gather_standalone_no_precomputation(positions)
        gathered_values=lattice_structure.gather_standalone_with_precomputation(positions)

        # initial_values=lattice_structure.values().clone()
        # #debug if we were to ungather the values we should obtain teh same values
        # lattice_structure.lattice.gather_backwards_standalone_with_precomputation(positions, gathered_values) 
        # after_values=lattice_structure.values().clone()
        # diff=initial_values-after_values
        # diff_norm=diff.norm()
        # print("diff norm is ", diff_norm)


        # ctx.save_for_backward(positions, lattice_structure.splatting_indices(), lattice_structure.splatting_weights() )
        ctx.save_for_backward(positions)
        # ctx.save_for_backward(positions, lattice_structure.splatting_indices().clone(), lattice_structure.splatting_weights().clone() )
        ctx.lattice_structure = lattice_structure
        ctx.val_full_dim=lattice_values.shape[1]
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking


        #sanity check that we sliced enough values
        if with_error_checking:
            print("after gathering we recreate the splatting indices and they should all be valid")
            indices=lattice_structure.splatting_indices()
            indices=indices.detach()
            indices=indices[lattice_structure.nr_lattice_vertices()*lattice_structure.pos_dim()]
            nr_invalid=(indices==-1).sum()
            print("nr invalid is ", nr_invalid)
            if(nr_invalid.item()>500):
                sys.exit("there are too many positions which end up in empty space...")


        return gathered_values



       
    @staticmethod
    def backward(ctx, grad_sliced_values):
        
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking

        if with_debug_output:
            print("gather backwards")
            print("input gather backwards is grad_sliced_values which has shape", grad_sliced_values.shape)
            print("input gather backwards is grad_sliced_values which has norm", grad_sliced_values.norm())


        # positions,splatting_indices, splatting_weights =ctx.saved_tensors
        positions, =ctx.saved_tensors
        lattice_py = ctx.lattice_structure
        val_full_dim=ctx.val_full_dim


        # lattice_py.set_splatting_indices(splatting_indices)
        # lattice_py.set_splatting_weights(splatting_weights)

        # print("at the begginign of gather the values has shape ", lattice_py.values().shape)


        # lattice_py.begin_splat_modify_only_values()
        # help(lattice_py.lattice)
        lattice_py.set_val_full_dim(val_full_dim)
        lattice_py.lattice.gather_backwards_standalone_with_precomputation(positions, grad_sliced_values) 
        lattice_values=lattice_py.values() #we get a pointer to the values so they don't dissapear when we realease the lettice
      
        # print("at the end of gather the values has shape ", lattice_py.values().shape)


        ctx.lattice_structure=0 # release the pointer to this so it gets cleaned up
        # print("WE ARE NOT RELEASING THE LATTICE")

       
        # print("GATHER backwards, returning grad lattice_values", lattice_values)
        if with_debug_output:
            print("GATHER backwards, returning grad lattice_values of norm", lattice_values.norm())
            print("GATHER backwards, returning grad lattice_values of sum", lattice_values.sum())
            print("GATHER backwards, returning grad lattice_values of shape", lattice_values.shape)

        return lattice_values, None, None, None, None, None

class GatherElevatedLattice(Function):
    @staticmethod
    def forward(ctx, lattice_structure, lv_to_gather_from, ls_to_gather_from,  with_debug_output, with_error_checking):
       

        #attempt 2
        ls_to_gather_from.set_values(lv_to_gather_from)
        ls_to_gather_from.set_val_dim(lv_to_gather_from.shape[1])
        ls_to_gather_from.set_val_full_dim(lv_to_gather_from.shape[1])


        gathered_values=lattice_structure.gather_elevated_standalone_no_precomputation(ls_to_gather_from)


        # ctx.save_for_backward(positions, lattice_structure.splatting_indices(), lattice_structure.splatting_weights() )
        ctx.save_for_backward(lattice_structure.splatting_indices().clone(), lattice_structure.splatting_weights().clone() )
        ctx.lattice_structure = lattice_structure
        ctx.ls_to_gather_from = ls_to_gather_from
        ctx.val_full_dim=lv_to_gather_from.shape[1]
        ctx.with_debug_output=with_debug_output
        ctx.with_error_checking=with_error_checking


        #sanity check that we sliced enough values
        if with_error_checking:
            print("after gathering we recreate the splatting indices and they should all be valid")
            indices=lattice_structure.splatting_indices()
            indices=indices.detach()
            indices=indices[lattice_structure.nr_lattice_vertices()*lattice_structure.pos_dim()]
            nr_invalid=(indices==-1).sum()
            print("nr invalid is ", nr_invalid)
            if(nr_invalid.item()>500):
                sys.exit("there are too many positions which end up in empty space...")


        return gathered_values



       
    @staticmethod
    def backward(ctx, grad_sliced_values):
        
        with_debug_output=ctx.with_debug_output
        with_error_checking=ctx.with_error_checking

        if with_debug_output:
            print("gather backwards")
            print("input gather backwards is grad_sliced_values which has shape", grad_sliced_values.shape)
            print("input gather backwards is grad_sliced_values which has norm", grad_sliced_values.norm())
            print("input gather backwards is grad_sliced_values which has max", grad_sliced_values.max())
            print("input gather backwards is grad_sliced_values which has min", grad_sliced_values.min())


        splatting_indices, splatting_weights =ctx.saved_tensors
        lattice_py = ctx.lattice_structure
        ls_gathered_from = ctx.ls_to_gather_from
        val_full_dim=ctx.val_full_dim


        lattice_py.set_splatting_indices(splatting_indices)
        lattice_py.set_splatting_weights(splatting_weights)


        lattice_py.set_val_full_dim(val_full_dim)
        lattice_py.lattice.gather_backwards_elevated_standalone_with_precomputation(ls_gathered_from.lattice, grad_sliced_values) 
        lattice_values=ls_gathered_from.values() #we get a pointer to the values so they don't dissapear when we realease the lettice
      
        # print("at the end of gather the values has shape ", lattice_py.values().shape)


        ctx.lattice_structure=0 # release the pointer to this so it gets cleaned up
        ctx.ls_gathered_from=0
        # print("WE ARE NOT RELEASING THE LATTICE")

       
        # print("GATHER backwards, returning grad lattice_values", lattice_values)
        if with_debug_output:
            print("GATHER backwards, returning grad lattice_values of shape", lattice_values.shape)
            print("GATHER backwards, returning grad lattice_values of sum", lattice_values.sum())
            print("GATHER backwards, returning grad lattice_values of norm", lattice_values.norm())
            print("GATHER backwards, returning grad lattice_values of min", lattice_values.min())
            print("GATHER backwards, returning grad lattice_values of max", lattice_values.max())

        # sys.exit("debug gather backwards")

        return None, lattice_values, None, None, None, None
# class CloneLattice(Function):
#     @staticmethod
#     def forward(ctx, lattice_py):
#         cloned_lattice_py=lattice_py.clone_lattice()

#         cloned_lattice_py.set_values(torch.rand(1))
#         cloned_lattice_py.data=cloned_lattice_py.values()

#         return cloned_lattice_py

#     @staticmethod
#     def backward(ctx, grad_cloned_lattice):
#         return grad_cloned_lattice


class CloneLattice(Function):
    @staticmethod
    def forward(ctx, lattice_py, filter_bank):

        convolved_lattice_py=lattice_py.clone_lattice()
        return convolved_lattice_py

    @staticmethod
    def backward(ctx, grad_cloned_lattice):
        return grad_cloned_lattice, None


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


        
       
