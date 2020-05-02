import torch
from torch.autograd import Function
from torch import Tensor
from torch.nn import functional as F

import sys
from latticenet  import HashTable
from latticenet  import Lattice
import numpy as np
import time
import math
import torch_scatter
from latticenet_py.lattice.lattice_py import LatticePy
from latticenet_py.lattice.lattice_funcs import *


class DropoutLattice(torch.nn.Module):
    def __init__(self, prob):
        super(DropoutLattice, self).__init__()
        self.dropout=torch.nn.Dropout2d(p=prob)
    def forward(self, lv):

        if(len(lv.shape)) is not 2:
            sys.exit("the lattice values must be two dimensional, nr_lattice vertices x val_dim.However it is",len(lv.shape) ) 

        #droput expect input of shape N,C,H,W and drops a full channel
        lv_drop=lv.transpose(0,1) #val_full_dim x nr_lattice_vertices
        lv_drop=lv_drop.unsqueeze(0).unsqueeze(3)
        lv_drop=self.dropout(lv_drop)
        lv_drop=lv_drop.squeeze(3).squeeze(0)
        lv_drop=lv_drop.transpose(0,1)

        # ls.set_values(lv_drop) 

        # return lv_drop, ls
        return lv_drop

#modules
class CreateVertsModule(torch.nn.Module):
    def __init__(self, with_debug_output, with_error_checking):
        super(CreateVertsModule, self).__init__()
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
    def forward(self, lattice_py, positions):
        return CreateVerts.apply(lattice_py, positions, self.with_debug_output, self.with_error_checking)


class SplatLatticeModule(torch.nn.Module):
    def __init__(self):
        super(SplatLatticeModule, self).__init__()
    def forward(self, lattice_py, positions, values):
        return SplatLattice.apply(lattice_py, positions, values )

class DistributeLatticeModule(torch.nn.Module):
    def __init__(self, experiment, with_debug_output, with_error_checking):
        super(DistributeLatticeModule, self).__init__()
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.experiment=experiment
        # self.dummy_weight = torch.nn.Parameter( torch.empty( 1 ).to("cuda") ) #works for ConvIm2RowLattice
    def forward(self, lattice_py, positions, values):
        # return DistributeLattice.apply(lattice_py, positions, values, self.dummy_weight)
        return DistributeLattice.apply(lattice_py, positions, values, self.experiment, self.with_debug_output, self.with_error_checking)

class DistributeCapLatticeModule(torch.nn.Module):
    def __init__(self,):
        super(DistributeCapLatticeModule, self).__init__()
    def forward(self, distributed, nr_positions, ls, cap):
        indices=ls.splatting_indices()
        weights=ls.splatting_weights()
        indices_long=indices.long()
        #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long<0]=0

        ones=torch.ones(indices.shape[0]).to("cuda")


        nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
        nr_points_per_simplex=nr_points_per_simplex[1:] #the invalid simplex is at zero, the one in which we accumulate or the splatting indices that are -1

        # mask = indices.ge(0.5).to("cuda")
        # mask=torch.cuda.FloatTensor(indices.size(0)).uniform_() > 0.995
        # mask=mask.unsqueeze(1)

        # nr_positions= distributed.size(0)/(ls.pos_dim() +1  )
        mask=ls.lattice.create_splatting_mask(nr_points_per_simplex.int(), int(nr_positions), cap )

        # indices=indices.unsqueeze(1)
        print("distributed ", distributed.shape)
        print("indices ", indices.shape)
        print("weights ", weights.shape)
        print("mask ", mask.shape)

        #print the tensors
        print("mask is ", mask)
        print("indices is ", indices)
        print("weights is ", weights)
        print("distributed is ", distributed)

        capped_indices=torch.masked_select(indices, mask )
        capped_weights=torch.masked_select(weights, mask )
        capped_distributed=torch.masked_select(distributed, mask.unsqueeze(1))
        capped_distributed=capped_distributed.view(-1,distributed.size(1))

        print("capped_indices ", capped_indices.shape)
        print("capped_distributed ", capped_distributed.shape)

        # print("capped_indices is ", capped_indices)
        # print("capped_weights is ", capped_weights)
        # print("capped_distributed is ", capped_distributed)

        ls.set_splatting_indices(capped_indices)
        ls.set_splatting_weights(capped_weights)



        return capped_distributed, capped_indices, ls





class ConvLatticeModule(torch.nn.Module):
    def __init__(self, nr_filters, neighbourhood_size, dilation=1, bias=True, with_debug_output=True, with_error_checking=True):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(ConvLatticeModule, self).__init__()
        self.first_time=True
        self.weight=None
        self.bias=None
        self.neighbourhood_size=neighbourhood_size
        self.nr_filters=nr_filters
        self.dilation=dilation
        self.use_bias=bias
        self.use_center_vertex_from_lattice_neigbhours=False
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking

    #as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self, filter_extent):
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu') #pytorch uses default leaky relu but we use relu as here https://github.com/szagoruyko/binary-wide-resnet/blob/master/wrn_mcdonnell.py and as in here https://github.com/pytorch/vision/blob/19315e313511fead3597e23075552255d07fcb2a/torchvision/models/resnet.py#L156
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        #use the same weight init as here  https://github.com/kevinzakka/densenet/blob/master/model.py and like the one from the original paper https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua#L138-L156
        # n = filter_extent*self.nr_filters
        # self.weight.data.normal_(0, np.sqrt(2. / n))
        # if self.bias is not None:
            # torch.nn.init.zeros_(self.bias)

    def forward(self, lattice_values, lattice_structure, lattice_neighbours_values=None, lattice_neighbours_structure=None):

        lattice_structure.set_values(lattice_values) #you have to set the values here and not in the conv func because if it's the first time we run this we need to have a valued val_full_dim
        if( lattice_neighbours_structure is not None):
            lattice_neighbours_structure.set_values(lattice_neighbours_values)


        if(self.first_time):
            self.first_time=False
            filter_extent=lattice_structure.lattice.get_filter_extent(self.neighbourhood_size)
            val_dim=lattice_structure.lattice.val_dim()
            self.weight = torch.nn.Parameter( torch.empty( filter_extent * val_dim, self.nr_filters ).to("cuda") ) #works for ConvIm2RowLattice
            if self.use_bias:
                self.bias = torch.nn.Parameter( torch.empty( self.nr_filters ).to("cuda") )
            # if(self.with_homogeneous_coord):
            #     self.bias = torch.nn.Parameter(torch.empty( self.nr_filters+1).to("cuda") )
            # else:
            #     self.bias = torch.nn.Parameter(torch.empty( self.nr_filters).to("cuda") )
            with torch.no_grad():
                self.reset_parameters(filter_extent)

        lv, ls=ConvIm2RowLattice.apply(lattice_values, lattice_structure, self.weight, self.dilation, lattice_neighbours_values, lattice_neighbours_structure, self.use_center_vertex_from_lattice_neigbhours, self.with_debug_output, self.with_error_checking)
        if self.use_bias:
            lv+=self.bias
        ls.set_values(lv)
        
        return lv, ls




class CoarsenLatticeModule(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(CoarsenLatticeModule, self).__init__()
        self.first_time=True
        self.nr_filters=nr_filters
        self.neighbourhood_size=1
        self.use_center_vertex_from_lattice_neigbhours=True
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        # self.use_center_vertex=True
        # self.bn1 = BatchNormLatticeModule(nr_filters)
        # self.relu = torch.nn.ReLU()

    #as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self, filter_extent):
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        

        #use the same weight init as here  https://github.com/kevinzakka/densenet/blob/master/model.py and like the one from the original paper https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua#L138-L156
        # n = filter_extent*self.nr_filters
        # self.weight.data.normal_(0, np.sqrt(2. / n))
        

    def forward(self, lattice_fine_values, lattice_fine_structure):
        lattice_fine_structure.set_values(lattice_fine_values)

        if(self.first_time):
            self.first_time=False
            filter_extent=lattice_fine_structure.lattice.get_filter_extent(self.neighbourhood_size) 
            val_dim=lattice_fine_structure.lattice.val_dim()
            self.weight = torch.nn.Parameter( torch.empty( filter_extent * val_dim, self.nr_filters ).to("cuda") ) #works for ConvIm2RowLattice
            # self.bias = torch.nn.Parameter(torch.empty( self.nr_filters).to("cuda") )
            with torch.no_grad():
                self.reset_parameters(filter_extent)

        lv, ls= CoarsenLattice.apply(lattice_fine_values, lattice_fine_structure, self.weight, self.use_center_vertex_from_lattice_neigbhours,  self.with_debug_output, self.with_error_checking) #this just does a convolution, we also need batch norm an non linearity

        # lv, ls = self.bn1(lv, ls)
        # lv=self.relu(lv)
        ls.set_values(lv)

        return lv, ls

class FinefyLatticeModule(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(FinefyLatticeModule, self).__init__()
        self.first_time=True
        self.nr_filters=nr_filters
        self.neighbourhood_size=1
        self.use_center_vertex_from_lattice_neigbhours=True
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        # self.use_center_vertex=True
        # self.bn1 = BatchNormLatticeModule(nr_filters)
        # self.relu = torch.nn.ReLU()

    #as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self, filter_extent):
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        # if self.bias is not None:
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in)
            # torch.nn.init.uniform_(self.bias, -bound, bound)

        # #use the same weight init as here  https://github.com/kevinzakka/densenet/blob/master/model.py and like the one from the original paper https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua#L138-L156
        # n = filter_extent*self.nr_filters
        # self.weight.data.normal_(0, np.sqrt(2. / n))

    def forward(self, lattice_coarse_values, lattice_coarse_structure, lattice_fine_structure):
        lattice_coarse_structure.set_values(lattice_coarse_values)
        lattice_fine_structure.set_val_dim(lattice_coarse_structure.val_dim())

        if(self.first_time):
            self.first_time=False
            filter_extent=lattice_fine_structure.lattice.get_filter_extent(self.neighbourhood_size) 
            val_dim=lattice_coarse_structure.lattice.val_dim()
            self.weight = torch.nn.Parameter( torch.empty( filter_extent * val_dim, self.nr_filters ).to("cuda") ) #works for ConvIm2RowLattice
            # self.bias = torch.nn.Parameter(torch.empty( self.nr_filters).to("cuda") )
            with torch.no_grad():
                self.reset_parameters(filter_extent)

        lv, ls= FinefyLattice.apply(lattice_coarse_values, lattice_coarse_structure, lattice_fine_structure, self.weight, self.use_center_vertex_from_lattice_neigbhours, self.with_debug_output, self.with_error_checking) #this just does a convolution, we also need batch norm an non linearity

        # lv, ls = self.bn1(lv, ls)
        # lv=self.relu(lv)
        ls.set_values(lv)

        return lv, ls



        # return CoarsenLattice.apply(lattice_fine_values, lattice_fine_structure, self.weight, self.bias, self.use_center_vertex)

#does a max pool around each verts towards the neigbhours
class CoarsenMaxLatticeModule(torch.nn.Module):
    def __init__(self, with_debug_output, with_error_checking):
        super(CoarsenMaxLatticeModule, self).__init__()
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.use_center_vertex_from_lattice_neigbhours=True

    def forward(self, lattice_values, lattice_structure):
        lattice_structure.set_values(lattice_values)

   
        # val_full_dim=lattice_values.shape[1]
        # filter_extent=lattice_structure.lattice.get_filter_extent(1)
        # nr_vertices=lattice_structure.nr_lattice_vertices()

        # lattice_rowified=lattice_structure.im2row(filter_extent=filter_extent, lattice_neighbours=None,  dilation=1, use_center_vertex_from_lattice_neighbours=False, flip_neighbours=False)
        #lattice rowified has shape  nr_vertices x (filter_extent*m_val_full_dim)
        # lattice_rowified=lattice_rowified.view(nr_vertices, filter_extent, val_full_dim)
        # lattice_values_maxed,_=lattice_rowified.max(1)
        # print("lattice_values_maxed", lattice_values_maxed.shape)

        # lattice_structure.set_values(lattice_values_maxed)


        # return lattice_values_maxed, lattice_structure



        #atteptm2

        #create a coarse lattice and return the lattice rowified of it
        lattice_rowified, coarse_structure= CoarsenAndReturnLatticeRowified.apply(lattice_values, lattice_structure, self.use_center_vertex_from_lattice_neigbhours,  self.with_debug_output, self.with_error_checking) 

        val_dim=lattice_values.shape[1]
        filter_extent=coarse_structure.lattice.get_filter_extent(1)
        nr_vertices=coarse_structure.nr_lattice_vertices()

        lattice_rowified=lattice_rowified.view(nr_vertices, filter_extent, val_dim)

        #during argmax, we try to ignore the values that have zero (the lattice vertices that are not allocated. the easiest way it to just set them to a negative number to ensure they are never selected by the max operation)
        lattice_rowified[lattice_rowified==0.0]=-9999999
        lattice_values_maxed,_=lattice_rowified.max(1)
        # print("lattice_values_maxed", lattice_values_maxed.shape)

        coarse_structure.set_values(lattice_values_maxed)


        return lattice_values_maxed, coarse_structure


#does a avg pool around each verts towards the neigbhours
class CoarsenAvgLatticeModule(torch.nn.Module):
    def __init__(self, with_debug_output, with_error_checking):
        super(CoarsenAvgLatticeModule, self).__init__()
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.use_center_vertex_from_lattice_neigbhours=True

    def forward(self, lattice_values, lattice_structure):
        lattice_structure.set_values(lattice_values)

   
        # val_full_dim=lattice_values.shape[1]
        # filter_extent=lattice_structure.lattice.get_filter_extent(1)
        # nr_vertices=lattice_structure.nr_lattice_vertices()

        # lattice_rowified=lattice_structure.im2row(filter_extent=filter_extent, lattice_neighbours=None,  dilation=1, use_center_vertex_from_lattice_neighbours=False, flip_neighbours=False)
        #lattice rowified has shape  nr_vertices x (filter_extent*m_val_full_dim)
        # lattice_rowified=lattice_rowified.view(nr_vertices, filter_extent, val_full_dim)
        # lattice_values_maxed,_=lattice_rowified.max(1)
        # print("lattice_values_maxed", lattice_values_maxed.shape)

        # lattice_structure.set_values(lattice_values_maxed)


        # return lattice_values_maxed, lattice_structure



        #atteptm2

        #create a coarse lattice and return the lattice rowified of it
        lattice_rowified, coarse_structure= CoarsenAndReturnLatticeRowified.apply(lattice_values, lattice_structure, self.use_center_vertex_from_lattice_neigbhours,  self.with_debug_output, self.with_error_checking) 

        val_dim=lattice_values.shape[1]
        filter_extent=coarse_structure.lattice.get_filter_extent(1)
        nr_vertices=coarse_structure.nr_lattice_vertices()

        lattice_rowified=lattice_rowified.view(nr_vertices, filter_extent, val_dim)


        #averagin of only the non zero values around each vertex neighboruhood can be done as  sum and afterwards a division by how many non zero elements have
        lattice_values_sum=lattice_rowified.sum(1)
        lattice_rowified[lattice_rowified!=0.0]=1.0
        lattice_values_non_zero=(lattice_rowified).sum(1)
        # print("lattice rowified is ", lattice_rowified)
        # print("lattice values non zero is ", lattice_values_non_zero)
        # print("lattice rowified has shape ", lattice_rowified.shape)
        # print("lattice values_sum has shape ", lattice_values_sum.shape)
        # print("lattice values_non_zero has shape ", lattice_values_non_zero.shape)
        lattice_values_avg=lattice_values_sum.div(lattice_values_non_zero+0.000001)




        coarse_structure.set_values(lattice_values_avg)


        return lattice_values_avg, coarse_structure

#does a gausian blur around the center vertex
class CoarsenBlurLatticeModule(torch.nn.Module):
    def __init__(self, with_debug_output, with_error_checking):
        super(CoarsenBlurLatticeModule, self).__init__()
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.use_center_vertex_from_lattice_neigbhours=True
        self.blur_filter=None

    def forward(self, lattice_values, lattice_structure):
        lattice_structure.set_values(lattice_values)

   
       
        #atteptm2

        #create a coarse lattice and return the lattice rowified of it
        lattice_rowified, coarse_structure= CoarsenAndReturnLatticeRowified.apply(lattice_values, lattice_structure, self.use_center_vertex_from_lattice_neigbhours,  self.with_debug_output, self.with_error_checking) 

        val_dim=lattice_values.shape[1]
        filter_extent=coarse_structure.lattice.get_filter_extent(1)
        nr_vertices=coarse_structure.nr_lattice_vertices()

        lattice_rowified=lattice_rowified.view(nr_vertices, filter_extent, val_dim)

        if self.blur_filter is None:
            self.blur_filter = torch.tensor([1.0, 1.0,  1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 8.0 ])/9.0
            self.blur_filter=self.blur_filter.to("cuda")
            self.blur_filter=self.blur_filter.unsqueeze(0).unsqueeze(2) #makes it 1x9x1


        #apply weight to each of the values of the lattice rowified 
        print("lattice riwifed has shape", lattice_rowified.shape)
        print("self.blur_filter has shape", self.blur_filter.shape)
        lattice_rowified=lattice_rowified.mul(self.blur_filter)

        #averagin of only the non zero values around each vertex neighboruhood can be done as  sum and afterwards a division by how many non zero elements have
        lattice_values_sum=lattice_rowified.sum(1)
        lattice_rowified[lattice_rowified!=0.0]=1.0
        lattice_values_non_zero=(lattice_rowified).sum(1)
        # print("lattice rowified is ", lattice_rowified)
        # print("lattice values non zero is ", lattice_values_non_zero)
        # print("lattice rowified has shape ", lattice_rowified.shape)
        # print("lattice values_sum has shape ", lattice_values_sum.shape)
        # print("lattice values_non_zero has shape ", lattice_values_non_zero.shape)
        lattice_values_avg=lattice_values_sum.div(lattice_values_non_zero+0.000001)




        coarse_structure.set_values(lattice_values_avg)


        return lattice_values_avg, coarse_structure
    
    

class SliceLatticeModule(torch.nn.Module):
    def __init__(self, with_debug_output=True, with_error_checking=True):
        super(SliceLatticeModule, self).__init__()
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
    def forward(self, lattice_values, lattice_structure, positions):

        lattice_structure.set_values(lattice_values)
        return SliceLattice.apply(lattice_values, lattice_structure, positions, self.with_debug_output, self.with_error_checking)

class GatherLatticeModule(torch.nn.Module):
    def __init__(self, with_debug_output=True, with_error_checking=True):
        super(GatherLatticeModule, self).__init__()
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
    def forward(self, lattice_values, lattice_structure, positions):

        lattice_structure.set_values(lattice_values)
        return GatherLattice.apply(lattice_values, lattice_structure, positions, self.with_debug_output, self.with_error_checking)




#the last attempt to make a fast slice without having a gigantic feature vector for each point. Rather from the features of the vertices, we regress directly the class probabilities
#the idea is to not do it with a gather but rather with a special slicing function that also gets as input some learnable weights
class SliceFastCUDALatticeModule(torch.nn.Module):
    def __init__(self, nr_classes, dropout_prob, experiment, with_debug_output=True, with_error_checking=True):
        super(SliceFastCUDALatticeModule, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.bottleneck=None
        self.stepdown=torch.nn.ModuleList([])
        self.bottleneck_size=8
        self.norm_pre_gather=None
        self.linear_pre_deltaW=None 
        self.linear_deltaW=None 
        self.linear_clasify=None
        self.tanh=torch.nn.Tanh()
        # self.relu=torch.nn.ReLU(inplace=True)
        self.dropout_prob=dropout_prob
        if(dropout_prob > 0.0):
            self.dropout =DropoutLattice(dropout_prob) 
        # self.drop_bottleneck =DropoutLattice(0.2) 

        self.conv1d=None

        self.experiment=experiment
    def forward(self, lv, ls, positions):

        # original_lv=lv.clone()
        ls.set_values(lv)

        nr_positions=positions.shape[0]
        pos_dim=positions.shape[1]
        val_dim=lv.shape[1]
        # print("SliceFast with nr_positions ", nr_positions, " pos_dim ", pos_dim, "val_full_dim ", val_full_dim)

        # #we want to regress first some weights deltas. For this we bottleneck first the values of the lattice so that the gather operation will be fast
        # if self.bottleneck is None: 
        #     self.bottleneck=GnRelu1x1(self.bottleneck_size, False, self.with_debug_output, self.with_error_checking)
        # lv_bottleneck, ls_bottleneck = self.bottleneck(lv, ls)

        #slowly reduce the features 
        if len(self.stepdown) is 0:
            for i in range(2):
                nr_channels_out= int( val_dim/np.power(2,i) ) 
                if nr_channels_out  < self.bottleneck_size:
                    sys.exit("We used to many linear layers an now the values are lower than the bottlenck size. Which means that the bottleneck would actually do an expansion...")
                print("adding stepdown with output of ", nr_channels_out)
                self.stepdown.append( GnRelu1x1(nr_channels_out , False, self.with_debug_output, self.with_error_checking)  )
                # self.stepdown.append( Gn1x1Gelu(nr_channels_out , False, self.with_debug_output, self.with_error_checking)  )
        if self.bottleneck is None:
            print("adding bottleneck with output of ", self.bottleneck_size)
            self.bottleneck=GnRelu1x1(self.bottleneck_size, False, self.with_debug_output, self.with_error_checking)            
            # self.bottleneck=Gn1x1Gelu(self.bottleneck_size, False, self.with_debug_output, self.with_error_checking)            
        # apply the stepdowns
        for i in range(2):
            if i == 0:
                lv_bottleneck, ls_bottleneck = self.stepdown[i](lv, ls)
            else:
                lv_bottleneck, ls_bottleneck = self.stepdown[i](lv_bottleneck, ls_bottleneck)
        # last bottleneck
        lv_bottleneck, ls_bottleneck = self.bottleneck(lv_bottleneck, ls_bottleneck)

     


        #SEEMS TO OVERFIT FASTER without this
        # after gathering we would like to do another bn relu conv but maybe the fastest option would be to apply the bn here because gathering doesnt change the statistics of the tensor too much
        # if self.norm_pre_gather is None:
            # self.norm_pre_gather = GroupNormLatticeModule(lv_bottleneck.shape[1])
        # lv_bottleneck, ls_bottleneck=self.norm_pre_gather(lv_bottleneck,ls_bottleneck)
        # lv_bottleneck=self.relu(lv_bottleneck) 
        # lv_bottleneck=gelu(lv_bottleneck) 
        # lv_bottleneck=self.drop_bottleneck(lv_bottleneck)

        # TIME_START("gather")
        sliced_bottleneck_rowified=GatherLattice.apply(lv_bottleneck, ls_bottleneck, positions, self.with_debug_output, self.with_error_checking)
        # TIME_END("gather")
        #sliced rowified, just after gathering has shape 1 x nr_positions x ( (m_pos_dim+1) x (val_full_dim+1)  ) ,so in our case nr_positions x (4* (self.bottleneck_size+1) )

        ##concat also the distributed
        # print("distributed_before_scattermax has shape ", distributed_before_scattermax.shape)
        # print("sliced_bottleneck_rowified has shape ", sliced_bottleneck_rowified.shape)
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.view(nr_positions*(pos_dim+1), -1 )
        # sliced_bottleneck_rowified=torch.cat((sliced_bottleneck_rowified,distributed_before_scattermax),1)
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.view(1,nr_positions, -1 )
        # print("sliced_bottleneck_rowified has shape ", sliced_bottleneck_rowified.shape)

        nr_vertices_per_simplex=ls.pos_dim()+1
        val_dim_of_each_vertex=int(sliced_bottleneck_rowified.shape[1]/ nr_vertices_per_simplex)

        #from this slice rowified we regress for each position some barycentric offest of size m_pos_dim+1
        # linear layers on the sliced rowified get us to a tensor of 1 x nr_positions x (m_pos_dim+1), this will be the weights offsets for eahc positions into the 4 lattice vertices
        if self.linear_deltaW is None:
            # self.linear_pre_deltaW=torch.nn.Linear(sliced_bottleneck_rowified.shape[2], sliced_bottleneck_rowified.shape[2], bias=False).to("cuda") 
            # self.gn_middle = torch.nn.GroupNorm( ls.pos_dim()+1,  self.bottleneck_size*4+4).to("cuda") #the nr of groups is the same as the m_pos_dim+1 which is usually 4
            # self.gn_middle = torch.nn.GroupNorm( self.bottleneck_size*4+4,  self.bottleneck_size*4+4).to("cuda") 
            # self.gn_middle = torch.nn.GroupNorm( 1,  self.bottleneck_size*4+4).to("cuda") 
            # self.gn_middle = torch.nn.GroupNorm( 1,  self.bottleneck_size+1).to("cuda") 
            # self.linear_deltaW=torch.nn.Linear(sliced_bottleneck_rowified.shape[2], pos_dim+1, bias=True).to("cuda") 
            self.linear_deltaW=torch.nn.Linear( int(sliced_bottleneck_rowified.shape[1]/ (ls.pos_dim()+1) ), 1, bias=True).to("cuda") 
            # self.gn = torch.nn.GroupNorm(1, sliced_bottleneck_rowified.shape[2]).to("cuda")
            with torch.no_grad():
                # torch.nn.init.kaiming_uniform_(self.linear_pre_deltaW.weight, mode='fan_in', nonlinearity='relu') 
                # self.linear_deltaW.weight.fill_(0.0) #we set the weights so that the initial deltaW are zero
                # self.linear_deltaW.bias.fill_(0.0)
                # self.linear_deltaW.weight.uniform_(-0.01,0.01) #we set the weights so that the initial deltaW are zero
                # self.linear_deltaW.bias.uniform_(-0.01, 0.01)
                torch.nn.init.kaiming_uniform_(self.linear_deltaW.weight, mode='fan_in', nonlinearity='tanh') 
                self.linear_deltaW.weight*=0.1 #make it smaller so that we start with delta weight that are close to zero
                torch.nn.init.zeros_(self.linear_deltaW.bias) 
            #attempt 2
            self.conv1d=torch.nn.Conv1d(sliced_bottleneck_rowified.shape[1], pos_dim+1, val_dim_of_each_vertex , stride=1, padding=0, dilation=1, groups=4, bias=True, padding_mode='zeros').to("cuda")
            with torch.no_grad():
                self.conv1d.weight*=0.1 #make it smaller so that we start with delta weight that are close to zero
                torch.nn.init.zeros_(self.conv1d.bias) 
            self.gamma  = torch.nn.Parameter( torch.ones( val_dim_of_each_vertex ).to("cuda") ) 
            self.beta  = torch.nn.Parameter( torch.zeros( val_dim_of_each_vertex ).to("cuda") ) 



        # sliced_bottleneck_rowified=self.relu(sliced_bottleneck_rowified) #shape 1 x nr_positions x (pos_dim+1)
        # print("pos dim +1", ls.pos_dim()+1 )
        # print("sliced_bottleneck_rowified has shape ", sliced_bottleneck_rowified.shape)
        #APPLY TODO decide if this is necessary. it seems to make overfitting a lot more difficult
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.transpose(1,2)
        # sliced_bottleneck_rowified=self.gn_middle(sliced_bottleneck_rowified) 
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.transpose(1,2)
        # sliced_bottleneck_rowified=F.gelu(sliced_bottleneck_rowified) 

        #in order to make it rotation invariant we have to apply it m_pos_dim times with various rotations of the kernels 
        # print("weights is" , self.linear_deltaW.weight.shape)
        # print("biases is" , self.linear_deltaW.bias.shape)
        # v= np.arange(0,12)
        # tens = torch.from_numpy(v)
        # tens= tens.reshape(2,6)
        # permutation=[]
        # nr_vertices_per_simplex=ls.pos_dim()+1
        # val_dim_of_each_vertex=int(sliced_bottleneck_rowified.shape[2]/ nr_vertices_per_simplex)
        # delta_weights=None
        # #DEBUG
        # # val_dim_of_each_vertex=2
        # for i in range (self.linear_deltaW.weight.shape[1]-val_dim_of_each_vertex): #weight will have shape like 4,36 we have to pemrute the 36 columns
        #     permutation.append(i+val_dim_of_each_vertex)
        # for i in range(val_dim_of_each_vertex):
        #     permutation.append(i)
        # print("permutation is ", permutation)

        # for i in range(ls.pos_dim()+1):
        #     # print("appying delta for rotation", i)
        #     #assume we have 3 vertices that we need to permute
        #     # tens=tens.transpose(0,1)
        #     # tens=tens.reshape(2,2,3)
        #     # permute=[1,2,0]
        #     # permute=[2,3,4,5,0,1]
        #     # tens=tens[:,:,permute]
        #     # tens=tens[:,permute]
        #     # tens=tens.transpose(0,1)
        #     # tens= tens.reshape(2,6)
        #     # print("tens is \n", tens)

        #     self.linear_deltaW.weight= torch.nn.Parameter(self.linear_deltaW.weight[:,permutation])
        #     delta_weights_cur_rot=self.linear_deltaW(sliced_bottleneck_rowified)
        #     if i==0:
        #         delta_weights=delta_weights_cur_rot
        #     else:
        #         delta_weights=torch.max(delta_weights, delta_weights_cur_rot)

        #chekc if the tensor and the parameter share memory




        ###ORIGNAL
        # delta_weights=self.linear_deltaW(sliced_bottleneck_rowified)








        ## ATTEMPT 2 try to predict each barycentric coordinate only for the corresponding vertex by using group convolution
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.transpose(1,2)
        # delta_weights=self.conv1d(sliced_bottleneck_rowified)
        # delta_weights=delta_weights.transpose(1,2)



        # TIME_START("remove_max")
        #attmept 4 predict a barycentric coordinate for each lattice vertex, and then use max over all the features in the simplex like in here https://arxiv.org/pdf/1611.04500.pdf
        sliced_bottleneck_rowified=sliced_bottleneck_rowified.view(nr_positions, nr_vertices_per_simplex, val_dim_of_each_vertex)
        # print("sliced_bottleneck_rowified has size", sliced_bottleneck_rowified.shape)
        #max over the al the vertices in a simplex
        max_vals,_=sliced_bottleneck_rowified.max(1)
        # max_vals=sliced_bottleneck_rowified.sum(2)/(ls.pos_dim()+1)
        max_vals=max_vals.unsqueeze(1)
        # print("max vals has size", max_vals.shape)
        # print("gamma is ", self.gamma)
        # print("beta is ", self.beta)

        sliced_bottleneck_rowified-= self.gamma* max_vals + self.beta #max vals broadcasts to all the vertices in the simplex and substracts the max from them
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.view(1,nr_positions, nr_vertices_per_simplex* val_dim_of_each_vertex)
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.transpose(1,2)
        # sliced_bottleneck_rowified=self.gn_middle(sliced_bottleneck_rowified) 
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.transpose(1,2)
        # sliced_bottleneck_rowified=sliced_bottleneck_rowified.view(1,nr_positions, nr_vertices_per_simplex, val_dim_of_each_vertex)
        # sliced_bottleneck_rowified=F.gelu(sliced_bottleneck_rowified)
        # sliced_bottleneck_rowified=torch.tanh(sliced_bottleneck_rowified)
        # print("gamma is", self.gamma)
        delta_weights=self.linear_deltaW(sliced_bottleneck_rowified)
        delta_weights=delta_weights.reshape(nr_positions, nr_vertices_per_simplex)
        # delta_weights=self.tanh(delta_weights)
        # TIME_END("remove_max")
 


















        # delta_weights=self.tanh(delta_weights)

        # sys.exit("what")

        # #gn,relu,linear
        # sliced_bottleneck_rowified_gn=sliced_bottleneck_rowified.transpose(1,2)
        # sliced_bottleneck_rowified_gn=self.gn(sliced_bottleneck_rowified_gn)
        # sliced_bottleneck_rowified_gn=sliced_bottleneck_rowified_gn.transpose(1,2)
        # sliced_bottleneck_rowified_gn=self.relu(sliced_bottleneck_rowified_gn)
        # delta_weights=self.linear_deltaW(sliced_bottleneck_rowified_gn)

        # print("linear deltaW weight is ", self.linear_deltaW.weight )


        # print("delta weights has shape ", delta_weights.shape)
        # sys.exit("what")
        if self.experiment=="slice_no_deform":
            # print("Using slice with no deform as the experiment is ", self.experiment)
            delta_weights*=0

        # print("delta weights is ", delta_weights)

        #ERROR FOR THE DELTAWEIGHTS
        #the delta of the barycentric coordinates should sum to zero so that the sum of the normal barycentric coordinates should still sum to 1.0
        # sum_bar=delta_weights.sum(2)
        # # print("sum_bar is ", sum_bar)
        # diff=sum_bar-0.0 #deviation from the expected value
        # diff2=diff.mul(diff)
        # # diff2=diff.abs()
        # delta_weight_error_sum=diff2.mean()
        delta_weight_error_sum=0.0

        #attempt 2 by just doing a dot product of every barycentric offset
        # dot_per_point=(delta_weights.mul(delta_weights)).sum(2)
        # delta_weight_error_sum=dot_per_point.mean()

        # print("delta_weight_error_sum is ", delta_weight_error_sum)
        # print("delta_weights min is ",delta_weights.min())
        # print("delta_weights max is ",delta_weights.max())

        #noerror
        # delta_weight_error_sum=0

        # #error based on norm, we want little movement
        # norm_bar=delta_weights.sum(2)
        # print("norm_bar is ", norm_bar)
        # diff2=norm_bar.mul(norm_bar) #we square it so that big norm (big movement in brycentric coords) have more error than small ones
        # delta_weight_error_sum=diff2.sum()
        # print("delta_weight_error_sum is ", delta_weight_error_sum)


        #we slice with the delta weights and we clasify at the same time
        if self.linear_clasify is None: #create the linear clasify but we use the tensors directly inside our cuda kernel
            self.linear_clasify=torch.nn.Linear(val_dim, self.nr_classes, bias=True).to("cuda") 

        # print("linear clasigfy weight has shpae", self.linear_clasify.weight.shape)
        # print("linear clasigfy bias has shpae", self.linear_clasify.bias.shape)
        # sys.exit("debug stop")

        if(self.dropout_prob > 0.0):
            lv=self.dropout(lv)


        ls.set_values(lv)
        # ls.set_values(original_lv)

        # print("chekc why is does the loss start higher when using the stepdown, it shouldnt make a difference because the additional branch doesnt do anything...")
        # print("lv is ", ls.values())
        # print("lv has norm ", ls.values().norm())
        # print("lv has max ", ls.values().max())
        # print("lv has min ", ls.values().min())
        # print("delta_weights is ", delta_weights)
        # sys.exit("debug stop")

        classes_logits = SliceClassifyLattice.apply(lv, ls, positions, delta_weights, self.linear_clasify.weight, self.linear_clasify.bias, self.nr_classes, self.with_debug_output, self.with_error_checking)

        # print("class logits is ", classes_logits)

        return classes_logits, delta_weight_error_sum


#this is required because if we do a batch norm of the lattice values directly it would be corrupted by all the zeros we have there
class BatchNormLatticeModule(torch.nn.Module):
    def __init__(self, nr_params, affine=True):
        super(BatchNormLatticeModule, self).__init__()
        self.bn=torch.nn.BatchNorm1d(num_features=nr_params, momentum=0.1, affine=affine).to("cuda")
        # self.bn=torch.nn.InstanceNorm1d(num_features=nr_params).to("cuda")
        # self.bn=torch.nn.LayerNorm(normalized_shape=nr_params, elementwise_affine=affine).to("cuda") #layer norm work better for small minibatches https://nealjean.com/ml/neural-network-normalization/
        # self.bn = torch.nn.GroupNorm(nr_params, nr_params).to("cuda")
    def forward(self,lattice_values, lattice_py):

        # lattice_py.set_values(lattice_values)

        # values_trim=lattice_values[0:lattice_py.nr_lattice_vertices(), :] # we need to get only the valid values otherwise batch norm will be botched
        # values_trim_normalized=self.bn(values_trim)

        # new_values=torch.zeros(lattice_py.capacity(), lattice_py.val_full_dim())
        # new_values[0:lattice_py.nr_lattice_vertices(), :] = values_trim_normalized

        # lattice_py.set_values(new_values)

        # return lattice_values, lattice_py



        #batch norm which only does batch norm over the whole values, of course this only work when they are not the same size as the capacity
        if(lattice_values.shape[1] == lattice_py.capacity()):
            sys.exit("This batchnor is thught to work for when using the values which has size exactly in the number of vertices")

        if(lattice_values.dim() is not 2):
            sys.exit("lattice should be 2 dimensional, nr_vertices x val_full_dim")
        
        # lattice_values=lattice_values.unsqueeze(0)
        # lattice_values=lattice_values.transpose(1,2)
        lattice_values=self.bn(lattice_values)
        # lattice_values=lattice_values.transpose(1,2)
        # lattice_values=lattice_values.squeeze(0)

        lattice_py.set_values(lattice_values)

        return lattice_values, lattice_py

#this is required because if we do a batch norm of the lattice values directly it would be corrupted by all the zeros we have there
class GroupNormLatticeModule(torch.nn.Module):
    def __init__(self, nr_params, affine=True):
        super(GroupNormLatticeModule, self).__init__()
        # self.gn = torch.nn.GroupNorm(nr_params, nr_params).to("cuda")
        nr_groups=32
        if nr_params<=32:
            nr_groups=int(nr_params/2)

        #TODO check that nr_params divides nicely by 32 and if not revert to something like nr_groups=nr_params

        self.gn = torch.nn.GroupNorm(nr_groups, nr_params).to("cuda") #having 32 groups is the best as explained in the GroupNormalization paper
        # self.gn = torch.nn.InstanceNorm1d(nr_params, affine=affine).to("cuda")
        #self.bn=torch.nn.BatchNorm1d(num_features=nr_params, momentum=0.1, affine=affine).to("cuda")
    def forward(self,lattice_values, lattice_py):

        #group norm which only does group norm over the whole values, of course this only work when they are not the same size as the capacity
        if(lattice_values.shape[1] == lattice_py.capacity()):
            sys.exit("This batchnor is thught to work for when using the values which has size exactly in the number of vertices")

        if(lattice_values.dim() is not 2):
            sys.exit("lattice should be 2 dimensional, nr_vertices x val_dim")

        #group norm wants the tensor to be N, C, L  (nr_batches, channels, nr_samples)
        lattice_values=lattice_values.unsqueeze(0)
        lattice_values=lattice_values.transpose(1,2)
        lattice_values=self.gn(lattice_values)
        lattice_values=lattice_values.transpose(1,2)
        lattice_values=lattice_values.squeeze(0)

        #do actually a bn just for testing
        #lattice_values=self.bn(lattice_values)

        lattice_py.set_values(lattice_values)

        return lattice_values, lattice_py

class LayerNormLatticeModule(torch.nn.Module):
    def __init__(self, nr_params, affine=True):
        super(LayerNormLatticeModule, self).__init__()
        self.bn=torch.nn.LayerNorm(normalized_shape=nr_params, elementwise_affine=affine).to("cuda") #layer norm work better for small minibatches https://nealjean.com/ml/neural-network-normalization/
    def forward(self,lattice_values, lattice_py):

        # lattice_py.set_values(lattice_values)

        # values_trim=lattice_values[0:lattice_py.nr_lattice_vertices(), :] # we need to get only the valid values otherwise batch norm will be botched
        # values_trim_normalized=self.bn(values_trim)

        # new_values=torch.zeros(lattice_py.capacity(), lattice_py.val_full_dim())
        # new_values[0:lattice_py.nr_lattice_vertices(), :] = values_trim_normalized

        # lattice_py.set_values(new_values)

        # return lattice_values, lattice_py



        #batch norm which only does batch norm over the whole values, of course this only work when they are not the same size as the capacity
        if(lattice_values.shape[1] == lattice_py.capacity()):
            sys.exit("This batchnor is thught to work for when using the values which has size exactly in the number of vertices")

        if(lattice_values.dim() is not 2):
            sys.exit("lattice should be 2 dimensional, nr_vertices x val_dim")
        
        # lattice_values=lattice_values.unsqueeze(0)
        lattice_values=self.bn(lattice_values)
        # lattice_values=lattice_values.squeeze(0)

        lattice_py.set_values(lattice_values)

        return lattice_values, lattice_py






class PointNetModule(torch.nn.Module):
    def __init__(self, nr_output_channels_per_layer, nr_outputs_last_layer, experiment, with_debug_output, with_error_checking):
        super(PointNetModule, self).__init__()
        self.first_time=True
        self.nr_output_channels_per_layer=nr_output_channels_per_layer
        self.nr_outputs_last_layer=nr_outputs_last_layer
        self.nr_linear_layers=len(self.nr_output_channels_per_layer)
        self.layers=torch.nn.ModuleList([])
        self.norm_layers=torch.nn.ModuleList([])
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.relu=torch.nn.ReLU(inplace=True)
        self.tanh=torch.nn.Tanh()
        self.leaky=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.experiment=experiment
    def forward(self, lattice_py, distributed, indices):
        if (self.first_time):
            with torch.no_grad():
                self.first_time=False

                #get the nr of channels of the distributed tensor
                nr_input_channels=distributed.shape[1] - 1
                if (self.experiment=="attention_pool"):
                    nr_input_channels=distributed.shape[1] 
                # initial_nr_channels=distributed.shape[1]

                nr_layers=0
                for i in range(len(self.nr_output_channels_per_layer)):
                    nr_output_channels=self.nr_output_channels_per_layer[i]
                    if self.with_debug_output:
                        print ("in ", nr_input_channels)
                        print ("out ", nr_output_channels)
                    is_last_layer=i==len(self.nr_output_channels_per_layer)-1 #the last layer is folowed by scatter max and not a batch norm therefore it needs a bias
                    # self.norm_layers.append( GroupNormLatticeModule(nr_params=nr_input_channels, affine=True)  )  #we disable the affine because it will be slow for semantic kitti
                    # print("is last layer is", is_last_layer)
                    self.layers.append( torch.nn.Linear(nr_input_channels, nr_output_channels, bias=is_last_layer).to("cuda")  )
                    with torch.no_grad():
                        torch.nn.init.kaiming_normal_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
                    self.norm_layers.append( GroupNormLatticeModule(nr_params=nr_output_channels, affine=True)  )  #we disable the affine because it will be slow for semantic kitti
                    nr_input_channels=nr_output_channels
                    nr_layers=nr_layers+1

                if (self.experiment=="attention_pool"):
                    self.pre_conv=torch.nn.Linear(nr_input_channels, nr_input_channels, bias=False).to("cuda") #the last distributed is the result of relu, so we want to start this paralel branch with a conv now
                    self.gamma  = torch.nn.Parameter( torch.ones( nr_input_channels ).to("cuda") ) 
                    with torch.no_grad():
                        torch.nn.init.kaiming_normal_(self.pre_conv.weight, mode='fan_in', nonlinearity='relu')
                    self.att_activ=GnRelu1x1(nr_input_channels, False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
                    self.att_scores=GnRelu1x1(nr_input_channels, True, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)


                self.last_conv=ConvLatticeModule(nr_filters=self.nr_outputs_last_layer, neighbourhood_size=1, dilation=1, bias=False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking) #disable the bias becuse it is followed by a gn

        # print("pointnet distributed at the beggining is ", distributed.shape)

        # initial_distributed=distributed

        barycentric_weights=distributed[:,-1]
        if ( self.experiment=="attention_pool"):
            distributed=distributed #when we use attention pool we use the distributed tensor that contains the barycentric weights
        else:
            distributed=distributed[:, :distributed.shape[1]-1] #IGNORE the barycentric weights for the moment and lift the coordinates of only the xyz and values
        # print("distriuted is ", distributed)
        # print("barycentric weights is ", barycentric_weights)

        # #run the distributed through all the layers
        experiment_that_imply_no_elevation=["pointnet_no_elevate", "pointnet_no_elevate_no_local_mean", "splat"]
        if self.experiment in experiment_that_imply_no_elevation:
            # print("not performing elevation by pointnet as the experiment is", self.experiment)
            pass
        else:
            for i in range(len(self.layers)): 

                #start directly with bn relu conv, so batch norming the input which may be a bad idea...
                # distributed, lattice_py=self.norm_layers[i] (distributed, lattice_py) 
                # distributed=self.relu(distributed) 
                # distributed=self.layers[i] (distributed)



                if (self.experiment=="attention_pool"):
                    distributed=self.layers[i] (distributed)
                    distributed, lattice_py=self.norm_layers[i] (distributed, lattice_py) 
                    distributed=self.relu(distributed) 
                else:
                    distributed=self.layers[i] (distributed)
                    if( i < len(self.layers)-1): #last tanh before the maxing need not be applied because it actually hurts the performance, also it's not used in the original pointnet https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
                        #last bn need not be applied because we will max over the lattices either way and then to a bn afterwards
                        distributed, lattice_py=self.norm_layers[i] (distributed, lattice_py) 
                    distributed=self.relu(distributed) 

                # #start with a first conv then does gn conv relu
                # if i!=0:
                #     distributed, lattice_py=self.norm_layers[i] (distributed, lattice_py) 
                # distributed=self.layers[i] (distributed)
                # if i!=0:
                #     distributed=self.relu(distributed) 



        # distributed=torch.cat((initial_distributed,distributed),1)


        # print("pointnet distributed at the end of all the first batch of linear layers is ", distributed.shape)

        indices_long=indices.long()

        #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long<0]=0




        if self.experiment=="splat":
            # print("performing splatting since the experiment is ", self.experiment)
            distributed_reduced = torch_scatter.scatter_mean(distributed, indices_long, dim=0)
        if self.experiment=="attention_pool":
            #attention pooling ##################################################
            #concat for each vertex the max over the simplex
            max_reduced, argmax = torch_scatter.scatter_max(distributed, indices_long, dim=0)
            max_per_vertex=torch.index_select(max_reduced, 0, indices_long)
            # distributed_with_max=torch.cat( (distributed, max_per_vertex),1 )
            distributed_with_max=distributed+self.gamma*max_per_vertex
            # print("gamma is ", self.gamma)

            pre_conv=self.pre_conv(distributed_with_max) 
            att_activ, lattice_py=self.att_activ(pre_conv, lattice_py)
            att_scores, lattice_py=self.att_scores(att_activ, lattice_py)
            att_scores=torch.exp(att_scores)
            att_scores_sum_reduced = torch_scatter.scatter_add(att_scores, indices_long, dim=0)
            att_scores_sum=torch.index_select(att_scores_sum_reduced, 0, indices_long)
            att_scores=att_scores/att_scores_sum
            #softmax them somehow
            distributed=distributed*att_scores
            distributed_reduced = torch_scatter.scatter_add(distributed, indices_long, dim=0)

            # #concat also the att scores weighted by the barucentric coords
            # barycentric_weights=barycentric_weights.unsqueeze(1)
            # att_scores_weighted= barycentric_weights*att_scores
            # att_scores_weighted_reduced,_ = torch_scatter.scatter_max(att_scores_weighted, indices_long, dim=0)
            # distributed_reduced=torch.cat((distributed_reduced,att_scores_weighted_reduced),1)

            #get also the nr of points in the lattice so the max pooled features can be different if there is 1 point then if there are 100
            ones=torch.cuda.FloatTensor( indices_long.shape[0] ).fill_(1.0)
            nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
            nr_points_per_simplex=nr_points_per_simplex.unsqueeze(1)
            minimum_points_per_simplex=4
            simplexes_with_few_points=nr_points_per_simplex<minimum_points_per_simplex
            distributed_reduced.masked_fill_(simplexes_with_few_points, 0)
        else:
            distributed_reduced, argmax = torch_scatter.scatter_max(distributed, indices_long, dim=0)
            # distributed_reduced = torch_scatter.scatter_mean(distributed, indices_long, dim=0)

            #get also the nr of points in the lattice so the max pooled features can be different if there is 1 point then if there are 100
            ones=torch.cuda.FloatTensor( indices_long.shape[0] ).fill_(1.0)
            nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
            nr_points_per_simplex=nr_points_per_simplex.unsqueeze(1)
            #attempt 3 just by concatenating the barycentric coords
            # argmax_flatened=argmax.flatten()
            # argmax_positive=argmax_flatened.clone()
            # argmax_positive[argmax_flatened<0]=0
            barycentric_reduced=torch.index_select(barycentric_weights, 0, argmax.flatten()) #we select for each vertex the 64 barycentric weights that got selected by the scatter max
            # barycentric_reduced=torch.index_select(barycentric_weights, 0, argmax_positive ) #we select for each vertex the 64 barycentric weights that got selected by the scatter max
            barycentric_reduced=barycentric_reduced.view(argmax.shape[0], argmax.shape[1])
            distributed_reduced=torch.cat((distributed_reduced,barycentric_reduced),1)
            # distributed_reduced=torch.cat((distributed_reduced,barycentric_reduced, nr_points_per_simplex),1)
            # distributed_reduced=torch.cat((distributed_reduced, nr_points_per_simplex),1)

            minimum_points_per_simplex=4
            simplexes_with_few_points=nr_points_per_simplex<minimum_points_per_simplex
            distributed_reduced.masked_fill_(simplexes_with_few_points, 0)
            # print("nr of simeplexes which have very low number of points ", simplexes_with_few_points.sum())





        if self.with_debug_output:
            print("distributed_reduced before the last layer has shape ", distributed_reduced.shape)
        distributed_reduced[0,:]=0 #the first layers corresponds to the invalid points, the ones that had an index of -1. We set it to 0 so it doesnt affect the prediction or the batchnorm


        lattice_py.set_values(distributed_reduced)
        lattice_py.set_val_dim(distributed_reduced.shape[1])

       # #bn-relu-conv
        # if distributed_reduced.shape[1] is not self.nr_outputs_last_layer:
            # distributed_reduced, lattice_py= self.last_norm(distributed_reduced, lattice_py)
            # distributed_reduced=self.relu(distributed_reduced)
            # distributed_reduced=gelu(distributed_reduced)
            # distributed_reduced=self.last_linear(distributed_reduced)
        distributed_reduced, lattice_py=self.last_conv(distributed_reduced, lattice_py)
        # distributed_reduced=gelu(distributed_reduced)

        # ones=torch.zeros(distributed.shape[0], 1, device="cuda")
        # nr_points_per_vertex = torch_scatter.scatter_add(ones, indices_long, dim=0)
        # distributed_reduced=torch.cat((distributed_reduced,nr_points_per_vertex),1)

        # print("distributed reduced at the finale is, ", distributed_reduced)
        if self.with_debug_output:
            print("distributed_reduced at the finale is shape ", distributed_reduced.shape)

        


    

        # print("distributed_reduced has shape ", distributed_reduced.shape)
        lattice_py.set_values(distributed_reduced)
        lattice_py.set_val_dim(distributed_reduced.shape[1])

        # return lattice_reduced
        return distributed_reduced, lattice_py



class PointNetDenseModule(torch.nn.Module):
    def __init__(self, growth_rate, nr_layers, nr_outputs_last_layer, with_debug_output, with_error_checking):
        super(PointNetDenseModule, self).__init__()
        self.first_time=True
        self.growth_rate=growth_rate
        self.nr_layers=nr_layers
        self.layers=torch.nn.ModuleList([])
        self.norm_layers=torch.nn.ModuleList([])
        self.dropout_layers=torch.nn.ModuleList([])
        self.nr_outputs_last_layer=nr_outputs_last_layer
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.relu=torch.nn.ReLU(inplace=True)
        self.tanh=torch.nn.Tanh()
        self.nr_before_called=0
        self.nr_after_called=0
    def forward(self, lattice_py, distributed, indices):
        if (self.first_time):
            self.first_time=False

            #get the nr of channels of the distributed tensor
            nr_input_channels=distributed.shape[1]

            for i in range(self.nr_layers):
                if self.with_debug_output:
                    print ("in ", nr_input_channels)
                    print ("out ", self.growth_rate)
                is_last_layer=i==self.nr_layers-1 #the last layer is folowed by scatter max and not a batch norm therefore it needs a bias
                # print("is last layer is", is_last_layer)
                # self.norm_layers.append( GroupNormLatticeModule(nr_input_channels)  ) 
                self.layers.append( torch.nn.Linear(nr_input_channels, self.growth_rate, bias=is_last_layer).to("cuda")  )
                self.norm_layers.append( GroupNormLatticeModule(self.growth_rate)  ) 
                nr_input_channels+=self.growth_rate


            self.last_norm=GroupNormLatticeModule(  (distributed.shape[1] + self.nr_layers*self.growth_rate)*1 )
            self.last_linear=torch.nn.Linear((distributed.shape[1] + self.nr_layers*self.growth_rate)*1, self.nr_outputs_last_layer, bias=False).to("cuda") 
            # self.last_norm=GroupNormLatticeModule(  (distributed.shape[1] + self.nr_layers*self.growth_rate)*2 )
            # self.last_linear=torch.nn.Linear((distributed.shape[1] + self.nr_layers*self.growth_rate)*2, self.nr_outputs_last_layer, bias=False).to("cuda") 

        # print("pointnet distributed at the beggining is ", distributed.shape)


        #run the distributed through all the layers
        for i in range(len(self.layers)): 

            prev_features=distributed


            # logger_mean_before.log(self.nr_before_called, distributed.var().item(), name='mean_before')
            # self.nr_before_called+=1

            # #THE WRONG WAY OF DOING IT BECAUSE THE INPUT GETS NORMALIZED 
            # #This doesnt seem to be that bad after all, the other way (the correct way), is slower as we need at least two layers in otder to gain any benefit from tanh and is also less accurate forsome reason
            # distributed, lattice_py=self.norm_layers[i] (distributed, lattice_py)
            # # logger_mean_after.log(self.nr_after_called, distributed.var().item(), name='mean_after')
            # # self.nr_after_called+=1
            # distributed=self.tanh(distributed) #tanh seems better than relu but not by much
            # distributed=self.layers[i] (distributed)

            #


            #the new way of doing it in which we start first with a conv, because the first distributed will be just the input to the network
            distributed=self.layers[i] (distributed)
            if( i < len(self.layers)-1): #last tanh before the maxing need not be applied because it actually hurts the performance, also it's not used in the original pointnet https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
                #last bn need not be applied because we will max over the lattices either way and then to a bn afterwards
                distributed, lattice_py=self.norm_layers[i] (distributed, lattice_py)
                distributed=self.relu(distributed) #tanh seems better than relu but not by much


            distributed=torch.cat((prev_features, distributed),1)


        # print("pointnet distributed at the end of all the first batch of linear layers is ", distributed.shape)

        indices_long=indices.long()
        #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long<0]=0


        # #agreagate through max all the position that would fall onto the same latice vertex
        TIME_START("scatter_max")
        distributed_reduced, argmax = torch_scatter.scatter_max(distributed, indices_long, dim=0)
        # distributed_reduced_min, argmin = torch_scatter.scatter_min(distributed, indices_long, dim=0)
        # distributed_reduced=torch.cat((distributed_reduced_min, distributed_reduced),1)

        # distributed_reduced = torch_scatter.scatter_mean(distributed, indices_long, dim=0 )
        # distributed_reduced = torch_scatter.scatter_std(distributed, indices_long, dim=0 )
        #Actually the max over the mean positions will give the network some knowledge of the surface orientation. For planar surfaces, some of the coordinate will have a mean of zero and therefore the max will also be zero, giving the network a good estimate of the oritnetation


        #if we do a scatter max over the concatenated tensor, the first part of the tensor only contains the positions, so therefore we will always choose the furthest point..
        #ACTUALLY THIS DOESNT MAKE MUCH SENSE because the distributed is already containing the mean positions, no readon to mean it again...
        #This works but is quite slow and doesn't yield much a of an increase in iou. I imagine the std to be quite slow because it requires sqrt and powers and divisions. Max operation is a ton faster
        # distributed_reduced_std = torch_scatter.scatter_std(distributed[:,:3], indices_long, dim=0 )
        # distributed_reduced, argmax = torch_scatter.scatter_max(distributed[:,3:], indices_long, dim=0)
        # distributed_reduced=torch.cat((distributed_reduced_std, distributed_reduced),1)


        TIME_END("scatter_max")
        # print("distributed reduced before the last layer is, ", distributed_reduced)
        if self.with_debug_output:
            print("distributed_reduced before the last layer has shape ", distributed_reduced.shape)
        # distributed_reduced=self.last_dropout(distributed_reduced)
        distributed_reduced[0,:]=0 #the first layers corresponds to the invalid points, the ones that had an index of -1. We set it to 0 so it doesnt affect the prediction or the batchnorm



        distributed_reduced, lattice_py= self.last_norm(distributed_reduced, lattice_py)
        distributed_reduced=self.relu(distributed_reduced)
        # distributed_reduced=self.tanh(distributed_reduced)
        distributed_reduced=self.last_linear(distributed_reduced)
        if self.with_debug_output:
            print("distributed_reduced at the finale is shape ", distributed_reduced.shape)


    

        # print("distributed_reduced has shape ", distributed_reduced.shape)
        lattice_py.set_values(distributed_reduced)
        lattice_py.set_val_dim(distributed_reduced.shape[1])

        # return lattice_reduced
        return distributed_reduced, lattice_py




        # return distributed_reduced



#clone module that started as convolution and slowly devolved into just clone 
class CloneLatticeModule(torch.nn.Module):
    def __init__(self):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(CloneLatticeModule, self).__init__()
        self.weight = torch.nn.Parameter( torch.empty( 1 ).to("cuda") ) #works for ConvIm2RowLattice

    def forward(self, lattice_py):
        return CloneLattice.apply(lattice_py, self.weight)

class GnRelu(torch.nn.Module):
    def __init__(self, with_debug_output, with_error_checking):
        super(GnRelu, self).__init__()
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=self.relu(lv)
        ls.set_values(lv)
        return lv, ls

class GnRelu1x1(torch.nn.Module):
    def __init__(self, out_channels, bias, with_debug_output, with_error_checking):
        super(GnRelu1x1, self).__init__()
        self.out_channels=out_channels
        # self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_homogeneous_coord=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear=None
        self.use_bias=bias
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
            self.linear= torch.nn.Linear(lv.shape[1], self.out_channels, bias=self.use_bias).to("cuda") 
            with torch.no_grad():
                #https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
                torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
                # n = 1*self.out_channels
                # self.linear.weight.data.normal_(0, np.sqrt(2. / n))
                # if self.linear.bias is not None:
                #     torch.nn.init.zeros_(self.linear.bias)
        lv, ls=self.norm(lv,ls)
        lv=self.relu(lv)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls

class GnGelu1x1(torch.nn.Module):
    def __init__(self, out_channels, bias, with_debug_output, with_error_checking):
        super(GnGelu1x1, self).__init__()
        self.out_channels=out_channels
        # self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_homogeneous_coord=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear=None
        self.use_bias=bias
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
            self.linear= torch.nn.Linear(lv.shape[1], self.out_channels, bias=self.use_bias).to("cuda") 
            with torch.no_grad():
                torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
                # n = 1*self.out_channels
                # self.linear.weight.data.normal_(0, np.sqrt(2. / n))
                # if self.linear.bias is not None:
                #     torch.nn.init.zeros_(self.linear.bias)
        lv, ls=self.norm(lv,ls)
        # lv=self.relu(lv)
        lv=F.gelu(lv)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls

class GnTanh1x1(torch.nn.Module):
    def __init__(self, out_channels, bias, with_debug_output, with_error_checking):
        super(GnTanh1x1, self).__init__()
        self.out_channels=out_channels
        # self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_homogeneous_coord=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.tanh = torch.nn.Tanh()
        self.linear=None
        self.use_bias=bias
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
            self.linear= torch.nn.Linear(lv.shape[1], self.out_channels, bias=self.use_bias).to("cuda") 
            with torch.no_grad():
                torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='tanh')
                # n = 1*self.out_channels
                # self.linear.weight.data.normal_(0, np.sqrt(2. / n))
                # if self.linear.bias is not None:
                #     torch.nn.init.zeros_(self.linear.bias)
        lv, ls=self.norm(lv,ls)
        # lv=self.relu(lv)
        lv=self.tanh(lv)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls


class Gn1x1Gelu(torch.nn.Module):
    def __init__(self, out_channels, bias, with_debug_output, with_error_checking):
        super(Gn1x1Gelu, self).__init__()
        self.out_channels=out_channels
        # self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_homogeneous_coord=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear=None
        self.use_bias=bias
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
            self.linear= torch.nn.Linear(lv.shape[1], self.out_channels, bias=self.use_bias).to("cuda") 
            with torch.no_grad():
                torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
                # n = 1*self.out_channels
                # self.linear.weight.data.normal_(0, np.sqrt(2. / n))
                # if self.linear.bias is not None:
                #     torch.nn.init.zeros_(self.linear.bias)
        lv, ls=self.norm(lv,ls)
        # lv=self.relu(lv)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        lv=F.gelu(lv)
        ls.set_values(lv)
        return lv, ls

class Gn1x1(torch.nn.Module):
    def __init__(self, out_channels, bias, with_debug_output, with_error_checking):
        super(Gn1x1, self).__init__()
        self.out_channels=out_channels
        # self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_homogeneous_coord=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.linear=None
        self.use_bias=bias
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
            self.linear= torch.nn.Linear(lv.shape[1], self.out_channels, bias=self.use_bias).to("cuda") 
            with torch.no_grad():
                torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
                # n = 1*self.out_channels
                # self.linear.weight.data.normal_(0, np.sqrt(2. / n))
                # if self.linear.bias is not None:
                    # torch.nn.init.zeros_(self.linear.bias)
        lv, ls=self.norm(lv,ls)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls

class Gn(torch.nn.Module):
    def __init__(self, with_debug_output, with_error_checking):
        super(Gn, self).__init__()
        self.norm= None
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        ls.set_values(lv)
        return lv, ls

class Conv1x1(torch.nn.Module):
    def __init__(self, out_channels, bias, with_debug_output, with_error_checking):
        super(Conv1x1, self).__init__()
        self.out_channels=out_channels
        # self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_homogeneous_coord=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.linear=None
        self.use_bias=bias
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.linear is None:
            self.linear= torch.nn.Linear(lv.shape[1], self.out_channels, bias=self.use_bias).to("cuda") 
            with torch.no_grad():
                torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
                # n = 1*self.out_channels
                # self.linear.weight.data.normal_(0, np.sqrt(2. / n))
                # if self.linear.bias is not None:
                #     torch.nn.init.zeros_(self.linear.bias)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls


class GnReluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout, with_debug_output, with_error_checking):
        super(GnReluConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
        # self.relu = torch.nn.ReLU()
    def forward(self, lv, ls, skip_connection=None):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=self.relu(lv)
        # lv=gelu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        # print("print before conv lv is", lv.shape)
        lv_1, ls_1 = self.conv(lv, ls)
        # print("print after conv lv is", lv_1.shape)
        if skip_connection is not None:
            lv_1+=skip_connection
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class GnGeluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout, with_debug_output, with_error_checking):
        super(GnGeluConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
    def forward(self, lv, ls, skip_connection=None):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=F.gelu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        # print("print before conv lv is", lv.shape)
        lv_1, ls_1 = self.conv(lv, ls)
        # print("print after conv lv is", lv_1.shape)
        if skip_connection is not None:
            lv_1+=skip_connection
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class GnConvGelu(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout, with_debug_output, with_error_checking):
        super(GnConvGelu, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias,  with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
    def forward(self, lv, ls, skip_connection=None):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)

        # drop here so that it doesnt affect the group norm 
        if self.with_dropout:
            lv = self.drop(lv)

        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)

        # if self.with_dropout:
        #     lv_1 = self.drop(lv_1)

        lv_1=F.gelu(lv_1)

        # if self.with_dropout:
        #     lv_1 = self.drop(lv_1)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class GnConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout, with_debug_output, with_error_checking):
        super(GnConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
    def forward(self, lv, ls, skip_connection=None):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)

        return lv_1, ls_1

class BnReluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_debug_output, with_error_checking):
        super(BnReluConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= None
        self.relu = torch.nn.ReLU(inplace=True)
        # self.relu = torch.nn.ReLU()
    def forward(self, lv, ls, skip_connection=None):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv.shape[1])
        lv, ls=self.bn(lv,ls)
        lv=self.relu(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        if skip_connection is not None:
            lv_1+=skip_connection
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class BnConvRelu(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_debug_output, with_error_checking):
        super(BnConvRelu, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= None
        # self.relu = torch.nn.ReLU(inplace=True)
        self.relu = torch.nn.ReLU()
        self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
        # self.dropout=DropoutLattice(0.2)
    def forward(self, lv, ls, skip_connection=None):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv.shape[1])
        lv, ls=self.bn(lv,ls)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        lv_1=self.relu(lv_1)
        if skip_connection is not None:
            lv_1+=skip_connection
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class ConvReluBn(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_debug_output, with_error_checking):
        super(ConvReluBn, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= BatchNormLatticeModule(nr_filters)
        self.relu = torch.nn.ReLU()
        self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
    def forward(self, lv, ls, skip_connection=None):

        lv_1, ls_1 = self.conv(lv, ls)
        lv_1=self.relu(lv_1)
        # lv_1=self.leaky(lv_1)
        # if skip_connection is not None:
            # lv_1+=skip_connection
        ls_1.set_values(lv_1)
        lv_1, ls_1=self.bn(lv_1,ls_1)
        if skip_connection is not None:
            lv_1+=skip_connection

        # #try a concat instead of a sum
        # if skip_connection is not None:
        #     lv_1_concat=torch.cat((lv_1,skip_connection),1)
        #     ls_1.set_values(lv_1_concat)

        # #another type of concat
        # lv_1_concat=torch.cat((lv_1,lv),1)
        # ls_1.set_values(lv_1_concat)




        ls_1.set_values(lv_1)

        return lv_1, ls_1



class GnCoarsen(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnCoarsen, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        # self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
        # self.drop=DropoutLattice(0.2)
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        # lv=self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1

class GnReluCoarsen(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnReluCoarsen, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
        # self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
        # self.drop=DropoutLattice(0.2)
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=self.relu(lv)
        # lv=self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1

class GnGeluCoarsen(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnGeluCoarsen, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=F.gelu(lv)
        # lv=self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1


class GnCoarsenGelu(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnCoarsenGelu, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        # lv=self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        lv_1=F.gelu(lv_1)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1

class BnReluCoarsen(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(BnReluCoarsen, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= None
        self.relu = torch.nn.ReLU(inplace=True)
        # self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv.shape[1])
        lv, ls=self.bn(lv,ls)
        lv=self.relu(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1

class BnCoarsenRelu(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(BnCoarsenRelu, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= None
        self.relu = torch.nn.ReLU(inplace=True)
        # self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv.shape[1])
        lv, ls=self.bn(lv,ls)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        lv_1=self.relu(lv_1)
        # lv_1=self.dropout(lv_1)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1

class CoarsenReluBn(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(CoarsenReluBn, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= BatchNormLatticeModule(nr_filters)
        self.relu = torch.nn.ReLU()
        self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
    def forward(self, lv, ls):

        lv_1, ls_1 = self.coarse(lv, ls)
        lv_1=self.relu(lv_1)
        # lv_1=self.leaky(lv_1)
        lv_1, ls_1=self.bn(lv_1,ls_1)
        ls_1.set_values(lv_1)

        return lv_1, ls_1



class GnReluFinefy(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnReluFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse=self.norm(lv_coarse,ls_coarse)
        lv_coarse=self.relu(lv_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class GnGeluFinefy(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnGeluFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
        # self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse=self.norm(lv_coarse,ls_coarse)
        # lv_coarse=self.relu(lv_coarse)
        lv_coarse=F.gelu(lv_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class BnReluFinefy(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(BnReluFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= None
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        #similar to densenet and resnet: bn, relu, conv
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse=self.bn(lv_coarse,ls_coarse)
        lv_coarse=self.relu(lv_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class BnFinefyRelu(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(BnFinefyRelu, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= None
        self.relu = torch.nn.ReLU(inplace=True)
        self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        #similar to densenet and resnet: bn, relu, conv
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse=self.bn(lv_coarse,ls_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        lv_1=self.relu(lv_1)
        # lv_1=self.dropout(lv_1)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class FinefyReluBn(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(FinefyReluBn, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= BatchNormLatticeModule(nr_filters)
        self.relu = torch.nn.ReLU()
        self.leaky= torch.nn.LeakyReLU(negative_slope=0.2)
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        lv_1=self.relu(lv_1)
        # lv_1=self.leaky(lv_1)
        lv_1, ls_1=self.bn(lv_1,ls_1)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

#A relu would destroy too much of the gradients and sometimes that is undesirable
class GnFinefy(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse=self.norm(lv_coarse,ls_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class GnFinefyGelu(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnFinefyGelu, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.norm= None
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse=self.norm(lv_coarse,ls_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        lv_1=F.gelu(lv_1)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class BnFinefy(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(BnFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.bn= None
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        #similar to densenet and resnet: bn, relu, conv
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse=self.bn(lv_coarse,ls_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class GnReluExpandMax(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnReluExpandMax, self).__init__()
        self.nr_filters=nr_filters
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.gn_relu_1x1=None
        self.coarsen=CoarsenMaxLatticeModule(with_debug_output=with_debug_output, with_error_checking=with_error_checking)
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.gn_relu_1x1 is None:
            self.gn_relu_1x1=GnRelu1x1(self.nr_filters,False, self.with_debug_output, self.with_error_checking)

        lv_e, ls_e = self.gn_relu_1x1(lv, ls)
        print("GN RELU EXPAND MAX, we expanded to a vla full dim of ", lv_e.shape[1])
        print("GN RELU EXPAND MAX, val full dim is indeed ", ls_e.val_full_dim() )
        lv_1, ls_1 = self.coarsen(lv_e,ls_e)
        return lv_1, ls_1

class GnReluExpandAvg(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnReluExpandAvg, self).__init__()
        self.nr_filters=nr_filters
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.gn_relu_1x1=None
        self.coarsen=CoarsenAvgLatticeModule(with_debug_output=with_debug_output, with_error_checking=with_error_checking)
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.gn_relu_1x1 is None:
            # self.gn_relu_1x1=GnRelu1x1(self.nr_filters,False, self.with_debug_output, self.with_error_checking)
            self.gn_relu_1x1=GnRelu1x1(self.nr_filters,False, self.with_debug_output, self.with_error_checking)

        lv_e, ls_e = self.gn_relu_1x1(lv, ls)
        print("GN RELU EXPAND MAX, we expanded to a vla full dim of ", lv_e.shape[1])
        print("GN RELU EXPAND MAX, val full dim is indeed ", ls_e.val_full_dim() )
        lv_1, ls_1 = self.coarsen(lv_e,ls_e)
        return lv_1, ls_1

class GnReluExpandBlur(torch.nn.Module):
    def __init__(self, nr_filters, with_debug_output, with_error_checking):
        super(GnReluExpandBlur, self).__init__()
        self.nr_filters=nr_filters
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.gn_relu_1x1=None
        self.coarsen=CoarsenBlurLatticeModule(with_debug_output=with_debug_output, with_error_checking=with_error_checking)
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.gn_relu_1x1 is None:
            # self.gn_relu_1x1=GnRelu1x1(self.nr_filters,False, self.with_debug_output, self.with_error_checking)
            self.gn_relu_1x1=GnRelu1x1(self.nr_filters,False, self.with_debug_output, self.with_error_checking)

        lv_e, ls_e = self.gn_relu_1x1(lv, ls)
        print("GN RELU EXPAND MAX, we expanded to a vla full dim of ", lv_e.shape[1])
        print("GN RELU EXPAND MAX, val full dim is indeed ", ls_e.val_full_dim() )
        lv_1, ls_1 = self.coarsen(lv_e,ls_e)
        return lv_1, ls_1



#similar to https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResnetBlock(torch.nn.Module):

    def __init__(self, nr_filters, dilations, biases, with_dropout, with_debug_output, with_error_checking):
        super(ResnetBlock, self).__init__()
        # self.nr_filters=nr_filters
        # self.conv1=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilations[0], with_homogeneous_coord=False)
        # self.conv2=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilations[1], with_homogeneous_coord=False)
        # self.bn1 = BatchNormLatticeModule(nr_filters)
        # self.bn2 = BatchNormLatticeModule(nr_filters)
        # self.relu = torch.nn.ReLU(inplace=True)
        # self.relu = torch.nn.ReLU()
        # self.skip_translation=None # the input to this resnet block may not have the same nr of channels as the nr_filters, in which case we need a linear layer to increase the dimensionality
        # torch.nn.Linear(nr_input_channels, nr_output_channels, bias=True).to("cuda") 

        #again with bn-relu-conv
        self.conv1=GnReluConv(nr_filters, dilations[0], biases[0], with_dropout=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.conv2=GnReluConv(nr_filters, dilations[1], biases[1], with_dropout=with_dropout, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        # self.gate  = torch.nn.Parameter( torch.ones( 1, nr_filters ).to("cuda") ) #gate for the skip connection https://openreview.net/pdf?id=Sywh5KYex
        # self.residual_gate  = torch.nn.Parameter( torch.ones( 1,1 ).to("cuda") ) #gate for the skip connection https://openreview.net/pdf?id=Sywh5KYex

        #does gn-conv-gelu
        # self.conv1=GnConvGelu(nr_filters, dilations[0], biases[0], with_dropout=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        # self.conv2=GnConvGelu(nr_filters, dilations[1], biases[1], with_dropout=with_dropout, with_debug_output=with_debug_output, with_error_checking=with_error_checking)

        # self.drop=None
        # if with_dropout:
            # self.drop=DropoutLattice(0.2)

    def forward(self, lv, ls):
        # if(self.skip_translation==None and  not lattice_values.shape[1]==self.nr_filters):
            # self.skip_translation=torch.nn.Linear(lattice_values.shape[1], self.nr_filters, bias=True).to("cuda") 

        # #-------------ORIGINAL RESNET ARCHITECTUE from the paper
        # identity = lattice_values
        # #do a increase in dimensionality if neccesary
        # # if(not lattice_values.shape[1]==self.nr_filters):
        #     # identity=self.skip_translation(identity)

        # out, lattice_structure = self.conv1(lattice_values, lattice_structure)
        # out, lattice_structure = self.bn1(out, lattice_structure)
        # out=self.relu(out)

        # out, lattice_structure = self.conv2(out, lattice_structure)
        # out, lattice_structure = self.bn2(out, lattice_structure)

        # if(lattice_values.shape[1]==self.nr_filters):
        #     out += identity
        # out = self.relu(out)
        # lattice_structure.set_values(out)

        # return out, lattice_structure
        # #----------FINISH ORIGINAL RESNET ARCHITECTUE

        # #------------RESNET ARCHITECTURE MADE BY ME as i've hear that bn after the relu is better-
        # identity = lattice_values
        # #do a increase in dimensionality if neccesary
        # if(not lattice_values.shape[1]==self.nr_filters):
        #     identity=self.skip_translation(identity)

        # out, lattice_structure = self.conv1(lattice_values, lattice_structure)
        # out=self.relu(out)
        # out, lattice_structure = self.bn1(out, lattice_structure)

        # out, lattice_structure = self.conv2(out, lattice_structure)
        # out = self.relu(out)

        # out, lattice_structure = self.bn2(out, lattice_structure)
        # out += identity

        # return out, lattice_structure
        # #------------FINISH RESNET ARCHITECTURE MADE BY ME as i've hear that bn after the relu is better-



        # #new way to doing it, my way
        # identity = lattice_values
        # #do a increase in dimensionality if neccesary
        # if(not lattice_values.shape[1]==self.nr_filters):
        #     identity=self.skip_translation(identity)

        # out, lattice_structure = self.bn1(lattice_values, lattice_structure)
        # out, lattice_structure = self.conv1(out, lattice_structure)
        # out=self.relu(out)

        # out, lattice_structure = self.bn2(out, lattice_structure)
        # out, lattice_structure = self.conv2(out, lattice_structure)
        # out = self.relu(out)

        # out += identity

        # return out, lattice_structure

        # print("resiadual_gate is ", self.residual_gate)
        # print("gate has norm ", self.gate.norm())

        # if self.drop is not None:
            # lv=self.drop(lv)

        #bn-relu-conv
        identity=lv
        # print("identity has shape ", lv.shape)


        lv, ls=self.conv1(lv,ls)
        # print("after c1 lv has shape ", lv.shape)
        # if self.drop is not None:
            # lv=self.drop(lv)
            # lv = F.dropout(lv, p=0.2, training=self.training)
        lv, ls=self.conv2(lv,ls)
        # print("after c2 lv has shape ", lv.shape)
        # lv=lv*self.residual_gate
        # if(lv.shape[1]==identity.shape[1]):
        lv+=identity
        # lv = F.gelu(lv)
        ls.set_values(lv)
        return lv, ls

#inspired from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
class BottleneckBlock(torch.nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''

    def __init__(self, out_channels, biases, with_debug_output, with_error_checking):
        super(BottleneckBlock, self).__init__()
        self.downsample = 4
        self.contract=GnRelu1x1(int(out_channels/self.downsample), biases[0], with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.conv=GnReluConv(int(out_channels/self.downsample), 1, biases[1], with_dropout=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        self.expand=GnRelu1x1(out_channels, biases[2], with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        # self.residual_gate  = torch.nn.Parameter( torch.ones( 1,1 ).to("cuda") ) #gate for the skip connection https://openreview.net/pdf?id=Sywh5KYex

        # #does gn-conv-gelu
        # self.contract=Gn1x1Gelu(int(out_channels/self.downsample), biases[0], with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        # self.conv=GnConvGelu(int(out_channels/self.downsample), 1, biases[1], with_dropout=False, with_debug_output=with_debug_output, with_error_checking=with_error_checking)
        # self.expand=Gn1x1Gelu(out_channels, biases[2], with_debug_output=with_debug_output, with_error_checking=with_error_checking)

    def forward(self, lv, ls):
        identity=lv
        lv, ls=self.contract(lv,ls)
        lv, ls=self.conv(lv,ls)
        lv, ls=self.expand(lv,ls)
        # lv=lv*self.residual_gate
        lv+=identity
        # lv = F.gelu(lv)
        ls.set_values(lv)
        return lv, ls
      
      

#a bit of a naive implementation of densenet which is not very memory efficient. Idealy the storage should be shared as explained in the suplementary material of densenet
class DensenetBlock(torch.nn.Module):

    def __init__(self, nr_filters, dilation_list, nr_layers, with_debug_output, with_error_checking):
        super(DensenetBlock, self).__init__()
        self.nr_filters=nr_filters
        self.layers=torch.nn.ModuleList([])
        for i in range(nr_layers):
            self.layers.append( GnReluConv( nr_filters, dilation_list[i], with_debug_output, with_error_checking) )

    def forward(self, lv, ls):

        stack=lv
        output=[]

        for i in range(len(self.layers)):
            lv_new, ls = self.layers[i](stack, ls)
            stack=torch.cat((stack,lv_new),1)
            output.append(lv_new)

        output_concatenated=torch.cat(output,1)
        ls.set_values(output_concatenated)
        return output_concatenated, ls

