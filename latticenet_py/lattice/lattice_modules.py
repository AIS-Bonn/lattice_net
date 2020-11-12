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

        return lv_drop


class SplatLatticeModule(torch.nn.Module):
    def __init__(self):
        super(SplatLatticeModule, self).__init__()
    def forward(self, lattice_py, positions, values):
        return SplatLattice.apply(lattice_py, positions, values )

class DistributeLatticeModule(torch.nn.Module):
    def __init__(self, experiment):
        super(DistributeLatticeModule, self).__init__()
        self.experiment=experiment
    def forward(self, lattice_py, positions, values):
        return DistributeLattice.apply(lattice_py, positions, values, self.experiment )

# class DistributeCapLatticeModule(torch.nn.Module):
#     def __init__(self,):
#         super(DistributeCapLatticeModule, self).__init__()
#     def forward(self, distributed, nr_positions, ls, cap):
#         indices=ls.splatting_indices()
#         weights=ls.splatting_weights()
#         indices_long=indices.long()
#         #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
#         indices_long[indices_long<0]=0

#         ones=torch.ones(indices.shape[0]).to("cuda")


#         nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
#         nr_points_per_simplex=nr_points_per_simplex[1:] #the invalid simplex is at zero, the one in which we accumulate or the splatting indices that are -1

#         # mask = indices.ge(0.5).to("cuda")
#         # mask=torch.cuda.FloatTensor(indices.size(0)).uniform_() > 0.995
#         # mask=mask.unsqueeze(1)

#         # nr_positions= distributed.size(0)/(ls.pos_dim() +1  )
#         mask=ls.lattice.create_splatting_mask(nr_points_per_simplex.int(), int(nr_positions), cap )

#         # indices=indices.unsqueeze(1)
#         print("distributed ", distributed.shape)
#         print("indices ", indices.shape)
#         print("weights ", weights.shape)
#         print("mask ", mask.shape)

#         #print the tensors
#         print("mask is ", mask)
#         print("indices is ", indices)
#         print("weights is ", weights)
#         print("distributed is ", distributed)

#         capped_indices=torch.masked_select(indices, mask )
#         capped_weights=torch.masked_select(weights, mask )
#         capped_distributed=torch.masked_select(distributed, mask.unsqueeze(1))
#         capped_distributed=capped_distributed.view(-1,distributed.size(1))

#         print("capped_indices ", capped_indices.shape)
#         print("capped_distributed ", capped_distributed.shape)

#         # print("capped_indices is ", capped_indices)
#         # print("capped_weights is ", capped_weights)
#         # print("capped_distributed is ", capped_distributed)

#         ls.set_splatting_indices(capped_indices)
#         ls.set_splatting_weights(capped_weights)



#         return capped_distributed, capped_indices, ls



class DepthwiseConvLatticeModule(torch.nn.Module):
    def __init__(self, nr_filters, neighbourhood_size, dilation=1, bias=True ):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(DepthwiseConvLatticeModule, self).__init__()
        self.first_time=True
        self.weight=None
        self.bias=None
        self.neighbourhood_size=neighbourhood_size
        self.nr_filters=nr_filters
        self.dilation=dilation
        self.use_bias=bias
        self.use_center_vertex_from_lattice_neigbhours=False

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
            self.weight = torch.nn.Parameter( torch.empty( filter_extent,  val_dim  ).to("cuda") ) #works for ConvIm2RowLattice
            if self.use_bias:
                self.bias = torch.nn.Parameter( torch.empty( val_dim ).to("cuda") )
            # if(self.with_homogeneous_coord):
            #     self.bias = torch.nn.Parameter(torch.empty( self.nr_filters+1).to("cuda") )
            # else:
            #     self.bias = torch.nn.Parameter(torch.empty( self.nr_filters).to("cuda") )
            with torch.no_grad():
                self.reset_parameters(filter_extent)

        lv, ls=DepthwiseConv.apply(lattice_values, lattice_structure, self.weight, self.dilation, lattice_neighbours_values, lattice_neighbours_structure, self.use_center_vertex_from_lattice_neigbhours )
        if self.use_bias:
            lv+=self.bias
        ls.set_values(lv)
        
        return lv, ls

class ConvLatticeModule(torch.nn.Module):
    def __init__(self, nr_filters, neighbourhood_size, dilation=1, bias=True ):
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

    #as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self, filter_extent):
        # torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu') #pytorch uses default leaky relu but we use relu as here https://github.com/szagoruyko/binary-wide-resnet/blob/master/wrn_mcdonnell.py and as in here https://github.com/pytorch/vision/blob/19315e313511fead3597e23075552255d07fcb2a/torchvision/models/resnet.py#L156

        fan = torch.nn.init._calculate_correct_fan(self.weight, "fan_out")
        gain = torch.nn.init.calculate_gain("relu", 1)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        # print("reset params, self use_bias is", self.use_bias)
        if self.bias is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias, -bound, bound)


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
            with torch.no_grad():
                self.reset_parameters(filter_extent)

        lv, ls=ConvIm2RowLattice.apply(lattice_values, lattice_structure, self.weight, self.dilation, lattice_neighbours_values, lattice_neighbours_structure, self.use_center_vertex_from_lattice_neigbhours )
        if self.use_bias:
            lv+=self.bias
        ls.set_values(lv)
        
        return lv, ls




class CoarsenLatticeModule(torch.nn.Module):
    def __init__(self, nr_filters ):
        super(CoarsenLatticeModule, self).__init__()
        self.first_time=True
        self.nr_filters=nr_filters
        self.neighbourhood_size=1
        self.use_center_vertex_from_lattice_neigbhours=True

    #as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self, filter_extent):
        # torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')

        fan = torch.nn.init._calculate_correct_fan(self.weight, "fan_out")
        #the fan out here actually refers to the fan in because we don't have a tranposed weight as pytorch usually expects it
        #we reduce the fan because actually most of the vertices don't have any nieghbours
        #half of them have filter_extent =1 and the other ones have filter extent the usual one of 9. So we settle for the middle and go for 5
        fan=fan/2
        
        gain = torch.nn.init.calculate_gain("relu", 1)
        std = gain / math.sqrt(fan) *2.0
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
        

    def forward(self, lattice_fine_values, lattice_fine_structure):
        lattice_fine_structure.set_values(lattice_fine_values)

        if(self.first_time):
            self.first_time=False
            filter_extent=lattice_fine_structure.lattice.get_filter_extent(self.neighbourhood_size) 
            val_dim=lattice_fine_structure.lattice.val_dim()
            self.weight = torch.nn.Parameter( torch.empty( filter_extent * val_dim, self.nr_filters ).to("cuda") ) #works for ConvIm2RowLattice
            with torch.no_grad():
                self.reset_parameters(filter_extent)

        lv, ls= CoarsenLattice.apply(lattice_fine_values, lattice_fine_structure, self.weight, self.use_center_vertex_from_lattice_neigbhours ) #this just does a convolution, we also need batch norm an non linearity

        return lv, ls

class FinefyLatticeModule(torch.nn.Module):
    def __init__(self, nr_filters ):
        super(FinefyLatticeModule, self).__init__()
        self.first_time=True
        self.nr_filters=nr_filters
        self.neighbourhood_size=1
        self.use_center_vertex_from_lattice_neigbhours=True

    #as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self, filter_extent):
        # torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')

        fan = torch.nn.init._calculate_correct_fan(self.weight, "fan_out")
        #the fan out here actually refers to the fan in because we don't have a tranposed weight as pytorch usually expects it
        #we reduce the fan because actually most of the vertices don't have any nieghbours
        #half of them have filter_extent =1 and the other ones have filter extent the usual one of 9. So we settle for the middle and go for 5
        fan=fan/2
        

        gain = torch.nn.init.calculate_gain("relu", 1)
        std = gain / math.sqrt(fan) *2.0
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)


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

        lv, ls= FinefyLattice.apply(lattice_coarse_values, lattice_coarse_structure, lattice_fine_structure, self.weight, self.use_center_vertex_from_lattice_neigbhours ) #this just does a convolution, we also need batch norm an non linearity

        return lv, ls


class SliceLatticeModule(torch.nn.Module):
    def __init__(self ):
        super(SliceLatticeModule, self).__init__()
    def forward(self, lattice_values, lattice_structure, positions):

        lattice_structure.set_values(lattice_values)
        return SliceLattice.apply(lattice_values, lattice_structure, positions)

class GatherLatticeModule(torch.nn.Module):
    def __init__(self):
        super(GatherLatticeModule, self).__init__()
    def forward(self, lattice_values, lattice_structure, positions):

        lattice_structure.set_values(lattice_values)
        return GatherLattice.apply(lattice_values, lattice_structure, positions)




#the last attempt to make a fast slice without having a gigantic feature vector for each point. Rather from the features of the vertices, we regress directly the class probabilities
#the idea is to not do it with a gather but rather with a special slicing function that also gets as input some learnable weights
class SliceFastCUDALatticeModule(torch.nn.Module):
    def __init__(self, nr_classes, dropout_prob, experiment):
        super(SliceFastCUDALatticeModule, self).__init__()
        self.nr_classes=nr_classes
        self.bottleneck=None
        self.stepdown=torch.nn.ModuleList([])
        self.bottleneck_size=8
        self.norm_pre_gather=None
        self.linear_pre_deltaW=None 
        self.linear_deltaW=None 
        self.linear_clasify=None
        self.tanh=torch.nn.Tanh()
        self.dropout_prob=dropout_prob
        if(dropout_prob > 0.0):
            self.dropout =DropoutLattice(dropout_prob) 


        self.experiment=experiment
    def forward(self, lv, ls, positions):

        ls.set_values(lv)

        nr_positions=positions.shape[0]
        pos_dim=positions.shape[1]
        val_dim=lv.shape[1]

        #slowly reduce the features 
        if len(self.stepdown) is 0:
            for i in range(2):
                nr_channels_out= int( val_dim/np.power(2,i) ) 
                if nr_channels_out  < self.bottleneck_size:
                    sys.exit("We used to many linear layers an now the values are lower than the bottlenck size. Which means that the bottleneck would actually do an expansion...")
                print("adding stepdown with output of ", nr_channels_out)
                self.stepdown.append( GnRelu1x1(nr_channels_out , False)  )
                # self.stepdown.append( Gn1x1Gelu(nr_channels_out , False, self.with_debug_output, self.with_error_checking)  )
        if self.bottleneck is None:
            print("adding bottleneck with output of ", self.bottleneck_size)
            self.bottleneck=GnRelu1x1(self.bottleneck_size, False)            
        # apply the stepdowns
        for i in range(2):
            if i == 0:
                lv_bottleneck, ls_bottleneck = self.stepdown[i](lv, ls)
            else:
                lv_bottleneck, ls_bottleneck = self.stepdown[i](lv_bottleneck, ls_bottleneck)
        # last bottleneck
        lv_bottleneck, ls_bottleneck = self.bottleneck(lv_bottleneck, ls_bottleneck)

     

        sliced_bottleneck_rowified=GatherLattice.apply(lv_bottleneck, ls_bottleneck, positions)
       

        nr_vertices_per_simplex=ls.pos_dim()+1
        val_dim_of_each_vertex=int(sliced_bottleneck_rowified.shape[1]/ nr_vertices_per_simplex)

        #from this slice rowified we regress for each position some barycentric offest of size m_pos_dim+1
        # linear layers on the sliced rowified get us to a tensor of nr_positions x (m_pos_dim+1), this will be the weights offsets for each positions into the 4 lattice vertices
        if self.linear_deltaW is None:
            self.linear_deltaW=torch.nn.Linear( val_dim_of_each_vertex, 1, bias=True).to("cuda") 
            with torch.no_grad():
                torch.nn.init.kaiming_uniform_(self.linear_deltaW.weight, mode='fan_in', nonlinearity='tanh') 
                self.linear_deltaW.weight*=0.1 #make it smaller so that we start with delta weight that are close to zero
                torch.nn.init.zeros_(self.linear_deltaW.bias) 
            self.gamma  = torch.nn.Parameter( torch.ones( val_dim_of_each_vertex ).to("cuda") ) 
            self.beta  = torch.nn.Parameter( torch.zeros( val_dim_of_each_vertex ).to("cuda") ) 





        #attmept 4 predict a barycentric coordinate for each lattice vertex, and then use max over all the features in the simplex like in here https://arxiv.org/pdf/1611.04500.pdf
        sliced_bottleneck_rowified=sliced_bottleneck_rowified.view(nr_positions, nr_vertices_per_simplex, val_dim_of_each_vertex)
        #max over the al the vertices in a simplex
        max_vals,_=sliced_bottleneck_rowified.max(1)
        max_vals=max_vals.unsqueeze(1)
     

        sliced_bottleneck_rowified-= self.gamma* max_vals + self.beta #max vals broadcasts to all the vertices in the simplex and substracts the max from them
        delta_weights=self.linear_deltaW(sliced_bottleneck_rowified)
        delta_weights=delta_weights.reshape(nr_positions, nr_vertices_per_simplex)




        if self.experiment=="slice_no_deform":
            delta_weights*=0

        # print("delta weights is ", delta_weights)

        #ERROR FOR THE DELTAWEIGHTS
        # #the delta of the barycentric coordinates should sum to zero so that the sum of the normal barycentric coordinates should still sum to 1.0
        # sum_bar=delta_weights.sum(1)
        # # print("sum_bar is ", sum_bar)
        # diff=sum_bar-0.0 #deviation from the expected value
        # diff2=diff.mul(diff)
        # # diff2=diff.abs()
        # delta_weight_error_sum=diff2.mean()
        # delta_weight_error_sum=0.0

        

        #we slice with the delta weights and we clasify at the same time
        if self.linear_clasify is None: #create the linear clasify but we use the tensors directly inside our cuda kernel
            self.linear_clasify=torch.nn.Linear(val_dim, self.nr_classes, bias=True).to("cuda") 


        if(self.dropout_prob > 0.0):
            lv=self.dropout(lv)


        ls.set_values(lv)
      
        classes_logits = SliceClassifyLattice.apply(lv, ls, positions, delta_weights, self.linear_clasify.weight, self.linear_clasify.bias, self.nr_classes)

        return classes_logits


class BatchNormLatticeModule(torch.nn.Module):
    def __init__(self, nr_params, affine=True):
        super(BatchNormLatticeModule, self).__init__()
        self.bn=torch.nn.BatchNorm1d(num_features=nr_params, momentum=0.1, affine=affine).to("cuda")
    def forward(self,lattice_values, lattice_py):

        if(lattice_values.dim() is not 2):
            sys.exit("lattice should be 2 dimensional, nr_vertices x val_full_dim")
        
        lattice_values=self.bn(lattice_values)

        lattice_py.set_values(lattice_values)

        return lattice_values, lattice_py

class GroupNormLatticeModule(torch.nn.Module):
    def __init__(self, nr_params, affine=True):
        super(GroupNormLatticeModule, self).__init__()
        nr_groups=32
        #if the groups is not diivsalbe so for example if we have 80 params
        if nr_params%nr_groups!=0:
            nr_groups= int(nr_params/2)
        # if nr_params<=32:
            # nr_groups=int(nr_params/2)


        self.gn = torch.nn.GroupNorm(nr_groups, nr_params).to("cuda") #having 32 groups is the best as explained in the GroupNormalization paper
    def forward(self,lattice_values, lattice_py):

        if(lattice_values.dim() is not 2):
            sys.exit("lattice should be 2 dimensional, nr_vertices x val_dim")

        #group norm wants the tensor to be N, C, L  (nr_batches, channels, nr_samples)
        lattice_values=lattice_values.unsqueeze(0)
        lattice_values=lattice_values.transpose(1,2)
        # print("lattice values is ", lattice_values.shape)
        lattice_values=self.gn(lattice_values)
        lattice_values=lattice_values.transpose(1,2)
        lattice_values=lattice_values.squeeze(0)

        lattice_py.set_values(lattice_values)

        return lattice_values, lattice_py



class PointNetModule(torch.nn.Module):
    def __init__(self, nr_output_channels_per_layer, nr_outputs_last_layer, experiment):
        super(PointNetModule, self).__init__()
        self.first_time=True
        self.nr_output_channels_per_layer=nr_output_channels_per_layer
        self.nr_outputs_last_layer=nr_outputs_last_layer
        self.nr_linear_layers=len(self.nr_output_channels_per_layer)
        self.layers=torch.nn.ModuleList([])
        self.norm_layers=torch.nn.ModuleList([])
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
                    is_last_layer=i==len(self.nr_output_channels_per_layer)-1 #the last layer is folowed by scatter max and not a batch norm therefore it needs a bias
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
                    self.att_activ=GnRelu1x1(nr_input_channels, False)
                    self.att_scores=GnRelu1x1(nr_input_channels, True)


                self.last_conv=ConvLatticeModule(nr_filters=self.nr_outputs_last_layer, neighbourhood_size=1, dilation=1, bias=False) #disable the bias becuse it is followed by a gn



        barycentric_weights=distributed[:,-1]
        if ( self.experiment=="attention_pool"):
            distributed=distributed #when we use attention pool we use the distributed tensor that contains the barycentric weights
        else:
            distributed=distributed[:, :distributed.shape[1]-1] #IGNORE the barycentric weights for the moment and lift the coordinates of only the xyz and values

        # #run the distributed through all the layers
        experiment_that_imply_no_elevation=["pointnet_no_elevate", "pointnet_no_elevate_no_local_mean", "splat"]
        if self.experiment in experiment_that_imply_no_elevation:
            # print("not performing elevation by pointnet as the experiment is", self.experiment)
            pass
        else:
            for i in range(len(self.layers)): 

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



        indices_long=indices.long()

        #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long<0]=0



        if self.experiment=="splat":
            distributed_reduced = torch_scatter.scatter_mean(distributed, indices_long, dim=0)
        if self.experiment=="attention_pool":
            #attention pooling ##################################################
            #concat for each vertex the max over the simplex
            max_reduced, argmax = torch_scatter.scatter_max(distributed, indices_long, dim=0)
            max_per_vertex=torch.index_select(max_reduced, 0, indices_long)
            distributed_with_max=distributed+self.gamma*max_per_vertex

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

            #get also the nr of points in the lattice so the max pooled features can be different if there is 1 point then if there are 100
            ones=torch.cuda.FloatTensor( indices_long.shape[0] ).fill_(1.0)
            nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
            nr_points_per_simplex=nr_points_per_simplex.unsqueeze(1)
            minimum_points_per_simplex=4
            simplexes_with_few_points=nr_points_per_simplex<minimum_points_per_simplex
            distributed_reduced.masked_fill_(simplexes_with_few_points, 0)
        else:
            distributed_reduced, argmax = torch_scatter.scatter_max(distributed, indices_long, dim=0)

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


        distributed_reduced[0,:]=0 #the first layers corresponds to the invalid points, the ones that had an index of -1. We set it to 0 so it doesnt affect the prediction or the batchnorm


        lattice_py.set_values(distributed_reduced)
        lattice_py.set_val_dim(distributed_reduced.shape[1])

      
        distributed_reduced, lattice_py=self.last_conv(distributed_reduced, lattice_py)


        lattice_py.set_values(distributed_reduced)
        lattice_py.set_val_dim(distributed_reduced.shape[1])

        return distributed_reduced, lattice_py




class GnRelu1x1(torch.nn.Module):
    def __init__(self, out_channels, bias):
        super(GnRelu1x1, self).__init__()
        self.out_channels=out_channels
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

        lv, ls=self.norm(lv,ls)
        lv=self.relu(lv)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls

class GnGelu1x1(torch.nn.Module):
    def __init__(self, out_channels, bias):
        super(GnGelu1x1, self).__init__()
        self.out_channels=out_channels
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

        lv, ls=self.norm(lv,ls)
        # lv=self.relu(lv)
        lv=F.gelu(lv)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls



class Gn(torch.nn.Module):
    def __init__(self):
        super(Gn, self).__init__()
        self.norm= None
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        ls.set_values(lv)
        return lv, ls



class GnReluDepthwiseConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout):
        super(GnReluDepthwiseConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=DepthwiseConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
    def forward(self, lv, ls ):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=self.relu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnReluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout):
        super(GnReluConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
        # self.relu = torch.nn.ReLU()
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=self.relu(lv)
        # lv=gelu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class GnGeluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout):
        super(GnGeluConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias)
        self.norm= None
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=F.gelu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        ls_1.set_values(lv_1)

        return lv_1, ls_1

class BnReluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias):
        super(BnReluConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias)
        self.bn= None
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, lv, ls):

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv.shape[1])
        lv, ls=self.bn(lv,ls)
        lv=self.relu(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        ls_1.set_values(lv_1)

        return lv_1, ls_1



class GnCoarsen(torch.nn.Module):
    def __init__(self, nr_filters):
        super(GnCoarsen, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters)
        self.norm= None
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1

class GnReluCoarsen(torch.nn.Module):
    def __init__(self, nr_filters):
        super(GnReluCoarsen, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters)
        self.norm= None
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=self.relu(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1

class GnGeluCoarsen(torch.nn.Module):
    def __init__(self, nr_filters):
        super(GnGeluCoarsen, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters)
        self.norm= None
    def forward(self, lv, ls, concat_connection=None):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls=self.norm(lv,ls)
        lv=F.gelu(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1=torch.cat((lv_1, concat_connection),1)
            ls_1.set_values(lv_1)


        return lv_1, ls_1



class GnReluFinefy(torch.nn.Module):
    def __init__(self, nr_filters):
        super(GnReluFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters)
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
    def __init__(self, nr_filters):
        super(GnGeluFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters)
        self.norm= None
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        #similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse=self.norm(lv_coarse,ls_coarse)
        lv_coarse=F.gelu(lv_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnFinefy(torch.nn.Module):
    def __init__(self, nr_filters):
        super(GnFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters)
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




#similar to https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResnetBlock(torch.nn.Module):

    def __init__(self, nr_filters, dilations, biases, with_dropout):
        super(ResnetBlock, self).__init__()
        
        #again with bn-relu-conv
        self.conv1=GnReluConv(nr_filters, dilations[0], biases[0], with_dropout=False)
        self.conv2=GnReluConv(nr_filters, dilations[1], biases[1], with_dropout=with_dropout)

        # self.conv1=GnReluDepthwiseConv(nr_filters, dilations[0], biases[0], with_dropout=False)
        # self.conv2=GnReluDepthwiseConv(nr_filters, dilations[1], biases[1], with_dropout=with_dropout)

        # self.residual_gate  = torch.nn.Parameter( torch.ones( 1 ).to("cuda") ) #gate for the skip connection https://openreview.net/pdf?id=Sywh5KYex


    def forward(self, lv, ls):
      

        identity=lv


        lv, ls=self.conv1(lv,ls)
        lv, ls=self.conv2(lv,ls)
        # lv=lv*self.residual_gate
        lv+=identity
        ls.set_values(lv)
        return lv, ls

#inspired from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
class BottleneckBlock(torch.nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''

    def __init__(self, out_channels, biases):
        super(BottleneckBlock, self).__init__()
        self.downsample = 4
        self.contract=GnRelu1x1(int(out_channels/self.downsample), biases[0])
        self.conv=GnReluConv(int(out_channels/self.downsample), 1, biases[1], with_dropout=False)
        self.expand=GnRelu1x1(out_channels, biases[2])
        # self.residual_gate  = torch.nn.Parameter( torch.ones( 1 ).to("cuda") ) #gate for the skip connection https://openreview.net/pdf?id=Sywh5KYex

    def forward(self, lv, ls):
        identity=lv
        lv, ls=self.contract(lv,ls)
        lv, ls=self.conv(lv,ls)
        lv, ls=self.expand(lv,ls)
        # lv=lv*self.residual_gate
        lv+=identity
        ls.set_values(lv)
        return lv, ls
      
      

#a bit of a naive implementation of densenet which is not very memory efficient. Idealy the storage should be shared as explained in the suplementary material of densenet
class DensenetBlock(torch.nn.Module):

    def __init__(self, nr_filters, dilation_list, nr_layers):
        super(DensenetBlock, self).__init__()
        self.nr_filters=nr_filters
        self.layers=torch.nn.ModuleList([])
        for i in range(nr_layers):
            self.layers.append( GnReluConv( nr_filters, dilation_list[i]) )

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

