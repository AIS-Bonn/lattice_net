import torch
from torch.autograd import Function
from torch import Tensor

import sys
import os
# import warnings
from termcolor import colored
# http://wiki.ros.org/Packages#Client_Library_Support
# import rospkg
# rospack = rospkg.RosPack()
# sf_src_path=rospack.get_path('surfel_renderer')
# sf_build_path=os.path.abspath(sf_src_path + "/../../build/surfel_renderer")
# sys.path.append(sf_build_path) #contains the modules of pycom

from easypbr  import *
from latticenet  import *
from lattice.lattice_py import LatticePy
from lattice.lattice_funcs import *
from lattice.lattice_modules import *

from functools import reduce
from torch.nn.modules.module import _addindent


#

class LNN(torch.nn.Module):
    def __init__(self, nr_classes, model_params, with_debug_output, with_error_checking):
        super(LNN, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking

        #a bit more control
        self.model_params=model_params
        self.nr_downsamples=model_params.nr_downsamples()
        self.nr_blocks_down_stage=model_params.nr_blocks_down_stage()
        self.nr_blocks_bottleneck=model_params.nr_blocks_bottleneck()
        self.nr_blocks_up_stage=model_params.nr_blocks_up_stage()
        self.nr_levels_down_with_normal_resnet=model_params.nr_levels_down_with_normal_resnet()
        self.nr_levels_up_with_normal_resnet=model_params.nr_levels_up_with_normal_resnet()
        compression_factor=model_params.compression_factor()
        dropout_last_layer=model_params.dropout_last_layer()
        experiment=model_params.experiment()
        #check that the experiment has a valid string
        valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        if experiment not in valid_experiment:
            err = "Experiment " + experiment + " is not valid"
            sys.exit(err)





        self.distribute=DistributeLatticeModule(experiment, self.with_debug_output, self.with_error_checking) 
        self.distribute_cap=DistributeCapLatticeModule() 
        #self.distributed_transform=DistributedTransform( [32], self.with_debug_output, self.with_error_checking)  
        self.pointnet_layers=model_params.pointnet_layers()
        self.start_nr_filters=model_params.pointnet_start_nr_channels()
        print("pointnet layers is ", self.pointnet_layers)
        self.point_net=PointNetModule( self.pointnet_layers, self.start_nr_filters, experiment, self.with_debug_output, self.with_error_checking)  
        # self.start_nr_filters=model_params.pointnet_start_nr_channels()
        # self.point_net=PointNetDenseModule( growth_rate=16, nr_layers=2, nr_outputs_last_layer=self.start_nr_filters, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking) 

        # self.first_conv=ConvLatticeModule(nr_filters=self.start_nr_filters, neighbourhood_size=1, dilation=1, bias=True, with_homogeneous_coord=False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)




        #just pointnet
        # self.nr_downsamples=0
        # self.nr_blocks_down_stage=[]
        # self.nr_blocks_bottleneck=0
        # self.nr_blocks_up_stage=[]

        #####################
        # Downsampling path #
        #####################
        # self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        self.maxpool_list=torch.nn.ModuleList([])
        corsenings_channel_counts = []
        skip_connection_channel_counts = []
        cur_channels_count=self.start_nr_filters
        for i in range(self.nr_downsamples):
            
            #create the resnet blocks
            self.resnet_blocks_per_down_lvl_list.append( torch.nn.ModuleList([]) )
            # cur_channels_count=self.start_nr_filters*np.power(2,i)
            for j in range(self.nr_blocks_down_stage[i]):
                if i<self.nr_levels_down_with_normal_resnet:
                    print("adding down_resnet_block with nr of filters", cur_channels_count )
                    # should_use_dropout= i!=0 #we only use dropout fr the resnet blocks that are not in the first downsample. The first downsample doesnt have many parameters so dropout is not needed
                    # should_use_dropout= False #using dropout in the resnet blocks seems to make it worse. But we keep the dropout in the corsenig because that seems good
                    # should_use_dropout=True # TODO Dropout works best with shapenet but probably should be disabled for semantic kitti
                    should_use_dropout=False
                    print("adding down_resnet_block with dropout", should_use_dropout )
                    self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,False], should_use_dropout, self.with_debug_output, self.with_error_checking) )
                else:
                    print("adding down_bottleneck_block with nr of filters", cur_channels_count )
                    self.resnet_blocks_per_down_lvl_list[i].append( BottleneckBlock(cur_channels_count, [False,False,False], self.with_debug_output, self.with_error_checking) )
            skip_connection_channel_counts.append(cur_channels_count)
            nr_channels_after_coarsening=int(cur_channels_count*2*compression_factor)
            # nr_channels_after_coarsening=int(cur_channels_count)
            print("adding bnReluCorsen which outputs nr of channels ", nr_channels_after_coarsening )
            # self.maxpool_list.append( CoarsenMaxLatticeModule( self.with_debug_output, self.with_error_checking ) )
            # self.coarsens_list.append( GnReluExpandMax(nr_channels_after_coarsening, self.with_debug_output, self.with_error_checking)) #is actually the worse one...
            # self.coarsens_list.append( GnReluExpandAvg(nr_channels_after_coarsening, self.with_debug_output, self.with_error_checking))
            # self.coarsens_list.append( GnReluExpandBlur(nr_channels_after_coarsening, self.with_debug_output, self.with_error_checking)) 
            self.coarsens_list.append( GnReluCoarsen(nr_channels_after_coarsening, self.with_debug_output, self.with_error_checking)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            # self.coarsens_list.append( GnCoarsenGelu(nr_channels_after_coarsening, self.with_debug_output, self.with_error_checking)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            cur_channels_count=nr_channels_after_coarsening
            corsenings_channel_counts.append(cur_channels_count)

        #####################
        #     Bottleneck    #
        #####################
        self.resnet_blocks_bottleneck=torch.nn.ModuleList([])
        # nr_filters=self.start_nr_filters*np.power(2,self.nr_downsamples)
        # nr_filters=self.start_nr_filters*np.power(2,self.nr_downsamples)
        for j in range(self.nr_blocks_bottleneck):
                print("adding bottleneck_resnet_block with nr of filters", cur_channels_count )
                # self.resnet_blocks_bottleneck.append( ResnetBlock(cur_channels_count, [1,1], [False,False], True, self.with_debug_output, self.with_error_checking) )
                self.resnet_blocks_bottleneck.append( BottleneckBlock(cur_channels_count, [False,False,False], self.with_debug_output, self.with_error_checking) )

        self.upsampling_method="finefy" #finefy, slice_elevated, slice_elevated_deform
        # self.upsampling_method="slice_elevated" #finefy, slice_elevated, slice_elevated_deform
        # self.upsampling_method="slice_elevated_deform" #finefy, slice_elevated, slice_elevated_deform
        self.do_concat_for_vertical_connection=True
        #######################
        #   Upsampling path   #
        #######################
        self.finefy_list=torch.nn.ModuleList([])
        self.up_activation_list=torch.nn.ModuleList([])
        self.up_match_dim_list=torch.nn.ModuleList([])
        self.up_bn_match_dim_list=torch.nn.ModuleList([])
        self.resnet_blocks_per_up_lvl_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsamples):
            nr_chanels_skip_connection=skip_connection_channel_counts.pop()
            # nr_chanels_end_of_corsening=corsenings_channel_counts.pop()
            # print("nr_chanels_skip_connection ", nr_chanels_skip_connection)
            # print("nr_chanels_end_of_corsening ", nr_chanels_end_of_corsening)

            # if the finefy is the deepest one int the network then it just divides by 2 the nr of channels because we know it didnt get as input two concatet tensors
            nr_chanels_finefy=int(cur_channels_count/2)

            #do it with finefy
            if self.upsampling_method=="finefy":
                print("adding bnReluFinefy which outputs nr of channels ", nr_chanels_finefy )
                # self.finefy_list.append( BnReluFinefy(nr_chanels_skip_connection, self.with_debug_output, self.with_error_checking))
                # self.finefy_list.append( GnReluFinefy(nr_chanels_skip_connection, self.with_debug_output, self.with_error_checking))
                # seems that the relu in BnReluFinefy stops too much of the gradient from flowing up the network, altought we lose one non-linearity, a BnFinefy seems a lot more eneficial for the general flow of gradients as the network converges a lot faster
                # self.finefy_list.append( GnFinefy(nr_chanels_skip_connection, self.with_debug_output, self.with_error_checking))
                # self.finefy_list.append( GnFinefyGelu(nr_chanels_finefy, self.with_debug_output, self.with_error_checking))
                self.finefy_list.append( GnReluFinefy(nr_chanels_finefy, self.with_debug_output, self.with_error_checking))
                # self.finefy_list.append( GnGeluFinefy(nr_chanels_skip_connection, self.with_debug_output, self.with_error_checking))
                # self.finefy_list.append( FinefyLatticeModule(nr_chanels_skip_connection, self.with_debug_output, self.with_error_checking))
            elif self.upsampling_method=="slice_elevated":
                # sys.exit("Not implemented because we switched to the new groupnorm and this is still using batchnorm")
                # self.up_activation_list.append( Gn(self.with_debug_output, self.with_error_checking) ) #one cna aply the gn relu before the slice because it's more efficient. Slicing doesnt change the mean of hte tensor too much and relu gives the same results weather it's before or after
                self.finefy_list.append( SliceElevatedVertsLatticeModule(self.with_debug_output, self.with_error_checking)  )
                # self.finefy_list.append( SliceElevatedVertsLatticeModule(True, True)  )
                # self.up_match_dim_list.append( Conv1x1(nr_chanels_skip_connection,False, self.with_debug_output, self.with_error_checking) )
                self.up_match_dim_list.append( GnRelu1x1(nr_chanels_skip_connection,False, self.with_debug_output, self.with_error_checking) )
            elif self.upsampling_method=="slice_elevated_deform":
                # self.finefy_list.append( SliceDeformElevatedLatticeModule(self.with_debug_output, self.with_error_checking)  )
                self.finefy_list.append( SliceDeformElevatedLatticeModule(True, True)  )
                self.up_match_dim_list.append( Conv1x1(nr_chanels_skip_connection,False, self.with_debug_output, self.with_error_checking) )
               
            else:
                sys.exit("Upsampling methos is not known")

            #after finefy we do a concat with the skip connection so the number of channels doubles
            if self.do_concat_for_vertical_connection:
                cur_channels_count=nr_chanels_skip_connection+nr_chanels_finefy
            else:
                cur_channels_count=nr_chanels_skip_connection

            self.resnet_blocks_per_up_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_up_stage[i]):
                is_last_conv=j==self.nr_blocks_up_stage[i]-1 and i==self.nr_downsamples-1 #the last conv of the last upsample is followed by a slice and not a bn, therefore we need a bias
                # print("adding up_resnet_block with nr of filters", cur_channels_count ) 
                # self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,is_last_conv], self.with_debug_output, self.with_error_checking) )
                if i>=self.nr_downsamples-self.nr_levels_up_with_normal_resnet:
                    print("adding up_resnet_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,is_last_conv], False, self.with_debug_output, self.with_error_checking) )
                else:
                    print("adding up_bottleneck_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( BottleneckBlock(cur_channels_count, [False,False,is_last_conv], self.with_debug_output, self.with_error_checking) )

        # self.last_bottleneck=GnRelu1x1(self.start_nr_filters, True, self.with_debug_output, self.with_error_checking)

        # self.point_net_efficient=PointNetEfficientModule( growth_rate=8, nr_layers=3 , with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking) 
        # self.slice=SliceLatticeModule(with_homogeneous_coord=False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.slice_conv=SliceConvLatticeModule(nr_classes=nr_classes, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.slice_deform=SliceDeformLatticeModule(nr_classes=nr_classes, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.slice_deform_full=SliceDeformFullLatticeModule(nr_classes=nr_classes, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.slice_fast_pytorch=SliceFastPytorchLatticeModule(nr_classes=nr_classes, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.slice_fast_bottleneck_pytorch=SliceFastBottleneckPytorchLatticeModule(nr_classes=nr_classes, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.slice_fast_cuda=SliceFastCUDALatticeModule(nr_classes=nr_classes, dropout_prob=dropout_last_layer, experiment=experiment, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.slice_classify=SliceClassifyLatticeModule(nr_classes=nr_classes, dropout_prob=dropout_last_layer, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.stepdown = StepDownModule([], nr_classes, dropout_last_layer, self.with_debug_output, self.with_error_checking)
        #stepdown densenetmodule is too slow as it requires to copy the whole pointcloud when it concatenas. For semantic kitti this is just too much
        # self.stepdown = StepDownModuleDensenetNoBottleneck(16, 3, nr_classes, self.with_debug_output, self.with_error_checking)
       
        self.logsoftmax=torch.nn.LogSoftmax(dim=1)
        # self.softmax=torch.nn.Softmax(dim=2)


        if experiment!="none":
            warn="USING EXPERIMENT " + experiment
            # warnings.warn(warn)
            print(colored("-------------------------------", 'yellow'))
            print(colored(warn, 'yellow'))
            print(colored("-------------------------------", 'yellow'))

    def forward(self, ls, positions, values):

        #create lattice vertices and fill in the splatting_indices
        
        # TIME_START("create_verts")
        # ls=self.create_verts(ls,positions)
        # TIME_END("create_verts")

        #with coarsening block--------------------------------------------------
        # TIME_START("distribute_py")
        with torch.set_grad_enabled(False):
            distributed, indices=self.distribute(ls, positions, values)
        # TIME_END("distribute_py")
        # ls.compute_nr_points_per_lattice_vertex()
        # print( ls.nr_lattice_vertices() )
        # print("after distribute indices is ", ls.splatting_indices())

        # print("distributed has shape", distributed.shape)
        # print("indices has shape", indices.shape)
        #remove some rows of the distribured and indices depending if the corresponding lattice vertex has to many incident points
        # distributed, indices,ls=self.distribute_cap(distributed, positions.size(1), ls, cap=20)

        #transform
        # TIME_START("distribute_transform")
        # distributed, ls= self.distributed_transform(ls, distributed, indices)
        # TIME_END("distribute_transform")

        # TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        # TIME_END("pointnet+bn")

        # print("lv after pointnet has shape ", lv.shape)

        # lv, ls = self.first_conv(lv, ls)

        # ls.compute_nr_points_per_lattice_vertex()

        #create a whole thing with downsamples and all
        fine_structures_list=[]
        fine_values_list=[]
        # TIME_START("down_path")
        for i in range(self.nr_downsamples):
            # print("DOWNSAPLE ", i, " with input lv of shape ", lv.shape)

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 
                # print("after resnet indices is ", ls.splatting_indices())

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(lv)

            #now we do a downsample
            # print("down input shape ", lv.shape[1])
            # lv, ls =self.maxpool_list[i](lv,ls)
            lv, ls = self.coarsens_list[i] ( lv, ls)
            # print("after coarsen indices is ", ls.splatting_indices())
            # print("DOWNSAPLE ", i, " with out lv of shape ", lv.shape)

        # TIME_END("down_path")

        # #bottleneck
        # print("bottleneck input shape ", lv.shape[1])
        for j in range(self.nr_blocks_bottleneck):
            lv, ls = self.resnet_blocks_bottleneck[j] ( lv, ls) 


        #upsample (we start from the bottom of the u, so the upsampling that is closest to the blottlenck)
        # TIME_START("up_path")
        for i in range(self.nr_downsamples):
            # print("UPSAMPLE ", i)

            fine_values=fine_values_list.pop()
            fine_structure=fine_structures_list.pop()

            # if self.upsampling_method=="slice_elevated": #if we are not using finefy, we are just slicing, so now we can do now a gn-relu and a 1x1 conv after the slice
                # lv, ls = self.up_activation_list[i](lv, ls) 

            #finefy
            # print("finefy input shape", lv.shape)
            lv, ls = self.finefy_list[i] ( lv, ls, fine_structure  )
            # print("after finefy lv is " , lv.shape)


            if self.upsampling_method=="slice_elevated" or self.upsampling_method=="slice_elevated_deform": #if we are not using finefy, we are just slicing, so now we have to reduce the nr of channels
                lv, ls = self.up_match_dim_list[i](lv, ls) 

            #concat or adding for the vertical connection
            if self.do_concat_for_vertical_connection: 
                lv=torch.cat((lv, fine_values ),1)
            else:
                lv+=fine_values

       

            # print("after concating and up_mathing dim lv is " , lv.shape[1])

            #resnet blocks
            for j in range(self.nr_blocks_up_stage[i]):
                lv, ls = self.resnet_blocks_per_up_lvl_list[i][j] ( lv, ls) 
        # TIME_END("up_path")


        # print("print values before slice_conv has shape", lv.shape)

        #slicing is quite expensive, because we gather all the values around the simplex. so we try to reduce the nr of values per point 
        # lv, ls = self.last_bottleneck(lv, ls) #sligt regression from using it, Reaches only 75.4 on motoribke instead of 75.8
  
        # TIME_START("slice")
        # sv=self.slice(lv, ls, positions)
        # sv=self.slice_conv(lv, ls, positions)
        # sv=self.slice_deform(lv, ls, positions)
        # sv=self.slice_deform_full(lv, ls, positions)
        # sv=self.slice_fast_pytorch(lv, ls, positions, distributed)
        # sv=self.slice_fast_bottleneck_pytorch(lv, ls, positions, distributed)
        # print("just before calling slice_Fast_cuda in models.py indices is ", ls.splatting_indices())
        sv, delta_weight_error_sum=self.slice_fast_cuda(lv, ls, positions)
        # sv=self.slice_classify(lv, ls, positions)
        # TIME_END("slice")

        self.per_point_features=sv
        # TIME_START("stepdown")
        # s_final=self.stepdown(sv)
        # TIME_END("stepdown")


        logsoftmax=self.logsoftmax(sv)
        # softmax=self.softmax(sv)
        # logsoftmax=self.logsoftmax(s_final)

        # delta_weight_error_sum=torch.tensor(0).to("cuda")

        # logsoftmax=logsoftmax
        # softmax=softmax.squeeze(0)
        sv=sv.squeeze(0)

        return logsoftmax, sv, delta_weight_error_sum
        # return logsoftmax, s_final

    def prepare_cloud(self, cloud):
        # TIME_START("prepare")
        # distance_tensor=torch.from_numpy(cloud.D).unsqueeze(0).float().to("cuda")
        # positions_tensor=positions_tensor[:,:,0:2].clone() #get only the first 2 column because I want to debug some stuff with the coarsening of the lattice
        # print("prearing cloud with possitions tensor of shape", positions_tensor.shape)
        # values_tensor=torch.zeros(1, positions_tensor.shape[1], 1) #not really necessary but at the moment I have no way of passing an empty value array
        # values_tensor=positions_tensor #usualyl makes the things worse... it converges faster to a small loss but not as small as just setting the values to one
        # values_tensor=positions_tensor[:,:, 1].clone().unsqueeze(2) #just the height (so the y coordinate) #not this is shape. 1xnr_pointsx1

        #use xyz,distance as value, just as squeezeseg (reaches only 69 on the motorbike.)
        # values_tensor=torch.cat((positions_tensor,distance_tensor),2) #actually this works the best

        #use height above groundonly (still not super good for the bike but good for the knife)
        # values_tensor=positions_tensor[:,:, 1].clone().unsqueeze(2) #just the height (so the y coordinate) #not this is shape. 1xnr_pointsx1
        with torch.set_grad_enabled(False):

            if self.model_params.positions_mode()=="xyz":
                positions_tensor=torch.from_numpy(cloud.V).float().to("cuda")
            elif self.model_params.positions_mode()=="xyz+rgb":
                xyz_tensor=torch.from_numpy(cloud.V).float().to("cuda")
                rgb_tensor=torch.from_numpy(cloud.C).float().to("cuda")
                positions_tensor=torch.cat((xyz_tensor,rgb_tensor),1)
            elif self.model_params.positions_mode()=="xyz+intensity":
                xyz_tensor=torch.from_numpy(cloud.V).float().to("cuda")
                intensity_tensor=torch.from_numpy(cloud.I).float().to("cuda")
                positions_tensor=torch.cat((xyz_tensor,intensity_tensor),1)
            else:
                err="positions mode of ", self.model_params.positions_mode() , " not implemented"
                sys.exit(err)


            if self.model_params.values_mode()=="none":
                values_tensor=torch.zeros(positions_tensor.shape[0], 1) #not really necessary but at the moment I have no way of passing an empty value array
            elif self.model_params.values_mode()=="intensity":
                values_tensor=torch.from_numpy(cloud.I).float().to("cuda")
            elif self.model_params.values_mode()=="rgb":
                values_tensor=torch.from_numpy(cloud.C).float().to("cuda")
            elif self.model_params.values_mode()=="rgb+height":
                rgb_tensor=torch.from_numpy(cloud.C).float().to("cuda")
                height_tensor=torch.from_numpy(cloud.V[:,1]).unsqueeze(1).float().to("cuda")
                values_tensor=torch.cat((rgb_tensor,height_tensor),1)
            elif self.model_params.values_mode()=="rgb+xyz":
                rgb_tensor=torch.from_numpy(cloud.C).float().to("cuda")
                xyz_tensor=torch.from_numpy(cloud.V).float().to("cuda")
                values_tensor=torch.cat((rgb_tensor,xyz_tensor),1)
            elif self.model_params.values_mode()=="height":
                height_tensor=torch.from_numpy(cloud.V[:,1]).unsqueeze(1).float().to("cuda")
                values_tensor=height_tensor
            elif self.model_params.values_mode()=="xyz":
                xyz_tensor=torch.from_numpy(cloud.V).float().to("cuda")
                values_tensor=xyz_tensor
            else:
                err="values mode of ", self.model_params.values_mode() , " not implemented"
                sys.exit(err)


            target=cloud.L_gt
            target_tensor=torch.from_numpy(target).long().squeeze(1).to("cuda").squeeze(0)
        # print("maximum class idx is ", target_tensor.max() )
        # TIME_END("prepare")

        return positions_tensor, values_tensor, target_tensor

    #like in here https://github.com/drethage/fully-convolutional-point-network/blob/60b36e76c3f0cc0512216e9a54ef869dbc8067ac/data.py 
    #also the Enet paper seems to have a similar weighting
    def compute_class_weights(self, class_frequencies, background_idx):
        """ Computes class weights based on the inverse logarithm of a normalized frequency of class occurences.
        Args:
        class_counts: np.array
        Returns: list[float]
        """
        # class_counts /= np.sum(class_counts[0:self._empty_class_id])
        # class_weights = (1 / np.log(1.2 + class_counts))

        # class_weights[self._empty_class_id] = self._special_weights['empty']
        # class_weights[self._masked_class_id] = self._special_weights['masked']

        # return class_weights.tolist()


        #doing it my way but inspired by their approach of using the logarithm
        class_frequencies_tensor=torch.from_numpy(class_frequencies).float().to("cuda")
        class_weights = (1.0 / torch.log(1.05 + class_frequencies_tensor)) #the 1.2 says pretty much what is the maximum weight that we will assign to the least frequent class. Try plotting the 1/log(x) and you will see that I mean. The lower the value, the more weight we give to the least frequent classes. But don't go below the value of 1.0
        #1 / log(1.01+0.000001) = 100
        class_weights[background_idx]=0.00000001

        return class_weights

    
        #https://github.com/pytorch/pytorch/issues/2001
    def summary(self,file=sys.stderr):
        def repr(model):
            # We treat the extra repr like the sub-module, one item per line
            extra_lines = []
            extra_repr = model.extra_repr()
            # empty string will be split into list ['']
            if extra_repr:
                extra_lines = extra_repr.split('\n')
            child_lines = []
            total_params = 0
            for key, module in model._modules.items():
                mod_str, num_params = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
                total_params += num_params
            lines = extra_lines + child_lines

            for name, p in model._parameters.items():
                if p is not None:
                    total_params += reduce(lambda x, y: x * y, p.shape)

            main_str = model._get_name() + '('
            if lines:
                # simple one-liner info, which most builtin Modules will use
                if len(extra_lines) == 1 and not child_lines:
                    main_str += extra_lines[0]
                else:
                    main_str += '\n  ' + '\n  '.join(lines) + '\n'

            main_str += ')'
            if file is sys.stderr:
                main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
            else:
                main_str += ', {:,} params'.format(total_params)
            return main_str, total_params

        string, count = repr(self)
        if file is not None:
            print(string, file=file)
        return count

