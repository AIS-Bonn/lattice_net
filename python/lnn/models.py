import torch
from torch.autograd import Function
from torch import Tensor

import sys
import os
# http://wiki.ros.org/Packages#Client_Library_Support
# import rospkg
# rospack = rospkg.RosPack()
# sf_src_path=rospack.get_path('surfel_renderer')
# sf_build_path=os.path.abspath(sf_src_path + "/../../build/surfel_renderer")
# sys.path.append(sf_build_path) #contains the modules of pycom

from easypbr  import *
from latticenet  import *
from lattice_py import LatticePy
from lattice_funcs import *
from lattice_modules import *


##Network
class LNN(torch.nn.Module):
    def __init__(self, nr_classes, with_debug_output, with_error_checking):
        super(LNN, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.splat=SplatLatticeModule()
        self.distribute=DistributeLatticeModule(self.with_debug_output, self.with_error_checking) 
        #run those positions through point_net like architecture which at the end they will max over all the points that are in the same lattice


        #another coarsening one 
        self.point_net=PointNetModule( [16,32],64 , self.with_debug_output, self.with_error_checking) 
        # self.res1=ResnetBlock(32, [1,1])
        # self.des1=DensenetBlock(32, 1)
        self.conv1=BnConvRelu(64, 1, self.with_debug_output, self.with_error_checking)
        self.coarsening_block_1=CoarseningBlock( [128,128], [1,2], self.with_debug_output, self.with_error_checking)
        self.coarsening_block_2=CoarseningBlock( [256,256], [1,2], self.with_debug_output, self.with_error_checking)
        self.coarsening_block_3=CoarseningBlock( [512,512], [1,1], self.with_debug_output, self.with_error_checking)
        # self.coarsening_block_4=CoarseningBlock( [512,512], [1,1], self.with_debug_output, self.with_error_checking) #this doesnt have many vertices so we don't use that much dilation

        # self.finefy_block_4=FinefyBlock( [512,512], [1,1], self.with_debug_output, self.with_error_checking) #decoarsenes coarsening block 4
        self.finefy_block_3=FinefyBlock( [512,512], [1,1], self.with_debug_output, self.with_error_checking)#decoarsenes coarsening block 3
        self.finefy_block_2=FinefyBlock( [256,256], [1,2], self.with_debug_output, self.with_error_checking)#decoarsenes coarsening block 2
        self.finefy_block_1=FinefyBlock( [128,128], [1,2], self.with_debug_output, self.with_error_checking)#decoarsenes coarsening block 1

       
        self.stepdown = StepDownModule([], nr_classes, self.with_debug_output, self.with_error_checking)

        self.slice=SliceLatticeModule(with_homogeneous_coord=False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.relu=torch.nn.ReLU()
        self.tanh=torch.nn.Tanh()
        # self.bn=torch.nn.BatchNorm1d(nr_params).to("cuda")
        self.bn=None
        self.bn0=None
        self.softmax=torch.nn.LogSoftmax(dim=2)

        self.first_forward=True
        self.prev_distributed=torch.zeros(1)
        self.prev_distributed_reduced=torch.zeros(1)
        self.prev_indices=torch.zeros(1)

    def forward(self, ls, positions, values):



        #with coarsening block--------------------------------------------------
        TIME_START("distribute_py")
        distributed, indices=self.distribute(ls, positions, values)
        TIME_END("distribute_py")

        TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        TIME_END("pointnet+bn")

        #just one coarsening block
        lv, ls = self.conv1(lv,ls)
        lv_2, ls_2=self.coarsening_block_1(lv, ls) 
        lv_3, ls_3=self.coarsening_block_2(lv_2, ls_2) 
        lv_4, ls_4=self.coarsening_block_3(lv_3, ls_3) 

        #concatenating the fine part 
        # lv_4_f, ls_4_f=self.finefy_block_4(lv_5, ls_5, lv_4, ls_4)
        lv_3_f, ls_3_f=self.finefy_block_3(lv_4, ls_4, lv_3, ls_3)
        # lv_3_f, ls_3_f=self.finefy_block_3(lv_4_f, ls_4_f, lv_3, ls_3)
        lv_2_f, ls_2_f=self.finefy_block_2(lv_3_f, ls_3_f, lv_2, ls_2)
        lv_1_f, ls_1_f=self.finefy_block_1(lv_2_f, ls_2_f, lv, ls)

        #bn, slice relu
        # if self.bn0 is None:
            # self.bn0=torch.nn.BatchNorm1d(lv_1_f.shape[1]).to("cuda")
        # lv_1_f=self.bn0(lv_1_f)
        # lv_1_f=self.relu(lv_1_f)
        sv_2=self.slice(lv_1_f, ls, positions)

        #bn relu stepdown
        # if self.bn is None:
        #     self.bn=torch.nn.BatchNorm1d(sv_2.shape[2]).to("cuda")
        # sv_2=sv_2.squeeze(0)
        # sv_2=self.bn(sv_2)
        # sv_2=sv_2.unsqueeze(0)
        # sv_2=torch.nn.functional.relu(sv_2)
        s_final=self.stepdown(sv_2)
        sliced_values=self.softmax(s_final)


       
        # return s_final
        return sliced_values
        #with coarsening block finished-----------------------------------------------------



class LNN_tiramisu(torch.nn.Module):
    def __init__(self, nr_classes, with_debug_output, with_error_checking):
        super(LNN_tiramisu, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.splat=SplatLatticeModule()
        self.distribute=DistributeLatticeModule(self.with_debug_output, self.with_error_checking) 
        #run those positions through point_net like architecture which at the end they will max over all the points that are in the same lattice

       
        #another coarsening one 
        self.last_pointnet_feature_nr=16
        self.point_net=PointNetModule( [8,16,16], self.last_pointnet_feature_nr , self.with_debug_output, self.with_error_checking) 
        # self.des1=DensenetBlock(12, 1, 2, self.with_debug_output, self.with_error_checking)
       

        # self.stepdown = StepDownModule([300, 100, 50], nr_classes, self.with_debug_output, self.with_error_checking)
        # self.stepdown = StepDownModule([16,16,16], nr_classes, self.with_debug_output, self.with_error_checking)
        # self.stepdown = StepDownModule([], nr_classes, self.with_debug_output, self.with_error_checking)
        # try another one!
        # self.stepdown = StepDownModuleResnet(64,[64,64], nr_classes, self.with_debug_output, self.with_error_checking)
        self.stepdown = StepDownModuleDensenet(64,[16,16], nr_classes, self.with_debug_output, self.with_error_checking)
       
        self.slice=SliceLatticeModule(with_homogeneous_coord=False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.gather=GatherLatticeModule(with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.relu=torch.nn.ReLU()
        self.bn0=None
        self.bn=None
        self.softmax=torch.nn.LogSoftmax(dim=2)

        self.first_forward=True
        self.prev_distributed=torch.zeros(1)
        self.prev_distributed_reduced=torch.zeros(1)
        self.prev_indices=torch.zeros(1)


        #attempt 2 #similar to https://github.com/bfortuner/pytorch_tiramisu/blob/master/models/tiramisu.py
        self.growth_rate=8
        self.nr_downsamples=2 # if there are 2 downsamples means that we will have 2 denseblocks, each will have nr of laters 2 and then 4 and then 6 and so on

        #####################
        # Downsampling path #
        #####################
        self.dense_blocks_down_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        # nr_layers_per_down_block=[4, 5, 7, 10] #from https://github.com/SimJeg/FC-DenseNet/blob/master/config/FC-DenseNet103.py
        # nr_layers_per_down_block=[5, 5, 5]
        nr_layers_per_down_block=[4, 5, 6]
        # nr_layers_per_down_block=[4, 5]
        skip_connection_channel_counts = []
        cur_channels_count=self.last_pointnet_feature_nr
        for i in range(self.nr_downsamples):
            self.dense_blocks_down_list.append(  DensenetBlock(self.growth_rate, 1, nr_layers_per_down_block[i], self.with_debug_output, self.with_error_checking)  )
            cur_channels_count += (self.growth_rate*nr_layers_per_down_block[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            # self.coarsens_list.append( BnCoarsenRelu(cur_channels_count, self.with_debug_output, self.with_error_checking))
            self.coarsens_list.append( BnReluCoarsen(cur_channels_count, self.with_debug_output, self.with_error_checking))

        #####################
        #     Bottleneck    #
        #####################
        self.layers_bottleneck=7
        self.bottleneck = DensenetBlock(self.growth_rate, 1, self.layers_bottleneck, self.with_debug_output, self.with_error_checking) 
        prev_block_channels = self.growth_rate*self.layers_bottleneck

        #######################
        #   Upsampling path   #
        #######################
        self.finefy_list=torch.nn.ModuleList([])
        self.dense_blocks_up_list=torch.nn.ModuleList([])
        nr_layers_per_up_block=nr_layers_per_down_block[::-1]
        for i in range(self.nr_downsamples):
            # self.finefy_list.append( BnFinefyRelu(prev_block_channels, self.with_debug_output, self.with_error_checking))
            self.finefy_list.append( BnReluFinefy(prev_block_channels, self.with_debug_output, self.with_error_checking))
            # self.finefy_list.append( FinefyLatticeModule(prev_block_channels, self.with_debug_output, self.with_error_checking))
            self.dense_blocks_up_list.append(  DensenetBlock(self.growth_rate, 1, nr_layers_per_up_block[i], self.with_debug_output, self.with_error_checking)  )
            prev_block_channels = self.growth_rate*nr_layers_per_up_block[i]

        # self.conv_first=BnConvRelu(64, 1, self.with_debug_output, self.with_error_checking)
        # self.conv_last=BnConvRelu(64, 1, self.with_debug_output, self.with_error_checking)
        self.conv_bn_conv_relu=BnConvRelu(64, 1, self.with_debug_output, self.with_error_checking)
        self.conv_bn_relu_conv=BnReluConv(64, 1, self.with_debug_output, self.with_error_checking)

        # self.last_dropout=DropoutLattice(0.2) 
        self.bn0=None
        self.bn=None



    def forward(self, ls, positions, values):

  
        #with coarsening block--------------------------------------------------
        TIME_START("distribute_py")
        distributed, indices=self.distribute(ls, positions, values)
        TIME_END("distribute_py")

        TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        TIME_END("pointnet+bn")

        #conv_input
        #todo
        # lv_first, ls = self.conv_first(lv,ls)
        # lv_first=torch.cat((lv_first,lv),1)


        #downsample
        stack=lv
        fine_structures_list=[]
        fine_values_list=[]
        for i in range(self.nr_downsamples):
            print("DOWNSAPLE ", i)

            #denseblock
            print("db input shape ", stack.shape[1])
            lv, ls = self.dense_blocks_down_list[i] ( stack, ls) 

            #now we need a concat
            stack=torch.cat((stack, lv),1)

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(stack)

            #now we do a downsample
            print("down input shape ", stack.shape[1])
            stack, ls = self.coarsens_list[i] ( stack, ls)


        # #bottleneck
        # print("bottleneck input shape ", stack.shape[1])
        block_to_upsample, ls = self.bottleneck(stack, ls)
        # print("bottleneck output shape ", block_to_upsample.shape[1])

        #upsample (we start from the bottom of the u, so the upsampling that is closest to the blottlenck)
        for i in range(self.nr_downsamples):
            print("UPSAMPLE ", i)

            fine_values=fine_values_list.pop()
            fine_structure=fine_structures_list.pop()

            #finefy
            print("finefy input shape", block_to_upsample.shape[1])
            stack, ls = self.finefy_list[i] ( block_to_upsample, ls, fine_structure  )
            print("after finefy stack is " , stack.shape[1])

            stack=torch.cat((stack, fine_values ),1)
            print("after after concating stack is " , stack.shape[1])

            block_to_upsample, ls=self.dense_blocks_up_list[i]( stack, ls)
            print("after denseblock ", i , "block to upsample is ", block_to_upsample.shape[1])

            #do a dum thing and concat also the stack
            # block_to_upsample=torch.cat((block_to_upsample,stack),1)


        #why the fuck does bn-relu-conv perform so much worse???????????????????????????
        # block_to_upsample, ls = self.conv_bn_conv_relu(lv, ls)
        # block_to_upsample, ls = self.conv_bn_relu_conv(lv, ls)

    

        # if self.bn0 is None:
        #     self.bn0=torch.nn.BatchNorm1d(block_to_upsample.shape[1]).to("cuda")
        # block_to_upsample=self.bn0(block_to_upsample)
        # block_to_upsample=self.relu(block_to_upsample)
        # block_to_upsample=self.last_dropout(block_to_upsample)
        # sv_2=self.slice(block_to_upsample, ls, positions)
        # sv_2_sliced=sv_2
        sv_2=self.gather(block_to_upsample, ls, positions)
        #max over the axis 2 (m_pos_dim)
        # print("gathered values is ", sv_2)
        # print("gathered values has shape is ", sv_2.shape)


        # #trying some other dumb stuff like maxing the values over all the points instead of summing like slicing does
        # #shape is 1 x nr_positions x ( (m_pos_dim+1)x(m_val_full_dim+1) )
        # #reshape into 1 x nr_positions x (m_pos_dim+1) x (m_val_full_dim+1)
        # sv_2=sv_2.reshape(1,sv_2.shape[1], ls.pos_dim()+1, ls.val_full_dim()+1)
        # sv_max, _ = sv_2.max(2)
        # # sv_max = sv_2.sum(2)
        # print("sv_max has size", sv_max.shape)
        # sv_2=sv_max

        # #sv_2 an sv_2_should be almost the same
        # sv_2=sv_2[:,:,:80]
        # print("sv_2 has shape ", sv_2.shape)
        # print("sv_2_sliced has shape ", sv_2_sliced.shape)
        # diff=sv_2-sv_2_sliced
        # diff_norm=diff.norm()
        # diff_sum=diff.sum()
        # print("sv_2_sliced is ", sv_2_sliced)
        # print("sv_2 is ", sv_2)
        # print("diff_norm between the two sv_2 is ", diff_norm)
        # print("diff_sum between the two sv_2 is ", diff_sum)


        
        # if self.bn is None:
        #     self.bn=torch.nn.BatchNorm1d(sv_2.shape[2]).to("cuda")
        # sv_2=sv_2.squeeze(0)
        # sv_2=self.bn(sv_2)
        # sv_2=sv_2.unsqueeze(0)
        # sv_2=torch.nn.functional.relu(sv_2)
        s_final=self.stepdown(sv_2)


        sliced_values=self.softmax(s_final)

     






        # # ##attempt 2
        # #downsample
        # stack=lv
        # finer_structures=[]
        # finer_values=[]
        # lv, ls = self.des1(stack,ls)
        # stack=torch.cat((stack, lv),1)
        # finer_structures.append(ls) #saving them for when we do finefy
        # finer_values.append(stack)
        # print("in coarsen we ented with a stack of shape ", stack.shape)
        # stack, ls = self.coarse1 ( stack, ls)
        # finer_structures.reverse()
        # finer_values.reverse()

        # #bottleneck
        # print("in bottleneck we ented with a stack of shape ", stack.shape)
        # block_to_upsample, ls = self.bottleneck(stack, ls)
        # print("after bottleneck the block to upsample is ", block_to_upsample.shape)

        # #upsample
        # stack, ls = self.fine1 ( block_to_upsample, ls, finer_structures[0]  )
        # stack=torch.cat((stack,finer_values[0]),1)
        # block_to_upsample, ls=self.des2( stack, ls)

    
        # sv_2=self.slice(block_to_upsample, ls, positions)
        # s_final=self.stepdown(sv_2)
        # sliced_values=self.softmax(s_final)




       
        # return s_final
        return sliced_values
        #with coarsening block finished-----------------------------------------------------

class LNN_unet(torch.nn.Module):
    def __init__(self, nr_classes, with_debug_output, with_error_checking):
        super(LNN_unet, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.splat=SplatLatticeModule()
        self.distribute=DistributeLatticeModule(self.with_debug_output, self.with_error_checking) 
        #run those positions through point_net like architecture which at the end they will max over all the points that are in the same lattice


        #another coarsening one 
        self.point_net=PointNetModule( [8,16,16],16 , self.with_debug_output, self.with_error_checking) 
        # self.res1=ResnetBlock(32, [1,1])
        # self.des1=DensenetBlock(32, 1)
        self.conv1=BnConvRelu(32, 1, self.with_debug_output, self.with_error_checking)
        self.coarsening_block_1=CoarseningBlock( [64,64], [1,2], self.with_debug_output, self.with_error_checking)
        self.coarsening_block_2=CoarseningBlock( [128,128], [1,2], self.with_debug_output, self.with_error_checking)
        self.coarsening_block_3=CoarseningBlock( [128,128], [1,1], self.with_debug_output, self.with_error_checking)
        # self.coarsening_block_4=CoarseningBlock( [512,512], [1,1], self.with_debug_output, self.with_error_checking) #this doesnt have many vertices so we don't use that much dilation

        # self.finefy_block_4=FinefyBlock( [512,512], [1,1], self.with_debug_output, self.with_error_checking) #decoarsenes coarsening block 4
        self.finefy_block_3=FinefyBlock( [128,128], [1,1], self.with_debug_output, self.with_error_checking)#decoarsenes coarsening block 3
        self.finefy_block_2=FinefyBlock( [128,128], [1,2], self.with_debug_output, self.with_error_checking)#decoarsenes coarsening block 2
        self.finefy_block_1=FinefyBlock( [64,64], [1,2], self.with_debug_output, self.with_error_checking)#decoarsenes coarsening block 1

       
        # self.stepdown = StepDownModule([], nr_classes, self.with_debug_output, self.with_error_checking)
        self.stepdown = StepDownModuleDensenet(32,[16,16], nr_classes, self.with_debug_output, self.with_error_checking)

        self.slice=SliceLatticeModule(with_homogeneous_coord=False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.gather=GatherLatticeModule(with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.relu=torch.nn.ReLU()
        self.tanh=torch.nn.Tanh()
        # self.bn=torch.nn.BatchNorm1d(nr_params).to("cuda")
        self.bn=None
        self.bn0=None
        self.softmax=torch.nn.LogSoftmax(dim=2)

        self.first_forward=True
        self.prev_distributed=torch.zeros(1)
        self.prev_distributed_reduced=torch.zeros(1)
        self.prev_indices=torch.zeros(1)

    def forward(self, ls, positions, values):



        #with coarsening block--------------------------------------------------
        TIME_START("distribute_py")
        distributed, indices=self.distribute(ls, positions, values)
        TIME_END("distribute_py")

        TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        TIME_END("pointnet+bn")

        #just one coarsening block
        lv, ls = self.conv1(lv,ls)
        lv_2, ls_2=self.coarsening_block_1(lv, ls) 
        lv_3, ls_3=self.coarsening_block_2(lv_2, ls_2) 
        lv_4, ls_4=self.coarsening_block_3(lv_3, ls_3) 

        #concatenating the fine part 
        # lv_4_f, ls_4_f=self.finefy_block_4(lv_5, ls_5, lv_4, ls_4)
        lv_3_f, ls_3_f=self.finefy_block_3(lv_4, ls_4, lv_3, ls_3)
        # lv_3_f, ls_3_f=self.finefy_block_3(lv_4_f, ls_4_f, lv_3, ls_3)
        lv_2_f, ls_2_f=self.finefy_block_2(lv_3_f, ls_3_f, lv_2, ls_2)
        lv_1_f, ls_1_f=self.finefy_block_1(lv_2_f, ls_2_f, lv, ls)

        #bn, slice relu
        # if self.bn0 is None:
            # self.bn0=torch.nn.BatchNorm1d(lv_1_f.shape[1]).to("cuda")
        # lv_1_f=self.bn0(lv_1_f)
        # lv_1_f=self.relu(lv_1_f)
        sv_2=self.slice(lv_1_f, ls, positions)
        # sv_2=self.gather(lv_1_f, ls, positions) ########SOmething is wrong with it, it LEAKES MEMORY

        #bn relu stepdown
        # if self.bn is None:
        #     self.bn=torch.nn.BatchNorm1d(sv_2.shape[2]).to("cuda")
        # sv_2=sv_2.squeeze(0)
        # sv_2=self.bn(sv_2)
        # sv_2=sv_2.unsqueeze(0)
        # sv_2=torch.nn.functional.relu(sv_2)
        s_final=self.stepdown(sv_2)
        sliced_values=self.softmax(s_final)


       
        # return s_final
        return sliced_values
        #with coarsening block finished-----------------------------------------------------



class LNN_skippy(torch.nn.Module):
    def __init__(self, nr_classes, with_debug_output, with_error_checking):
        super(LNN_skippy, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.splat=SplatLatticeModule()
        self.distribute=DistributeLatticeModule(self.with_debug_output, self.with_error_checking) 
        #run those positions through point_net like architecture which at the end they will max over all the points that are in the same lattice

       
        #another coarsening one 
        # self.last_pointnet_feature_nr=32
        # self.point_net=PointNetModule( [8,16,32], self.last_pointnet_feature_nr , self.with_debug_output, self.with_error_checking)  #has a 2k more parameters than the densenet one but makes the train loss go lower
        self.last_pointnet_feature_nr=8
        self.point_net=PointNetModule( [8], self.last_pointnet_feature_nr , self.with_debug_output, self.with_error_checking)  #has a 2k more parameters than the densenet one but makes the train loss go lower
        # self.last_pointnet_feature_nr=7 + 8*3 #a bit hacky to hardocde it here but in this constructor we don;t know that distribute will give us a feature of 4 dimensions
        # self.point_net=PointNetDenseModule( growth_rate=8, nr_layers=3 , with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking) 
 
       
        self.slice=SliceLatticeModule(with_homogeneous_coord=False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.gather=GatherLatticeModule(with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.slice_conv=SliceConvLatticeModule(with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.relu=torch.nn.ReLU()
        self.bn0=None
        self.bn=None
        self.softmax=torch.nn.LogSoftmax(dim=2)

        self.first_forward=True
        self.prev_distributed=torch.zeros(1)
        self.prev_distributed_reduced=torch.zeros(1)
        self.prev_indices=torch.zeros(1)


        #attempt 2 #similar to https://github.com/bfortuner/pytorch_tiramisu/blob/master/models/tiramisu.py
        self.growth_rate=24
        self.nr_downsamples=2 # if there are 2 downsamples means that we will have 2 denseblocks, each will have nr of laters 2 and then 4 and then 6 and so on

        #####################
        # Downsampling path #
        #####################
        self.dense_blocks_down_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        # nr_layers_per_down_block=[4, 5, 7, 10] #from https://github.com/SimJeg/FC-DenseNet/blob/master/config/FC-DenseNet103.py
        # nr_layers_per_down_block=[5, 5, 5]
        # nr_layers_per_down_block=[4, 5, 6]
        # nr_layers_per_down_block=[4, 5]
        # nr_layers_per_down_block=[5, 5, 5]
        # nr_layers_per_down_block=[6, 8]
        nr_layers_per_down_block=[6,8]
        compression_factor=0.5
        skip_connection_channel_counts = []
        cur_channels_count=self.last_pointnet_feature_nr
        for i in range(self.nr_downsamples):
            self.dense_blocks_down_list.append(  DensenetBlock(self.growth_rate, [1]*nr_layers_per_down_block[i], nr_layers_per_down_block[i], self.with_debug_output, self.with_error_checking)  )
            cur_channels_count += (self.growth_rate*nr_layers_per_down_block[i])
            skip_connection_channel_counts.append(cur_channels_count)
            self.coarsens_list.append( BnReluCoarsen(int(cur_channels_count*compression_factor), self.with_debug_output, self.with_error_checking))
            cur_channels_count=int(cur_channels_count*compression_factor)

        #####################
        #     Bottleneck    #
        #####################
        self.layers_bottleneck=8
        self.bottleneck = DensenetBlock(self.growth_rate, [1]*self.layers_bottleneck, self.layers_bottleneck, self.with_debug_output, self.with_error_checking) 

        #######################
        #   Upsampling path   #
        #######################
        self.finefy_list=torch.nn.ModuleList([])
        self.dense_blocks_up_list=torch.nn.ModuleList([])
        nr_layers_per_up_block=nr_layers_per_down_block[::-1] #it just reverts the list
        for i in range(self.nr_downsamples):
            nr_channels_after_finify=skip_connection_channel_counts.pop()
            self.finefy_list.append( BnReluFinefy(nr_channels_after_finify, self.with_debug_output, self.with_error_checking))
            # self.finefy_list.append( FinefyLatticeModule(nr_channels_after_finify, self.with_debug_output, self.with_error_checking))
            # self.finefy_list.append( BnFinefy(nr_channels_after_finify, self.with_debug_output, self.with_error_checking))
            self.dense_blocks_up_list.append(  DensenetBlock(self.growth_rate, [1]*nr_layers_per_up_block[i], nr_layers_per_up_block[i], self.with_debug_output, self.with_error_checking)  )

        # self.conv_first=BnConvRelu(64, 1, self.with_debug_output, self.with_error_checking)
        # self.conv_last=BnConvRelu(64, 1, self.with_debug_output, self.with_error_checking)
        # self.conv_bn_conv_relu=BnConvRelu(64, 1, self.with_debug_output, self.with_error_checking)
        # self.conv_bn_relu_conv=BnReluConv(64, 1, self.with_debug_output, self.with_error_checking)


       # self.stepdown = StepDownModule([300, 100, 50], nr_classes, self.with_debug_output, self.with_error_checking)
        # self.stepdown = StepDownModule([16,16,16], nr_classes, self.with_debug_output, self.with_error_checking)
        # self.stepdown = StepDownModule([], nr_classes, self.with_debug_output, self.with_error_checking)
        # try another one!
        # self.stepdown = StepDownModuleResnet(64,[64,64], nr_classes, self.with_debug_output, self.with_error_checking)
        # self.stepdown = StepDownModuleDensenet(64,[16,16], nr_classes, self.with_debug_output, self.with_error_checking)
        # self.stepdown = StepDownModuleDensenet(128,[16,16], nr_classes, self.with_debug_output, self.with_error_checking)
        # self.stepdown = StepDownModuleDensenet(197,[], nr_classes, self.with_debug_output, self.with_error_checking) #the first block acts as finefy and then we have a clasification layer
        self.stepdown = StepDownModuleDensenetNoBottleneck(16, 3, nr_classes, self.with_debug_output, self.with_error_checking)
        # self.stepdown = StepDownModuleDensenetNoBottleneck(64, 1, nr_classes, self.with_debug_output, self.with_error_checking)
        #with 16x3 we have nr params 933324
        #with 8x5 we have nr params 932644 which gets sligthly worse iou than 16x3
        #with 32x2 we have nr of params 935620 which gets a bit better accuracy thatn 16x3, it get 0.66 at 4 epochs
        # with 64x1 we have nr of params 934252 which still reacher 0.66 at epoch 4
        # with stepdown that does just the last linear into the nr classes we have nr_of_params 924572  we get waaay worse accuracy, don't even try it!



        #trying other architecture so I need the building blocks here
        # self.growth_rate=32
        # self.des1=DensenetBlock(self.growth_rate, [1,1], 2, self.with_debug_output, self.with_error_checking)
        #with gr of 8 and 4 layers, params is 36864, reaches 0.55 at epoch 13
        #with gr of 8 adn 8 layers, params is 82016, reahces 0.55 at epoch 13, so it's not that much better in iou wthat the one with 4,
        #with gr of 16 and 4 layers we have params, 79200, reaches 0.567 at epoch 13. Therefore it's more important to have more growth rate than just more layers
        #with gr of 32 and 2 layers we have params, 74336, reaches 0.562 at epoch 13
        #with gr of 32 and 2 layers but the second layer has a dilation of 2, reaches 0.582. Not bad for something that doesnt increase parameters at all




        # self.last_dropout=DropoutLattice(0.2) 
        self.bn0=None
        self.bn=None



    def forward(self, ls, positions, values):

  
        #with coarsening block--------------------------------------------------
        TIME_START("distribute_py")
        distributed, indices=self.distribute(ls, positions, values)
        TIME_END("distribute_py")

        TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        TIME_END("pointnet+bn")

        # ls.compute_nr_points_per_lattice_vertex()


        #downsample
        stack=lv
        fine_structures_list=[]
        fine_values_list=[]
        for i in range(self.nr_downsamples):
            print("DOWNSAPLE ", i)

            #denseblock
            print("db input shape ", stack.shape[1])
            lv, ls = self.dense_blocks_down_list[i] ( stack, ls) 

            #now we need a concat
            stack=torch.cat((stack, lv),1)

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(stack)

            #now we do a downsample
            print("down input shape ", stack.shape[1])
            stack, ls = self.coarsens_list[i] ( stack, ls)


        # #bottleneck
        # print("bottleneck input shape ", stack.shape[1])
        block_to_upsample, ls = self.bottleneck(stack, ls)
        # block_to_upsample=torch.cat((block_to_upsample,stack),1)

        #upsample (we start from the bottom of the u, so the upsampling that is closest to the blottlenck)
        for i in range(self.nr_downsamples):
            print("UPSAMPLE ", i)

            fine_values=fine_values_list.pop()
            fine_structure=fine_structures_list.pop()

            #NEW PART
            block_to_upsample=torch.cat((block_to_upsample,stack),1)

            #finefy
            print("finefy input shape", block_to_upsample.shape[1])
            stack, ls = self.finefy_list[i] ( block_to_upsample, ls, fine_structure  )
            print("after finefy stack is " , stack.shape[1])

            #NEW PART
            stack+=fine_values #skip connection from the downsampling part, The tiramisu instead of this did a concat
            
            # stack=torch.cat((stack, fine_values ),1)
            # print("after after concating stack is " , stack.shape[1])

            block_to_upsample, ls=self.dense_blocks_up_list[i]( stack, ls)
            print("after denseblock ", i , "block to upsample is ", block_to_upsample.shape[1])

            #do a dum thing and concat also the stack
            # block_to_upsample=torch.cat((block_to_upsample,stack),1)

        #last concat
        block_to_upsample=torch.cat((block_to_upsample,stack),1)
        print("last block to upsample before a gather is ", block_to_upsample.shape[1])




    

        # if self.bn0 is None:
        #     self.bn0=torch.nn.BatchNorm1d(block_to_upsample.shape[1]).to("cuda")
        # block_to_upsample=self.bn0(block_to_upsample)
        # block_to_upsample=self.relu(block_to_upsample)
        # block_to_upsample=self.last_dropout(block_to_upsample)
        # sv_2=self.slice(lv, ls, positions) #if slice is acoompanied by a stepdown with a bn-relu-conv it actually works better than a gather, o at least it converges faster
        # sv_2=self.slice(block_to_upsample, ls, positions) #if slice is acoompanied by a stepdown with a bn-relu-conv it actually works better than a gather, o at least it converges faster
        # sv_2_sliced=sv_2
        # sv_2=self.gather(block_to_upsample, ls, positions)
        # print("gathered values is ", sv_2)
        # sv_2=self.slice_conv(lv, ls, positions)
        sv_2=self.slice_conv(block_to_upsample, ls, positions)
        # sv_2=torch.cat((sv_1,))
        print("gathered values has shape is ", sv_2.shape)


        # #trying some other dumb stuff like maxing the values over all the points instead of summing like slicing does
        # #shape is 1 x nr_positions x ( (m_pos_dim+1)x(m_val_full_dim+1) )
        # #reshape into 1 x nr_positions x (m_pos_dim+1) x (m_val_full_dim+1)
        # sv_2=sv_2.reshape(1,sv_2.shape[1], ls.pos_dim()+1, ls.val_full_dim()+1)
        # sv_max, _ = sv_2.max(2)
        # # sv_max = sv_2.sum(2)
        # print("sv_max has size", sv_max.shape)
        # sv_2=sv_max

        # #sv_2 an sv_2_should be almost the same
        # sv_2=sv_2[:,:,:80]
        # print("sv_2 has shape ", sv_2.shape)
        # print("sv_2_sliced has shape ", sv_2_sliced.shape)
        # diff=sv_2-sv_2_sliced
        # diff_norm=diff.norm()
        # diff_sum=diff.sum()
        # print("sv_2_sliced is ", sv_2_sliced)
        # print("sv_2 is ", sv_2)
        # print("diff_norm between the two sv_2 is ", diff_norm)
        # print("diff_sum between the two sv_2 is ", diff_sum)


        
        # if self.bn is None:
        #     self.bn=torch.nn.BatchNorm1d(sv_2.shape[2]).to("cuda")
        # sv_2=sv_2.squeeze(0)
        # sv_2=self.bn(sv_2)
        # sv_2=sv_2.unsqueeze(0)
        # sv_2=torch.nn.functional.relu(sv_2)
        self.per_point_features=sv_2
        s_final=self.stepdown(sv_2)


        sliced_values=self.softmax(s_final)

        return sliced_values


#trying out something similr to the skippy version but experimentig a bit with increasing layers and growth rate
class LNN_skippy_v2(torch.nn.Module):
    def __init__(self, nr_classes, with_debug_output, with_error_checking):
        super(LNN_skippy_v2, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.relu=torch.nn.ReLU()
        self.distribute=DistributeLatticeModule(self.with_debug_output, self.with_error_checking) 

       
        #another coarsening one 
        self.last_pointnet_feature_nr=32
        self.point_net=PointNetModule( [8,16,32], self.last_pointnet_feature_nr , self.with_debug_output, self.with_error_checking)  #has a 2k more parameters than the densenet one but makes the train loss go lower
 
       
        self.slice_conv=SliceConvLatticeModule(with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.stepdown = StepDownModuleDensenetNoBottleneck(8, 4, nr_classes, self.with_debug_output, self.with_error_checking) #the more layers we have the better, we can make then with quite fre chanels each actually . So its better to have 8 channels and then 4 layers then 16 and only 3 layers
        self.softmax=torch.nn.LogSoftmax(dim=2)


        #trying other architecture so I need the building blocks here
        self.growth_rate=32
        self.des1=DensenetBlock(self.growth_rate, [1,2], 2, self.with_debug_output, self.with_error_checking)
        #with gr of 8 and 4 layers, params is 36864, reaches 0.55 at epoch 13
        #with gr of 8 adn 8 layers, params is 82016, reahces 0.55 at epoch 13, so it's not that much better in iou wthat the one with 4,
        #with gr of 16 and 4 layers we have params, 79200, reaches 0.567 at epoch 13. Therefore it's more important to have more growth rate than just more layers
        #with gr of 32 and 2 layers we have params, 74336, reaches 0.562 at epoch 13
        #with gr of 32 and 2 layers but the second layer has a dilation of 2, reaches 0.582. Not bad for something that doesnt increase parameters at all
        self.down=BnReluCoarsen(96, self.with_debug_output, self.with_error_checking)
        #adding a coarsening with the same nr of channels put us at  157472 params and reaches 0.5928 at iter 13
        #having stepdown with less nr_filters but more layers (8 channels and 4 layers) has params 155632 and reaches 0.60 at iter 13
        self.des2=DensenetBlock(self.growth_rate, [1,2], 2, self.with_debug_output, self.with_error_checking)
        #adding another denseblock puts us at 289584 and reaches 61.4 at epoch 13
        # self.up=BnReluFinefy(160, self.with_debug_output, self.with_error_checking)
        self.up=BnFinefy(96, self.with_debug_output, self.with_error_checking)
        #adding also a bn-relu-finefy we have nr of params 520304 but somehow the training loss decreases more slowly and we only reach 0.561 at epoch 13
        #bn finefy seems to work a bit better, reaching 0.567 at epoch 13
        #just finefy seems ot be the worse, so we will rather use the one that does bn finefy
        #having a BnFinefy have the same nr of channels as the previous block, means we can use a skip connection, params 359152 and reaches 0.618 at epoch 13
        self.des3=DensenetBlock(self.growth_rate, [1,2], 2, self.with_debug_output, self.with_error_checking)
        #adding the last dense layer puts us at nr parms 493104 and reaches 64.1 at epoch 13, runs at around 28ms and uses 567MB



    def forward(self, ls, positions, values):

  
        #with coarsening block--------------------------------------------------
        TIME_START("distribute_py")
        distributed, indices=self.distribute(ls, positions, values)
        TIME_END("distribute_py")

        TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        TIME_END("pointnet+bn")

        





        #tryi new architectures little by little
        lv_d, ls = self.des1(lv,ls)
        lv=torch.cat((lv,lv_d),1)
        identity=lv

        lv_2,ls_2=self.down(lv,ls)
        lv_2_d, ls_2 = self.des2(lv_2,ls_2)
        lv_2=torch.cat((lv_2,lv_2_d),1)

        lv,ls=self.up(lv_2, ls_2, ls)
        print("identity has shape", identity.shape)
        lv+=identity

        lv_d, ls = self.des3(lv,ls)
        lv=torch.cat((lv,lv_d),1)





        print("values just before slicing or gathering or whatever has shape ", lv.shape)
        sv_2=self.slice_conv(lv, ls, positions)
        print("sliced or gathered values has shape is ", sv_2.shape)


       
        s_final=self.stepdown(sv_2)
        sliced_values=self.softmax(s_final)

        return sliced_values




#trying again the resnet unet architecture but now with proper bn-relu-conv
class LNN_unet_v2(torch.nn.Module):
    def __init__(self, nr_classes, with_debug_output, with_error_checking):
        super(LNN_unet_v2, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking
        self.relu=torch.nn.ReLU()
        self.distribute=DistributeLatticeModule(self.with_debug_output, self.with_error_checking) 

       
        #another coarsening one 
        self.last_pointnet_feature_nr=128
        self.point_net=PointNetModule( [8,16,32], self.last_pointnet_feature_nr , self.with_debug_output, self.with_error_checking)  #has a 2k more parameters than the densenet one but makes the train loss go lower
        # self.point_net=PointNetResnetModule( 16, self.last_pointnet_feature_nr , self.with_debug_output, self.with_error_checking)  
 
       
        # self.slice=SliceLatticeModule(with_homogeneous_coord=False, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.slice_conv=SliceConvLatticeModule(with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.stepdown = StepDownModuleDensenetNoBottleneck(8, 4, nr_classes, self.with_debug_output, self.with_error_checking) #the more layers we have the better, we can make then with quite fre chanels each actually . Sio its better to have 8 channels and then 4 layers then 16 and only 3 layers
        # self.stepdown = StepDownModule([], nr_classes, self.with_debug_output, self.with_error_checking) 
        # self.stepdown = StepDownModule([64, 32], nr_classes, self.with_debug_output, self.with_error_checking) 
        # self.stepdown = StepDownModuleResnetNoBottleneck([64, 64, 64, 64], nr_classes, self.with_debug_output, self.with_error_checking)  #reaches 64.5 and 64.9 with no bottleneck
        # self.stepdown = StepDownModuleResnetNoBottleneck([160, 160], nr_classes, self.with_debug_output, self.with_error_checking)  #reaches 64.5 and 64.9 with no bottleneck
        self.softmax=torch.nn.LogSoftmax(dim=2)


        # #trying other architecture so I need the building blocks here
        # self.res1=ResnetBlock(64, [1,2], self.with_debug_output, self.with_error_checking)
        # #one resnet block with 32 filters, nr params is 26992, reaches 0.494 at epoch 13
        # #one restnet block with 64 filter nr of params is 97488, reaches 0.613 at epoch 13
        # self.res2=ResnetBlock(64, [1,2], self.with_debug_output, self.with_error_checking)
        # #another resnet block puts us at nr of params 171472, reaches 63.7 at epoch 13, runs at 11ms and uses 506 MB
        # self.down=BnReluCoarsen(128, self.with_debug_output, self.with_error_checking)
        # #bn-relu-coarsen gets to nr params 297936 and 61.5 so somehow lower than before
        # self.res3=ResnetBlock(128, [1,2], self.with_debug_output, self.with_error_checking)
        # self.res4=ResnetBlock(128, [1,2], self.with_debug_output, self.with_error_checking)
        # #two more resnet puts us at nr of params 888784 and 64.7 with 20ms runtime( faster than the densent that achieves 64.1 but with resnet we are a lot faster even though we have more parameters)
        # self.up=BnReluFinefy(64, self.with_debug_output, self.with_error_checking)
        # self.res5=ResnetBlock(64, [1,2], self.with_debug_output, self.with_error_checking)
        # self.res6=ResnetBlock(64, [1,2], self.with_debug_output, self.with_error_checking)
        # #after two more bn-relu-finefy and two more relu blocks we have nr params 1058128 and rach 65.9 at epoch 13 running at 27ms on average on the morotbike
        # #if we do a concat instead of a skip connection and we make the two last relus to be 128 insead of 64 we have nr params 1553616 adn reach 66.58 at epoch 13


        #trying out thing attempt2 
        self.res0_1=ResnetBlock(128, [1,1], self.with_debug_output, self.with_error_checking)
        self.res0_2=ResnetBlock(128, [2,2], self.with_debug_output, self.with_error_checking)
        #3 resnetblocks at 64 has nr params 245494
        #4 resnetblocks at 128, nr params 1259959 and iou 

        #128_lv1
        self.down1=BnReluCoarsen(256, self.with_debug_output, self.with_error_checking)
        self.res1_1=ResnetBlock(256, [1,1], self.with_debug_output, self.with_error_checking)
        # self.res1_2=ResnetBlock(256, [1,1], self.with_debug_output, self.with_error_checking)
        # self.res1_3=ResnetBlock(128, [1,1], self.with_debug_output, self.with_error_checking)

        #256_lv2
        self.down2=BnReluCoarsen(256, self.with_debug_output, self.with_error_checking)
        self.res2_1=ResnetBlock(256, [1,1], self.with_debug_output, self.with_error_checking)
        self.res2_2=ResnetBlock(256, [1,1], self.with_debug_output, self.with_error_checking)

        # #512_lv3 (have it at 256 at the moment to reduce memory)
        # self.down3=BnReluCoarsen(512, self.with_debug_output, self.with_error_checking)
        # self.res3_1=ResnetBlock(512, [1,1], self.with_debug_output, self.with_error_checking)
        # self.res3_2=ResnetBlock(512, [1,1], self.with_debug_output, self.with_error_checking)

        # #256_lv3
        # self.up_3=BnReluFinefy(256, self.with_debug_output, self.with_error_checking)
        # self.res_up_3_1=ResnetBlock(256+256, [1,1], self.with_debug_output, self.with_error_checking)
        # self.res_up_3_2=ResnetBlock(256+256, [1,1], self.with_debug_output, self.with_error_checking)

        #128_lv2
        self.up_2=BnReluFinefy(128, self.with_debug_output, self.with_error_checking)
        self.res_up_2_1=ResnetBlock(128+256, [1,1], self.with_debug_output, self.with_error_checking)
        self.res_up_2_2=ResnetBlock(128+256, [1,1], self.with_debug_output, self.with_error_checking)

        #64
        self.up_1=BnReluFinefy(64, self.with_debug_output, self.with_error_checking)
        self.res_up_1_1=ResnetBlock(64+128, [1,1], self.with_debug_output, self.with_error_checking)
        # self.res_up_1_2=ResnetBlock(64+64, [1,1], self.with_debug_output, self.with_error_checking)
        # self.res_up_1_3=ResnetBlock(64+64, [1,1], self.with_debug_output, self.with_error_checking)






    def forward(self, ls, positions, values):

  
        #with coarsening block--------------------------------------------------
        TIME_START("distribute_py")
        distributed, indices=self.distribute(ls, positions, values)
        TIME_END("distribute_py")

        TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        TIME_END("pointnet+bn")

        

        #tryi new architectures little by little
        lv, ls = self.res0_1(lv,ls)
        lv, ls = self.res0_2(lv,ls)

        identity_lv_0=lv
        lv_1, ls_1 = self.down1(lv,ls)
        lv_1, ls_1 = self.res1_1(lv_1,ls_1)

        # identity_lv_1=lv_1
        # lv_2, ls_2 = self.down2(lv_1,ls_1)
        # lv_2, ls_2 = self.res2_1(lv_2,ls_2)

        # identity_lv_2=lv_2
        # lv_3, ls_3 = self.down3(lv_2,ls_2)
        # lv_3, ls_3 = self.res3_1(lv_3,ls_3)


        # #up
        # lv_2, ls_2 = self.up_3(lv_3,ls_3, ls_2)
        # lv_2=torch.cat((identity_lv_2,lv_2),1)
        # lv_2, ls_2 = self.res_up_3_1(lv_2,ls_2)


        # lv_1, ls_1 = self.up_2(lv_2,ls_2, ls_1)
        # lv_1=torch.cat((identity_lv_1,lv_1),1)
        # lv_1, ls_1 = self.res_up_2_1(lv_1,ls_1)

        lv, ls = self.up_1(lv_1,ls_1, ls)
        lv=torch.cat((identity_lv_0,lv),1)
        lv, ls = self.res_up_1_1(lv,ls)






        print("values just before slicing or gathering or whatever has shape ", lv.shape)
        # sv_2=self.slice(lv, ls, positions)
        # sv_2=self.slice(lv_1, ls_1, positions)
        # sv_2=self.slice_conv(lv, ls, positions)
        sv_2=self.slice_conv(lv_1, ls_1, positions)
        print("sliced or gathered values has shape is ", sv_2.shape)

        # C=self.pca(sv_2)
        self.per_point_features=sv_2

       
        s_final=self.stepdown(sv_2)
        sliced_values=self.softmax(s_final)


        return sliced_values


class LNN_skippy_efficient(torch.nn.Module):
    def __init__(self, nr_classes, model_params, with_debug_output, with_error_checking):
        super(LNN_skippy_efficient, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking

        self.distribute=DistributeLatticeModule(self.with_debug_output, self.with_error_checking) 
        self.distribute_cap=DistributeCapLatticeModule() 
        #self.distributed_transform=DistributedTransform( [32], self.with_debug_output, self.with_error_checking)  
        self.start_nr_filters=model_params.pointnet_start_nr_channels()
        self.point_net=PointNetModule( [16,32,64], self.start_nr_filters, self.with_debug_output, self.with_error_checking)  
        # self.start_nr_filters=model_params.pointnet_start_nr_channels()
        # self.point_net=PointNetDenseModule( growth_rate=16, nr_layers=2, nr_outputs_last_layer=self.start_nr_filters, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking) 


        #a bit more control
        self.nr_downsamples=model_params.nr_downsamples()
        self.nr_blocks_down_stage=model_params.nr_blocks_down_stage()
        self.nr_blocks_bottleneck=model_params.nr_blocks_bottleneck()
        self.nr_blocks_up_stage=model_params.nr_blocks_up_stage()
        self.nr_levels_down_with_normal_resnet=model_params.nr_levels_down_with_normal_resnet()
        self.nr_levels_up_with_normal_resnet=model_params.nr_levels_up_with_normal_resnet()
        compression_factor=model_params.compression_factor()
        dropout_last_layer=model_params.dropout_last_layer()

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
                    should_use_dropout=True
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
            cur_channels_count=nr_channels_after_coarsening

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
            print("nr_chanels_skip_connection ", nr_chanels_skip_connection)

            #do it with finefy
            if self.upsampling_method=="finefy":
                print("adding bnReluFinefy which outputs nr of channels ", nr_chanels_skip_connection )
                # self.finefy_list.append( BnReluFinefy(nr_chanels_skip_connection, self.with_debug_output, self.with_error_checking))
                # self.finefy_list.append( GnReluFinefy(nr_chanels_skip_connection, self.with_debug_output, self.with_error_checking))
                # seems that the relu in BnReluFinefy stops too much of the gradient from flowing up the network, altought we lose one non-linearity, a BnFinefy seems a lot more eneficial for the general flow of gradients as the network converges a lot faster
                self.finefy_list.append( GnFinefy(nr_chanels_skip_connection, self.with_debug_output, self.with_error_checking))
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
                cur_channels_count=nr_chanels_skip_connection*2
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
        self.slice_fast_cuda=SliceFastCUDALatticeModule(nr_classes=nr_classes, dropout_prob=dropout_last_layer, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.slice_classify=SliceClassifyLatticeModule(nr_classes=nr_classes, dropout_prob=dropout_last_layer, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        # self.stepdown = StepDownModule([], nr_classes, dropout_last_layer, self.with_debug_output, self.with_error_checking)
        #stepdown densenetmodule is too slow as it requires to copy the whole pointcloud when it concatenas. For semantic kitti this is just too much
        # self.stepdown = StepDownModuleDensenetNoBottleneck(16, 3, nr_classes, self.with_debug_output, self.with_error_checking)
       
        self.logsoftmax=torch.nn.LogSoftmax(dim=2)
        

    def forward(self, ls, positions, values):

        #create lattice vertices and fill in the splatting_indices
        
        # TIME_START("create_verts")
        # ls=self.create_verts(ls,positions)
        # TIME_END("create_verts")

        #with coarsening block--------------------------------------------------
        # TIME_START("distribute_py")
        distributed, indices=self.distribute(ls, positions, values)
        # TIME_END("distribute_py")

        print("distributed has shape", distributed.shape)
        print("indices has shape", indices.shape)
        #remove some rows of the distribured and indices depending if the corresponding lattice vertex has to many incident points
        # distributed, indices,ls=self.distribute_cap(distributed, positions.size(1), ls, cap=20)

        #transform
        TIME_START("distribute_transform")
        # distributed, ls= self.distributed_transform(ls, distributed, indices)
        TIME_END("distribute_transform")

        TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        TIME_END("pointnet+bn")

        # print("lv after pointnet has shape ", lv.shape)

        # ls.compute_nr_points_per_lattice_vertex()

        #create a whole thing with downsamples and all
        fine_structures_list=[]
        fine_values_list=[]
        TIME_START("down_path")
        for i in range(self.nr_downsamples):
            print("DOWNSAPLE ", i, " with lv of shape ", lv.shape)

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(lv)

            #now we do a downsample
            # print("down input shape ", lv.shape[1])
            # lv, ls =self.maxpool_list[i](lv,ls)
            lv, ls = self.coarsens_list[i] ( lv, ls)

        TIME_END("down_path")

        # #bottleneck
        # print("bottleneck input shape ", lv.shape[1])
        for j in range(self.nr_blocks_bottleneck):
            lv, ls = self.resnet_blocks_bottleneck[j] ( lv, ls) 


        #upsample (we start from the bottom of the u, so the upsampling that is closest to the blottlenck)
        TIME_START("up_path")
        for i in range(self.nr_downsamples):
            # print("UPSAMPLE ", i)

            fine_values=fine_values_list.pop()
            fine_structure=fine_structures_list.pop()

            # if self.upsampling_method=="slice_elevated": #if we are not using finefy, we are just slicing, so now we can do now a gn-relu and a 1x1 conv after the slice
                # lv, ls = self.up_activation_list[i](lv, ls) 

            #finefy
            print("finefy input shape", lv.shape)
            lv, ls = self.finefy_list[i] ( lv, ls, fine_structure  )
            print("after finefy lv is " , lv.shape)


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
        TIME_END("up_path")


        # print("print values before slice_conv has shape", lv.shape)

        #slicing is quite expensive, because we gather all the values around the simplex. so we try to reduce the nr of values per point 
        # lv, ls = self.last_bottleneck(lv, ls) #sligt regression from using it, Reaches only 75.4 on motoribke instead of 75.8
  
        TIME_START("slice")
        # sv=self.slice(lv, ls, positions)
        # sv=self.slice_conv(lv, ls, positions)
        # sv=self.slice_deform(lv, ls, positions)
        # sv=self.slice_deform_full(lv, ls, positions)
        # sv=self.slice_fast_pytorch(lv, ls, positions, distributed)
        # sv=self.slice_fast_bottleneck_pytorch(lv, ls, positions, distributed)
        sv, delta_weight_error_sum=self.slice_fast_cuda(lv, ls, positions)
        # sv=self.slice_classify(lv, ls, positions)
        TIME_END("slice")

        self.per_point_features=sv
        # TIME_START("stepdown")
        # s_final=self.stepdown(sv)
        # TIME_END("stepdown")


        logsoftmax=self.logsoftmax(sv)
        # logsoftmax=self.logsoftmax(s_final)

        return logsoftmax, sv, delta_weight_error_sum
        # return logsoftmax, s_final



#inspired by PSP net https://medium.com/beyondminds/a-simple-guide-to-semantic-segmentation-effcf83e7e54
#we run normal convolution in the encoding part but in the decoding part we just slice and concatenate the features from every lvl
class LNN_jpu(torch.nn.Module):
    def __init__(self, nr_classes, with_debug_output, with_error_checking):
        super(LNN_jpu, self).__init__()
        self.nr_classes=nr_classes
        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking

        self.distribute=DistributeLatticeModule(self.with_debug_output, self.with_error_checking) 
        self.start_nr_filters=64
        self.point_net=PointNetDenseModule( growth_rate=16, nr_layers=1, nr_outputs_last_layer=self.start_nr_filters, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking) 

        self.nr_downsamples=2
        self.nr_blocks_per_level=3

        #####################
        # Downsampling path #
        #####################
        # self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        self.bottleneck_list=torch.nn.ModuleList([])
        self.test_bns= torch.nn.ModuleList([])
        compression_factor=1.0
        self.bottleneck_channels=128
        skip_connection_channel_counts = []
        cur_channels_count=self.start_nr_filters
        for i in range(self.nr_downsamples):
            
            #create the resnet blocks
            self.resnet_blocks_per_down_lvl_list.append( torch.nn.ModuleList([]) )
            # cur_channels_count=self.start_nr_filters*np.power(2,i)
            for j in range(self.nr_blocks_per_level):
                print("adding down_resnet_block with nr of filters", cur_channels_count )
                self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], self.with_debug_output, self.with_error_checking) )
            skip_connection_channel_counts.append(cur_channels_count)

            #bottlenech so they all have the same channel for JPU
            self.bottleneck_list.append(torch.nn.Linear(cur_channels_count, self.bottleneck_channels, bias=True).to("cuda") )
            # self.test_bns.append(  BatchNormLatticeModule(cur_channels_count ))
            self.test_bns.append( torch.nn.BatchNorm1d(cur_channels_count).to("cuda")  )


            nr_channels_after_coarsening=int(cur_channels_count*2*compression_factor)
            self.coarsens_list.append( BnReluCoarsen(nr_channels_after_coarsening, self.with_debug_output, self.with_error_checking))
            cur_channels_count=nr_channels_after_coarsening

        #####################
        #     Bottleneck    #
        #####################
        self.resnet_blocks_bottleneck=torch.nn.ModuleList([])
        # nr_filters=self.start_nr_filters*np.power(2,self.nr_downsamples)
        # nr_filters=self.start_nr_filters*np.power(2,self.nr_downsamples)
        for j in range(self.nr_blocks_per_level):
                print("adding bottleneck_resnet_block with nr of filters", cur_channels_count )
                self.resnet_blocks_bottleneck.append( ResnetBlock(cur_channels_count, [1,1], self.with_debug_output, self.with_error_checking) )
        self.bottleneck_list.append(torch.nn.Linear(cur_channels_count, self.bottleneck_channels, bias=True).to("cuda") )
        # self.test_bns.append(  BatchNormLatticeModule(cur_channels_count ))
        self.test_bns.append( torch.nn.BatchNorm1d(cur_channels_count).to("cuda")  )

        
        self.slice_elevated=SliceElevatedVertsLatticeModule(self.with_debug_output, self.with_error_checking)
        # self.last_bottleneck=torch.nn.Linear(nr_channels_after_finify*2, 64, bias=True).to("cuda") 

        self.last_res=None
        self.last_bottleneck=None

        # self.slice_conv=SliceConvLatticeModule(nr_classes=self.nr_classes, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.slice_deform=SliceDeformLatticeModule(nr_classes=self.nr_classes, with_debug_output=self.with_debug_output, with_error_checking=self.with_error_checking)
        self.stepdown = StepDownModule([], nr_classes, self.with_debug_output, self.with_error_checking)
       
        self.logsoftmax=torch.nn.LogSoftmax(dim=2)
        

    def forward(self, ls, positions, values):

        #create lattice vertices and fill in the splatting_indices
        
        # TIME_START("create_verts")
        # ls=self.create_verts(ls,positions)
        # TIME_END("create_verts")

        #with coarsening block--------------------------------------------------
        # TIME_START("distribute_py")
        distributed, indices=self.distribute(ls, positions, values)
        # TIME_END("distribute_py")

        TIME_START("pointnet+bn")
        lv, ls=self.point_net(ls, distributed, indices)
        TIME_END("pointnet+bn")

        print("lv after pointnet has shape ", lv.shape)

        ls_finest=ls
        lv_finest=None
        lv_after_pointnet=lv

        #create a whole thing with downsamples and all
        fine_structures_list=[]
        fine_values_list=[]
        for i in range(self.nr_downsamples):
            print("DOWNSAPLE ", i)

            #resnet blocks
            for j in range(self.nr_blocks_per_level):
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(lv)
            if i==0:
                lv_finest=lv

            #now we do a downsample
            print("down input shape ", lv.shape[1])
            lv, ls = self.coarsens_list[i] ( lv, ls)


        # #bottleneck
        print("bottleneck input shape ", lv.shape[1])
        for j in range(self.nr_blocks_per_level):
            lv, ls = self.resnet_blocks_bottleneck[j] ( lv, ls) 
        fine_structures_list.append(ls) 
        fine_values_list.append(lv)

        #get each value from each lvl and make it to be the same dimension (eg 64)
        for i in range(len(fine_values_list)):
            v=fine_values_list[i]
            s=fine_structures_list[i]
            if v.shape[1]!=self.bottleneck_channels:
                v=self.test_bns[i](v)
                v=self.bottleneck_list[i](v)
            s.set_values(v)
            fine_values_list[i]=v



        #slice from the finest lvl the second finest, and then the third an so on
        sliced_values_list=[]
        for i in range(len(fine_values_list)-1): #we don;t slice up the finest one becuase the fienst one doesnt need slicing, we already have the values for it
            v=fine_values_list[i+1]
            s=fine_structures_list[i+1]
            s.set_values(v)

            #slice 
            # print("slicing from lattice with sigma ", s.lattice.sigmas_tensor() )
            sv, ss = self.slice_elevated(v,s, ls_finest)
            print("sliced_elevated some values of dim ", sv.shape)
            sliced_values_list.append(sv)
        sliced_values_list.append(lv_finest) #we add also the finest values which we don't need to slice
        # sliced_values_list.append(lv_after_pointnet) #we add also the finest values which we don't need to slice

        print("concating will concat a list of nr of values", len(sliced_values_list))
        lv=torch.cat(sliced_values_list,1)
        ls_finest.set_values(lv)
        ls=ls_finest
        print("after concating we have lv of shape", lv.shape)
        print("before last rest ls has nr_Vertices", ls.nr_lattice_vertices())

        # one last resnet block
        if self.last_res is None :
            # self.last_res_0=ResnetBlock(lv.shape[1], [1,1], self.with_debug_output, self.with_error_checking)
            # self.last_res_1=ResnetBlock(lv.shape[1], [1,2], self.with_debug_output, self.with_error_checking)
            self.last_conv=BnReluConv(lv.shape[1], 1, self.with_debug_output, self.with_error_checking)
        # lv, ls = self.last_res_0(lv,ls)
        # lv, ls = self.last_res_1(lv,ls)
        identity=lv
        lv, ls = self.last_conv(lv,ls)
        lv+=identity



        #slice conv will use a gather to get all features along a positions, if the features are a lot then gathering will use too much memory 
        #we bottleneck the features
        if self.last_bottleneck is None:
            # self.last_bn=torch.nn.BatchNorm1d(lv.shape[1]).to("cuda")
            self.last_bottleneck=torch.nn.Linear(lv.shape[1], 64, bias=True).to("cuda") 
        # lv=self.last_bn(lv)
        lv=self.last_bottleneck(lv)
        ls.set_values(lv)
        print("print values before slice_conv has shape", lv.shape)


  
        TIME_START("slice_conv")
        # sv=self.slice_conv(lv, ls, positions)
        sv=self.slice_deform(lv, ls, positions)
        TIME_END("slice_conv")

        self.per_point_features=sv
        # TIME_START("stepdown")
        s_final=self.stepdown(sv)
        # TIME_END("stepdown")


        logsoftmax=self.logsoftmax(s_final)

        return logsoftmax, s_final


