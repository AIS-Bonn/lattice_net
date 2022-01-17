import torch

import sys
import os
# import warnings
from termcolor import colored



# from latticenet_py.lattice.lattice_py import LatticePy
from latticenet_py.lattice.lattice_funcs import *
from latticenet_py.lattice.lattice_modules import *

from functools import reduce
from torch.nn.modules.module import _addindent


def prepare_cloud(cloud, model_params):
       

    with torch.set_grad_enabled(False):

        if model_params.positions_mode()=="xyz":
            positions_tensor=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
        elif model_params.positions_mode()=="xyz+rgb":
            xyz_tensor=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
            rgb_tensor=torch.from_numpy(cloud.C.copy() ).float().to("cuda")
            positions_tensor=torch.cat((xyz_tensor,rgb_tensor),1)
        elif model_params.positions_mode()=="xyz+intensity":
            xyz_tensor=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
            intensity_tensor=torch.from_numpy(cloud.I.copy() ).float().to("cuda")
            positions_tensor=torch.cat((xyz_tensor,intensity_tensor),1)
        else:
            err="positions mode of ", model_params.positions_mode() , " not implemented"
            sys.exit(err)


        if model_params.values_mode()=="none":
            values_tensor=torch.zeros(positions_tensor.shape[0], 1).to("cuda") #not really necessary but at the moment I have no way of passing an empty value array
        elif model_params.values_mode()=="intensity":
            values_tensor=torch.from_numpy(cloud.I.copy() ).float().to("cuda")
        elif model_params.values_mode()=="rgb":
            values_tensor=torch.from_numpy(cloud.C.copy() ).float().to("cuda")
        elif model_params.values_mode()=="rgb+height":
            rgb_tensor=torch.from_numpy(cloud.C).float().to("cuda")
            height_tensor=torch.from_numpy(cloud.V[:,1].copy() ).unsqueeze(1).float().to("cuda")
            values_tensor=torch.cat((rgb_tensor,height_tensor),1)
        elif model_params.values_mode()=="rgb+xyz":
            rgb_tensor=torch.from_numpy(cloud.C.copy() ).float().to("cuda")
            xyz_tensor=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
            values_tensor=torch.cat((rgb_tensor,xyz_tensor),1)
        elif model_params.values_mode()=="height":
            height_tensor=torch.from_numpy(cloud.V[:,1].copy() ).unsqueeze(1).float().to("cuda")
            values_tensor=height_tensor
        elif model_params.values_mode()=="xyz":
            xyz_tensor=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
            values_tensor=xyz_tensor
        else:
            err="values mode of ", model_params.values_mode() , " not implemented"
            sys.exit(err)


        target=cloud.L_gt
        target_tensor=torch.from_numpy(target.copy() ).long().squeeze(1).to("cuda").squeeze(0)

    return positions_tensor, values_tensor, target_tensor



class LNN(torch.nn.Module):
    def __init__(self, nr_classes, model_params):
        super(LNN, self).__init__()
        self.nr_classes=nr_classes

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
        # experiment=model_params.experiment()
        #check that the experiment has a valid string
        # valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        # if experiment not in valid_experiment:
            # err = "Experiment " + experiment + " is not valid"
            # sys.exit(err)





        self.distribute=DistributeLatticeModule() 
        self.pointnet_channels_per_layer=model_params.pointnet_channels_per_layer()
        self.start_nr_filters=model_params.pointnet_start_nr_channels()
        print("pointnt_channels_per_layer is ", self.pointnet_channels_per_layer)
        self.point_net=PointNetModule( self.pointnet_channels_per_layer, self.start_nr_filters)  

        experiment="none"


        #####################
        # Downsampling path #
        #####################
        self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        self.maxpool_list=torch.nn.ModuleList([])
        corsenings_channel_counts = []
        skip_connection_channel_counts = []
        cur_channels_count=self.start_nr_filters
        for i in range(self.nr_downsamples):
            
            #create the resnet blocks
            self.resnet_blocks_per_down_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                if i<self.nr_levels_down_with_normal_resnet:
                    print("adding down_resnet_block with nr of filters", cur_channels_count )
                    should_use_dropout=False
                    print("adding down_resnet_block with dropout", should_use_dropout )
                    self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock(cur_channels_count, cur_channels_count,  [1,1], [False,False], should_use_dropout) )
                else:
                    print("adding down_bottleneck_block with nr of filters", cur_channels_count )
                    self.resnet_blocks_per_down_lvl_list[i].append( BottleneckBlock(cur_channels_count, cur_channels_count, [False,False,False]) )
                    # self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock2(cur_channels_count, cur_channels_count,  [1,1], [True,True], False)  )
            skip_connection_channel_counts.append(cur_channels_count)
            nr_channels_after_coarsening=int(cur_channels_count*2*compression_factor)
            print("adding bnReluCorsen which outputs nr of channels ", nr_channels_after_coarsening )
            # self.coarsens_list.append( GnReluCoarsen(cur_channels_count, nr_channels_after_coarsening)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            self.coarsens_list.append( CoarsenAct(cur_channels_count, nr_channels_after_coarsening)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            cur_channels_count=nr_channels_after_coarsening
            corsenings_channel_counts.append(cur_channels_count)

        #####################
        #     Bottleneck    #
        #####################
        self.resnet_blocks_bottleneck=torch.nn.ModuleList([])
        for j in range(self.nr_blocks_bottleneck):
                print("adding bottleneck_resnet_block with nr of filters", cur_channels_count )
                self.resnet_blocks_bottleneck.append( BottleneckBlock(cur_channels_count, cur_channels_count, [False,False,False]) )
                # self.resnet_blocks_bottleneck.append( ResnetBlock2(cur_channels_count, cur_channels_count,  [1,1], [True,True], False)  )

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

            # if the finefy is the deepest one int the network then it just divides by 2 the nr of channels because we know it didnt get as input two concatet tensors
            nr_chanels_finefy=int(cur_channels_count/2)

            #do it with finefy
            print("adding bnReluFinefy which outputs nr of channels ", nr_chanels_finefy )
            self.finefy_list.append( GnReluFinefy(cur_channels_count, nr_chanels_finefy ))
            # self.finefy_list.append( FinefyAct(cur_channels_count, nr_chanels_finefy ))

            #after finefy we do a concat with the skip connection so the number of channels doubles
            if self.do_concat_for_vertical_connection:
                cur_channels_count=nr_chanels_skip_connection+nr_chanels_finefy
            else:
                cur_channels_count=nr_chanels_skip_connection

            self.resnet_blocks_per_up_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_up_stage[i]):
                is_last_conv=j==self.nr_blocks_up_stage[i]-1 and i==self.nr_downsamples-1 #the last conv of the last upsample is followed by a slice and not a bn, therefore we need a bias
                if i>=self.nr_downsamples-self.nr_levels_up_with_normal_resnet:
                    print("adding up_resnet_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock(cur_channels_count, cur_channels_count,  [1,1], [False,is_last_conv], False) )
                    # self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock2(cur_channels_count, cur_channels_count,  [1,1], [True,True], False) )
                else:
                    print("adding up_bottleneck_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( BottleneckBlock(cur_channels_count, cur_channels_count, [False,False,is_last_conv] ) )
                    # self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock2(cur_channels_count, cur_channels_count,  [1,1], [True,True], False) )

        self.slice_fast_cuda=SliceFastCUDALatticeModule(in_channels=cur_channels_count, nr_classes=nr_classes, dropout_prob=dropout_last_layer, experiment=experiment)
        # self.slice=SliceLatticeModule()
        # self.classify=Conv1x1(out_channels=nr_classes, bias=True)
       
        self.logsoftmax=torch.nn.LogSoftmax(dim=1)


        if experiment!="none":
            warn="USING EXPERIMENT " + experiment
            print(colored("-------------------------------", 'yellow'))
            print(colored(warn, 'yellow'))
            print(colored("-------------------------------", 'yellow'))

    def forward(self, ls, positions, values):

        with torch.set_grad_enabled(False):
            ls, distributed, indices, weights=self.distribute(ls, positions, values)

        lv, ls=self.point_net(ls, distributed, indices)


        
        fine_structures_list=[]
        fine_values_list=[]
        # TIME_START("down_path")
        for i in range(self.nr_downsamples):

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                # print("start downsample stage ", i , " resnet block ", j, "lv has shape", lv.shape, " ls has val dim", ls.val_dim() )
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(lv)

            #now we do a downsample
            # print("start coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )
            lv, ls = self.coarsens_list[i] ( lv, ls)
            # print( "finished coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )

        # TIME_END("down_path")

        # #bottleneck
        for j in range(self.nr_blocks_bottleneck):
            # print("bottleneck stage", j,  "lv has shape", lv.shape, "ls has val_dim", ls.val_dim()  )
            lv, ls = self.resnet_blocks_bottleneck[j] ( lv, ls) 


        #upsample (we start from the bottom of the U-net, so the upsampling that is closest to the blottlenck)
        # TIME_START("up_path")
        for i in range(self.nr_downsamples):

            fine_values=fine_values_list.pop()
            fine_structure=fine_structures_list.pop()


            #finefy
            # print("start finefy stage", i,  "lv has shape", lv.shape, "ls has val_dim ", ls.val_dim(),  "fine strcture has val dim ", fine_structure.val_dim() )
            lv, ls = self.finefy_list[i] ( lv, ls, fine_structure  )

            #concat or adding for the vertical connection
            if self.do_concat_for_vertical_connection: 
                lv=torch.cat((lv, fine_values ),1)
            else:
                lv+=fine_values

            #resnet blocks
            for j in range(self.nr_blocks_up_stage[i]):
                # print("start resnet block in upstage", i, "lv has shape", lv.shape, "ls has val dim" , ls.val_dim() )
                lv, ls = self.resnet_blocks_per_up_lvl_list[i][j] ( lv, ls) 
        # TIME_END("up_path")



        sv =self.slice_fast_cuda(lv, ls, positions, indices, weights)
        # sv =self.slice(lv, ls, positions, indices, weights)
        # sv=self.classify(sv)


        logsoftmax=self.logsoftmax(sv)


        return logsoftmax, sv
        # return logsoftmax, s_final



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

            # main_str += ')'
            # if file is sys.stderr:
            #     main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
            # else:
            #     main_str += ', {:,} params'.format(total_params)
            # return main_str, total_params

            main_str += ')'
            if file is sys.stderr:
                main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
                for name, p in model._parameters.items():
                    if hasattr(p, 'grad'):
                        if(p.grad==None):
                            # print("p has no grad", name)
                            main_str+="p no grad"
                        else:
                            # print("p has gradnorm ", name ,p.grad.norm() )
                            main_str+= "\n" + name + " p has grad norm " + str(p.grad.norm())
            else:
                main_str += ', {:,} params'.format(total_params)
            return main_str, total_params

        string, count = repr(self)
        if file is not None:
            print(string, file=file)
        return count


