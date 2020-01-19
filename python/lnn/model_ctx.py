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
import numpy as np
import time
import math
import torch_scatter
from lattice_py import LatticePy
from lattice_funcs import *
from lattice_modules import *
import visdom
import torchnet
from adabound import AdaBound
from adamw import AdamW
from onecyclelr import OneCycleLR
from poly_scheduler import PolyScheduler
from focalloss import FocalLoss
from diceloss import GeneralizedSoftDiceLoss
from vis import Vis
# from diceloss import dice_loss
from models import *
from functools import reduce
from torch.nn.modules.module import _addindent
from dice import *
from jaccard import *
from lovasz import *


class ModelCtx():
    def __init__(self, base_lr=None, max_lr=None, weight_decay=0.0, batch_size=None, nr_epochs_per_half_cycle=None, exponential_gamma=None, model_params=None, with_debug_output=True, with_error_checking=True):
        self.clear()      
        

        self.base_lr=base_lr
        self.max_lr=max_lr
        self.weight_decay=weight_decay
        self.batch_size=batch_size
        self.nr_epochs_per_half_cycle=nr_epochs_per_half_cycle
        self.exponential_gamma=exponential_gamma
        self.model_params=model_params

        self.with_debug_output=with_debug_output
        self.with_error_checking=with_error_checking

    def clear(self):
        self.model=None
        self.optimizer=None
        self.scheduler=None
        self.loss_fn=None
        self.secondary_fn=None
        self.loss_per_batch=0

    #needed because the cycle lr from pytorch with exp_range only scales the max_lr at every iteration but I want it scaled at each cycle
    def _exp_range_scale_fn(self, x):
        # gamma=0.9
        gamma=self.exponential_gamma
        return gamma**(x)

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
        print("backgorund idx", background_idx)
        class_weights[background_idx]=0.00000001

        return class_weights




    
                # pred_softmax=self.model_ctx.forward(lattice_to_splat, positions, values, cloud.m_label_mngr.nr_classes(), start_lr )
    def forward(self, lattice_to_splat, positions, values, mode, nr_classes, nr_samples_train):
        if(self.model==None): #first cloud we have we get also the nr of classes and also we run one forward pass to initialize all the parameters
            # self.model=LNN(nr_classes, self.with_debug_output, self.with_error_checking).to("cuda")
            # self.model=LNN_tiramisu(nr_classes, self.with_debug_output, self.with_error_checking).to("cuda")
            # self.model=LNN_unet(nr_classes, self.with_debug_output, self.with_error_checking).to("cuda")
            # self.model=LNN_skippy(nr_classes, self.with_debug_output, self.with_error_checking).to("cuda")
            # self.model=LNN_skippy_v2(nr_classes, self.with_debug_output, self.with_error_checking).to("cuda")
            # self.model=LNN_unet_v2(nr_classes, self.with_debug_output, self.with_error_checking).to("cuda")
            
            # skippy uses a lot of time an memory on the pointnet part, we want to make it faster
            self.model=LNN_skippy_efficient(nr_classes, self.model_params, self.with_debug_output, self.with_error_checking).to("cuda")
            # self.model=LNN_jpu(nr_classes, self.with_debug_output, self.with_error_checking).to("cuda")
            if mode=="train":
                self.model.train()
            else:
                self.model.eval()

            pred_softmax, pred_raw, delta_weight_error_sum=self.model(lattice_to_splat, positions, values)

            if mode=="train":
                print("setting lr to ", self.max_lr)
                # params_with_no_wd=torch.nn.ParameterList()
                # params_with_with_wd=torch.nn.ParameterList()
                # for name, param in self.model.named_parameters():
                #     print("name is ", name)
                #     if "residual_gate" in name:
                #         print("setting with no weight decay")
                #         params_with_no_wd.append(param)
                #     else:
                #         params_with_no_wd.append(param)


                # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=0.0) #amsgrad seemd to actually make it worse
                # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=1e-4) #amsgrad seemd to actually make it worse
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay, amsgrad=True) #amsgrad seemd to actually make it worse
                # self.optimizer = torch.optim.AdamW(  [  {'params': params_with_with_wd },
                #                          {'params': params_with_no_wd, 'weight_decay': 0.0}
                #                          ], lr=self.base_lr, weight_decay=self.weight_decay ) #amsgrad seemd to actually make it worse
                # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9, nesterov=True, weight_decay=self.weight_decay)
                # self.optimizer = AdaBound(self.model.parameters(), lr=5e-4, final_lr=0.1)
                self.optimizer.zero_grad()
                # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1) #after x epochs, multiply learnign rate with gamma
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True, factor=0.1)
                # self.scheduler = PolyScheduler(self.optimizer, base_lr=self.base_lr, nr_iters_per_epochs=nr_samples_train, nr_epochs_to_train=self.nr_epochs_per_half_cycle)
                # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=5e-5/batch_size, max_lr=5e-2/batch_size, step_size_up=500 )
                step_size=nr_samples_train/self.batch_size * self.nr_epochs_per_half_cycle #from the paper https://arxiv.org/pdf/1506.01186.pdf. Specified how many epochs will be affected by the step size increment or decrement in the learning rate. Probably for very smalle datasets quite a lot of epochs should eb affected as they are not a lot of gradient updates for each epoch
                print("nr_sample is", nr_samples_train)
                print("step size is", step_size)
                # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=5e-5/batch_size, max_lr=1e-1/batch_size, step_size_up=step_size, mode="triangular2" )
                # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=5e-5/batch_size, max_lr=0.3e-1/batch_size, step_size_up=step_size, mode="exp_range", scale_mode="cycle", scale_fn=self._exp_range_scale_fn)
                # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.base_lr, max_lr=self.max_lr, step_size_up=step_size, mode="exp_range", scale_mode="cycle", scale_fn=self._exp_range_scale_fn)
                # self.scheduler = OneCycleLR(self.optimizer, num_steps=step_size, lr_range=(self.base_lr, self.max_lr) )

            elif mode=="eval":
                pass
            else: 
                err = "mode " + mode + "is not recognized"
                sys.exit("mode ", err)

        else:
            TIME_START("forward")
            pred_softmax, pred_raw, delta_weight_error_sum=self.model(lattice_to_splat, positions, values)
            TIME_END("forward")
        # pred_softmax=pred_softmax.squeeze(0)
        # pred_raw=pred_raw.squeeze(0)

        return pred_softmax, pred_raw, delta_weight_error_sum

    def backward(self, pred_softmax, pred_raw, target, delta_weight_error_sum, background_idx, class_frequencies, iter_nr):
        # print("pred_raw has shape ", pred_raw.shape)
        # print("pred_softmax has shape ", pred_softmax.shape)
        # print("target has shape ", target.shape)
        if self.loss_fn==None:
            class_weights_tensor=self.compute_class_weights(class_frequencies, background_idx)
            # class_frequencies_tensor=torch.from_numpy(class_frequencies).float().to("cuda")
            # print("class_frequencies is ", class_frequencies_tensor)
            # class_weights_tensor=1.0/class_frequencies_tensor
            print("class weights is " , class_weights_tensor)
            # sys.exit("what")
            # self.loss_fn=torch.nn.NLLLoss(ignore_index=background_idx, weight=class_weights_tensor) #TODO make it dynamic depending on the labelmngr.get_background_idx
            # self.loss_fn=torch.nn.NLLLoss(ignore_index=background_idx) #takes about 2ms for calculating the loss
            # self.loss_fn=GeneralizedSoftDiceLoss(ignore_index=background_idx, weight=class_weights_tensor) #TODO make it dynamic depending on the labelmngr.get_background_idx
            self.loss_fn=GeneralizedSoftDiceLoss(ignore_index=background_idx) #takes about 20ms for calculating the loss...
            # self.loss_fn=DiceLoss(mode="multiclass", smooth=0.0) #takes about 20ms for calculating the loss...
            # self.loss_fn=JaccardLoss(mode="multiclass", smooth=1.0) #works best by itself
            # self.loss_fn=LovaszLoss(ignore=background_id)
            # self.loss_fn=FocalLoss(gamma=2.0, ignore_index=background_idx, weight=class_weights_tensor) 
            # self.loss_fn=FocalLoss(gamma=2.0, ignore_index=background_idx) 

            #these guys use a combination of cross entropy and dice so we try it too https://arxiv.org/pdf/1809.10486.pdf
            self.secondary_fn=torch.nn.NLLLoss(ignore_index=background_idx) #takes about 2ms for calculating the loss
        # Compute and print loss.
        if isinstance(self.loss_fn, FocalLoss):
            loss = self.loss_fn(pred_raw, target) #acts as a cross entropy loss so we need the raw prediction
        else:
            # print("pred_softmax has shape ", pred_softmax.shape)
            # print("target has shape ", target.shape)
            # sys.exit("deug")

            #for dice loss, jacard and lovasz we need to rehsape
            # :param y_pred: NxCxHxW
            # :param y_true: NxHxW 
            # pred_softmax=pred_softmax.unsqueeze(2).unsqueeze(3)
            # target=target.unsqueeze(1).unsqueeze(2)

            loss = self.loss_fn(pred_softmax, target)
            print("loss dice is", loss.item() )
            loss += self.secondary_fn(pred_softmax, target)
            # loss = self.secondary_fn(pred_softmax, target)
        loss+=0.1*delta_weight_error_sum
        loss/=self.batch_size

        

        # print("\t loss is", loss.item())
        # print("summing ", loss.item())
        self.loss_per_batch+=loss.item()
        # print("loss_per_batch is ", self.loss_per_batch)

        # Backward pass: compute gradient of the loss with respect to model parameters
        TIME_START("backward")
        loss.backward()
        TIME_END("backward")
        # Calling the step function on an Optimizer makes an update to its parameters
        finished_batch=False
        loss_per_batch_local=self.loss_per_batch #save it into a local variable so we can put the other one to zero
        if( (iter_nr+1) %self.batch_size==0):
            if self.with_debug_output:
                print("updating model!")
                self.print_grad_norm()
            # sys.exit("debug")

            # grad_clip=0.5*self.batch_size #thre gradients sum per batch and terefore the norm would increase as we increase the batch size
            # torch.nn.utils.clip_grad_norm(self.model.parameters(),grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_per_batch=0
            # print("loss per batch is ", self.loss_per_batch)
            print("loss per batch local is ", loss_per_batch_local)
            if(isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR) or isinstance(self.scheduler, OneCycleLR) or isinstance(self.scheduler, PolyScheduler) ):
                # print("updating scheduler")
                self.scheduler.step() #for the cyclic lr
            finished_batch=True
            print("finished batch")
        # print("backwards did an update and is returning", did_update)
        # print("=============returning loss_per_batch_local", loss_per_batch_local)
        return loss_per_batch_local, finished_batch

    def loss(self, pred_softmax, pred_raw, target, background_idx, class_frequencies, iter_nr):
        if self.loss_fn==None:
            # class_frequencies_tensor=torch.from_numpy(class_frequencies).float().to("cuda")
            # class_weights_tensor=1.0/class_frequencies_tensor
            # self.loss_fn=torch.nn.NLLLoss(ignore_index=background_idx) #takes about 2ms for calculating the loss

            # self.loss_fn=torch.nn.CrossEntropyLoss(ignore_index=background_idx, weight=class_weights_tensor) #TODO make it dynamic depending on the labelmngr.get_background_idx
            self.loss_fn=GeneralizedSoftDiceLoss(ignore_index=background_idx) #takes about 20ms for calculating the loss...
            # self.secondary_fn=torch.nn.NLLLoss(ignore_index=background_idx) #takes about 2ms for calculating the loss
        # Compute and print loss.
        if isinstance(self.loss_fn, FocalLoss):
            loss = self.loss_fn(pred_raw, target) #acts as a cross entropy loss so we need the raw prediction
        else:
            # pred_softmax=pred_softmax.unsqueeze(2).unsqueeze(3)
            # target=target.unsqueeze(1).unsqueeze(2)
            loss = self.loss_fn(pred_softmax, target)
            # loss += self.secondary_fn(pred_softmax, target)
        loss/=self.batch_size
        # loss = self.loss_fn(pred_softmax, target)
        # print("\t loss is", loss.item())
        self.loss_per_batch+=loss.item()

        finished_batch=False
        loss_per_batch_local=self.loss_per_batch #save it into a local variable so we can put the other one to zero
        if( (iter_nr+1)%self.batch_size==0):
            self.loss_per_batch=0
            finished_batch=True
            print("finished batch")
        return loss_per_batch_local, finished_batch

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


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

        if self.model_params.positions_mode()=="xyz":
            positions_tensor=torch.from_numpy(cloud.V).unsqueeze(0).float().to("cuda")
        elif self.model_params.positions_mode()=="xyz+rgb":
            xyz_tensor=torch.from_numpy(cloud.V).unsqueeze(0).float().to("cuda")
            rgb_tensor=torch.from_numpy(cloud.C).unsqueeze(0).float().to("cuda")
            positions_tensor=torch.cat((xyz_tensor,rgb_tensor),2)
        elif self.model_params.positions_mode()=="xyz+intensity":
            xyz_tensor=torch.from_numpy(cloud.V).unsqueeze(0).float().to("cuda")
            intensity_tensor=torch.from_numpy(cloud.I).unsqueeze(0).float().to("cuda")
            positions_tensor=torch.cat((xyz_tensor,intensity_tensor),2)
        else:
            err="positions mode of ", self.model_params.positions_mode() , " not implemented"
            sys.exit(err)


        if self.model_params.values_mode()=="none":
            values_tensor=torch.zeros(1, positions_tensor.shape[1], 1) #not really necessary but at the moment I have no way of passing an empty value array
        elif self.model_params.values_mode()=="intensity":
            values_tensor=torch.from_numpy(cloud.I).unsqueeze(0).float().to("cuda")
        elif self.model_params.values_mode()=="rgb":
            values_tensor=torch.from_numpy(cloud.C).unsqueeze(0).float().to("cuda")
        elif self.model_params.values_mode()=="rgb+height":
            rgb_tensor=torch.from_numpy(cloud.C).unsqueeze(0).float().to("cuda")
            height_tensor=torch.from_numpy(cloud.V[:,1]).unsqueeze(0).unsqueeze(2).float().to("cuda")
            values_tensor=torch.cat((rgb_tensor,height_tensor),2)
        elif self.model_params.values_mode()=="rgb+xyz":
            rgb_tensor=torch.from_numpy(cloud.C).unsqueeze(0).float().to("cuda")
            xyz_tensor=torch.from_numpy(cloud.V).unsqueeze(0).float().to("cuda")
            values_tensor=torch.cat((rgb_tensor,xyz_tensor),2)
        elif self.model_params.values_mode()=="height":
            height_tensor=torch.from_numpy(cloud.V[:,1]).unsqueeze(0).unsqueeze(2).float().to("cuda")
            values_tensor=height_tensor
        elif self.model_params.values_mode()=="xyz":
            xyz_tensor=torch.from_numpy(cloud.V).unsqueeze(0).float().to("cuda")
            values_tensor=xyz_tensor
        else:
            err="values mode of ", self.model_params.values_mode() , " not implemented"
            sys.exit(err)



        target=cloud.L_gt
        target_tensor=torch.from_numpy(target).long().squeeze(1).to("cuda").squeeze(0)
        # print("maximum class idx is ", target_tensor.max() )
        # TIME_END("prepare")

        return positions_tensor, values_tensor, target_tensor

    #returns the number of trainable parameters in the network WARNING: it should be run only after the first forwards pass because only then will all the parameters be instantiated
    def nr_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def print_params(self):
        #print params 
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print (name, param.data)

    def print_grad_norm(self):
        #check norm gradient
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print (name, param.grad.data.norm().item())

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

        string, count = repr(self.model)
        if file is not None:
            print(string, file=file)
        return count

    #to deal with batch norm causing the evaluation to be a lot worse when model.eval. we use the answer from https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/16?u=radu_alexandru_rosu
    def set_batch_norm(self, val):
        def repr(model, val):
            for key, module in model._modules.items():
                
                print(module)
                if type(module)==torch.nn.BatchNorm1d:
                    print("setting track running stats to ", val)
                    module.track_running_stats = val

                #keep traversin recursivelly
                repr(module,val)
        repr(self.model, val)
        
    def nr_convolutional_layers(self):
        def repr(model):
            local_nr_conv_layers=0
            for key, module in model._modules.items():
                
                # print(module)
                if type(module)==ConvLatticeModule:
                    # print("nr conv layers increase, it is now ", nr_conv_layers)
                    local_nr_conv_layers+=1
            
                local_nr_conv_layers+=repr(module)
            return local_nr_conv_layers
        nr_conv_layers=0
        nr_conv_layers+=repr(self.model)
        return nr_conv_layers


    def per_point_features(self):
        return self.model.per_point_features

0
