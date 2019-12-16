

#BASED ON https://github.com/davidtvs/pytorch-lr-finder/blob/master/lr_finder.py

import torch
from torch.autograd import Function
from torch import Tensor

import sys
# sys.path.append('/media/rosu/Data/phd/c_ws/devel/lib/') #contains the modules of pycom
sys.path.append('/media/rosu/Data/phd/c_ws/build/surfel_renderer/') #contains the modules of pycom
from easypbr  import *
from latticenet  import *


from lattice_py import LatticePy
# from __future__ import print_function, with_statement, division
import copy
import os
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import torchnet
from scores import Scores

node_name="lnn"
port=8097
logger_lr_finder_train_loss = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_lr_finder_train_loss' }, port=port, env='train_'+node_name)
logger_lr_finder_test_loss = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_lr_finder_test_loss'}, port=port, env='train_'+node_name)
logger_lr_finder_test_iou = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_lr_finder_test_iou'}, port=port, env='train_'+node_name)


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]



class LRFinder(object):
    """Learning rate range test.
    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.
    Arguments:
        model (torch.nn.Module): wrapped model.
        optimizer (torch.optim.Optimizer): wrapped optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion (torch.nn.Module): wrapped loss function.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        memory_cache (boolean): if this flag is set to True, `state_dict` of model and
            optimizer will be cached in memory. Otherwise, they will be saved to files
            under the `cache_dir`.
        cache_dir (string): path for storing temporary files. If no path is specified,
            system-wide temporary directory is used.
            Notice that this parameter will be ignored if `memory_cache` is True.
    Example:
        >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai
    """
    def __init__(self, model_ctx, loader_train, loader_test, do_eval):
        self.model_ctx = model_ctx
        self.best_loss = None
        self.loader_train=loader_train
        self.loader_test=loader_test
        self.history = {"lr": [], "loss_train": [], "loss_test": []}
        self.do_eval=do_eval

    def cleanup(self):
        self.model_ctx.model=None
        self.model_ctx.optimizer=None
        self.model_ctx.schedule=None
        self.model_ctx.loss_fn=None
        self.loader_train.reset()
        self.loader_test.reset()

    def range_test(
        self,
        config_file,
        batch_size,
        start_lr=1e-5,
        end_lr=0.1,
        num_steps=100,
        step_mode="exp",
        # step_mode="linear",
        smooth_f=0.05,
        diverge_th=5,
    ):
        """Performs the learning rate range test.
        Arguments:
            train_loader (torch.utils.data.DataLoader): the training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional): if `None` the range test
                will only use the training loss. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.
        """
        # Reset test results
        # self.history = {"lr": [], "loss": []}
        self.best_loss = None

        lattice_to_splat=LatticePy()
        lattice_to_splat.create(config_file, "splated_lattice")
        first_time=True
        step_nr=0 # a step if for every batch
        samples_processed=0 #an iter is for every element of the batch
        scores=Scores()

        #set the model_ctx learnign rate and batch size and so on
        self.model_ctx.base_lr=start_lr
        self.model_ctx.max_lr=end_lr 
        self.model_ctx.batch_size=batch_size

        while True:
            if(self.loader_train.has_data() and self.loader_test.has_data() ): 
                cloud_train=self.loader_train.get_cloud()
                positions, values, target = self.model_ctx.prepare_cloud(cloud_train) #prepares the cloud for pytorch, returning tensors alredy in cuda
                #pred_softmax=self.model_ctx.forward(lattice_to_splat, positions, values, "train", start_lr, start_lr, cloud_train.m_label_mngr.nr_classes(), 1, 1 )
                pred_softmax=self.model_ctx.forward(lattice_to_splat, positions, values, "train", cloud_train.m_label_mngr.nr_classes(), self.loader_train.nr_samples()  )
                if(first_time):
                    print("overwriting the scheduler")
                    self.model_ctx.scheduler=None
                    first_time=False
                    self.model_ctx.model.train()        
                    # overwrite the schedulre from the model_ctx
                    if step_mode.lower() == "exp":
                        self.model_ctx.scheduler = ExponentialLR(self.model_ctx.optimizer, end_lr, num_steps)
                    elif step_mode.lower() == "linear":
                        raise ValueError("linear lr for some reason does not work at the moment ")
                        # self.model_ctx.scheduler = LinearLR(self.model_ctx.optimizer, end_lr, num_steps)
                    else:
                        raise ValueError("expected one of (exp, linear), got {}".format(step_mode))


                loss, finished_batch=self.model_ctx.backward(pred_softmax, target, cloud_train.m_label_mngr.get_idx_unlabeled(), samples_processed )



                #if we do an eval we run through all the test set and get the iou
                if self.do_eval:
                    samples_processed+=1
                    if finished_batch:
                        #update learning rate
                        self.model_ctx.scheduler.step()
                        cur_lr=self.model_ctx.scheduler.get_lr()[0]
                        print("doing eval")
                        self.model_ctx.model.eval()
                        while True:
                            if(self.loader_test.has_data()): 
                                cloud_test=self.loader_test.get_cloud()
                                positions, values, target = self.model_ctx.prepare_cloud(cloud_test) #prepares the cloud for pytorch, returning tensors alredy in cuda
                                #pred_softmax=self.model_ctx.forward(lattice_to_splat, positions, values, "eval", start_lr, start_lr, cloud_test.m_label_mngr.nr_classes(), 1, 1 )
                                pred_softmax=self.model_ctx.forward(lattice_to_splat, positions, values, "eval", cloud_train.m_label_mngr.nr_classes(), self.loader_test.nr_samples()  )
                                scores.accumulate_scores(pred_softmax, target)

                            if self.loader_test.is_finished():
                                self.loader_test.reset()
                                #print results
                                iou=scores.avg_class_iou()
                                logger_lr_finder_test_iou.log(cur_lr, iou, name='iou_test_at_lr')
                                scores.clear() #reset the nr of samples used for computing averages and so on to 0 so that we start accumulating anew on the next epoch
                                break
                        self.model_ctx.model.train()
                        step_nr+=1
                else:
                    #we plot the test loss because it's easier to ge than the iou
                    self.model_ctx.model.eval()
                    cloud_test=self.loader_test.get_cloud()
                    positions, values, target = self.model_ctx.prepare_cloud(cloud_test) #prepares the cloud for pytorch, returning tensors alredy in cuda
                    #pred_softmax=self.model_ctx.forward(lattice_to_splat, positions, values, "eval", start_lr, start_lr, cloud_test.m_label_mngr.nr_classes(), 1, 1 )
                    pred_softmax=self.model_ctx.forward(lattice_to_splat, positions, values, "eval", cloud_train.m_label_mngr.nr_classes(), self.loader_test.nr_samples()  )
                    loss_test_per_batch, finished_batch=self.model_ctx.loss(pred_softmax, target, cloud_test.m_label_mngr.get_idx_unlabeled(), samples_processed )
                    self.model_ctx.model.train()


                    samples_processed+=1

                    if finished_batch:
                        #update learning rate
                        self.model_ctx.scheduler.step()
                        cur_lr=self.model_ctx.scheduler.get_lr()[0]
                        self.history["lr"].append(cur_lr)
                        # Track the best loss and smooth it if smooth_f is specified
                        if step_nr == 0:
                            self.best_loss = loss
                        else:
                            if smooth_f > 0:
                                loss = smooth_f * loss + (1 - smooth_f) * self.history["loss_train"][-1]
                                loss_test_per_batch = smooth_f * loss_test_per_batch + (1 - smooth_f) * self.history["loss_train"][-1]
                            if loss < self.best_loss:
                                self.best_loss = loss
                        self.history["loss_train"].append(loss)
                        self.history["loss_test"].append(loss_test_per_batch)

                        #plot
                        logger_lr_finder_train_loss.log(cur_lr, loss, name='loss_train_at_lr')
                        logger_lr_finder_test_loss.log(cur_lr, loss_test_per_batch, name='loss_test_at_lr')
                        print("cur_lr is ", cur_lr, "loss is ", loss)
                        # logger_lr_finder.log(loss, cur_lr, name='loss_at_lr')
                        # 






                        # Check if the loss has diverged; if it has, stop the test
                        # self.history["loss"].append(loss)
                        if loss > diverge_th * self.best_loss:
                            print("Stopping early, the loss has diverged")
                            self.cleanup()
                            break

                        step_nr+=1
                        if(step_nr>num_steps):
                            self.cleanup()
                            break

                if self.loader_train.is_finished():
                    self.loader_train.reset()
                if self.loader_test.is_finished():
                    self.loader_test.reset()
