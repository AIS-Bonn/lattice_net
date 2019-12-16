from torch.optim import Optimizer

#adapted from https://github.com/jfzhang95/pytorch-deeplab-xception/blob/5ee873562f22b57b3691cebdc48720b3c65dedcb/utils/lr_scheduler.py
# lr = baselr * (1 - iter/maxiter) ^ 0.9

class PolyScheduler:
    """ Sets the learing rate of each parameter group by the one cycle learning rate policy
    proposed in https://arxiv.org/pdf/1708.07120.pdf. 

    It is recommended that you set the max_lr to be the learning rate that achieves 
    the lowest loss in the learning rate range test, and set min_lr to be 1/10 th of max_lr.

    So, the learning rate changes like min_lr -> max_lr -> min_lr -> final_lr, 
    where final_lr = min_lr * reduce_factor.

    Note: Currently only supports one parameter group.

    Args:
        optimizer:             (Optimizer) against which we apply this scheduler
        num_steps:             (int) of total number of steps/iterations
        lr_range:              (tuple) of min and max values of learning rate
        momentum_range:        (tuple) of min and max values of momentum
        annihilation_frac:     (float), fracion of steps to annihilate the learning rate
        reduce_factor:         (float), denotes the factor by which we annihilate the learning rate at the end
        last_step:             (int), denotes the last step. Set to -1 to start training from the beginning

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = OneCycleLR(optimizer, num_steps=num_steps, lr_range=(0.1, 1.))
        >>> for epoch in range(epochs):
        >>>     for step in train_dataloader:
        >>>         train(...)
        >>>         scheduler.step()

    Useful resources:
        https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6
        https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0
    """

    def __init__(self,
                 optimizer: Optimizer,
                 base_lr: float,
                 nr_iters_per_epochs: int,
                 nr_epochs_to_train:int ,
                 ):
        # Sanity check
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.base_lr=base_lr
        self.max_iter = nr_iters_per_epochs*nr_epochs_to_train
        self.last_step = -1 # so that when we start we start with current_step bein 0


    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


    def step(self):
        current_step = self.last_step + 1
        self.last_step = current_step

        lr = self.base_lr * (1 - current_step/self.max_iter) ** 0.9

        self.optimizer.param_groups[0]['lr'] = lr