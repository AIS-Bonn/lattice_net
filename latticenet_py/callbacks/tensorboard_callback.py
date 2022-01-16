from latticenet_py.callbacks.callback import *
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(Callback):

    def __init__(self, experiment_name):
        self.tensorboard_writer = SummaryWriter("tensorboard_logs/"+experiment_name)
        self.experiment_name=experiment_name
        

    def after_forward_pass(self, phase, loss, loss_dice, lr, pred_softmax, target, cloud, **kwargs):
        # self.vis.log(phase.iter_nr, loss, "loss_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        # self.vis.log(phase.iter_nr, loss_dice, "loss_dice_"+phase.name, "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        # if phase.grad:
            # self.vis.log(phase.iter_nr, lr, "lr", "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=30)
        self.tensorboard_writer.add_scalar('LatticeNet/' + phase.name + '/loss', loss, phase.iter_nr)


    def epoch_ended(self, phase, **kwargs):
        mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        self.tensorboard_writer.add_scalar('LatticeNet/' + phase.name + '/mean_iou', mean_iou, phase.epoch_nr)
        # self.vis.log(phase.epoch_nr, mean_iou, "iou_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=1)