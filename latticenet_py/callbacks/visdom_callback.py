from latticenet_py.callbacks.callback import *
from latticenet_py.callbacks.vis import *

class VisdomCallback(Callback):

    def __init__(self):
        self.vis=Vis("lnn", 8097)
        # self.iter_nr=0

    def after_forward_pass(self, phase, loss, loss_dice, lr, pred_softmax, target, cloud, **kwargs):
        self.vis.log(phase.iter_nr, loss, "loss_"+phase.name, "loss_"+phase.name, smooth=True)
        self.vis.log(phase.iter_nr, loss_dice, "loss_dice_"+phase.name, "loss_dice_"+phase.name, smooth=True)
        if phase.grad:
            self.vis.log(phase.iter_nr, lr, "lr", "lr", smooth=False)


    def epoch_ended(self, phase, **kwargs):
        mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        self.vis.log(phase.epoch_nr, mean_iou, "iou_"+phase.name, "iou_"+phase.name, smooth=False)