from callbacks.callback import *
from callbacks.vis import *

class VisdomCallback(Callback):

    def __init__(self):
        self.vis=Vis("lnn", 8097)
        # self.iter_nr=0

    def after_forward_pass(self, phase, loss, lr, pred_softmax, target, cloud, **kwargs):
        self.vis.log(phase.iter_nr, loss.item(), "loss_"+phase.name, "loss_"+phase.name, smooth=True)
        if phase.grad:
            self.vis.log(phase.iter_nr, lr, "lr", "lr", smooth=False)


    def epoch_ended(self, phase, **kwargs):
        mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        self.vis.log(phase.epoch_nr, mean_iou, "iou_"+phase.name, "iou_"+phase.name, smooth=False)