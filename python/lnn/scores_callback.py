from callback import *
from vis import *

class ScoresCallback(Callback):

    def __init__(self):
        self.vis=Vis("lnn", 8097)
        # self.iter_nr=0

    def after_forward_pass(self, phase, loss, **kwargs):
        self.vis.log(phase.iter_nr, loss.item(), "loss_"+phase.name, "loss_"+phase.name, smooth=True)
        # self.iter_nr+=1