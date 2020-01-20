from callback import *

class StateCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self, phase, loss, **kwargs):
        phase.iter_nr+=1
        phase.samples_processed_this_epoch+=1
        phase.loss_acum_per_epoch+=loss

    def epoch_started(self, phase, **kwargs):
        phase.loss_acum_per_epoch=0.0

    def epoch_ended(self, phase, **kwargs):
        phase.epoch_nr+=1

    def phase_started(self, phase, **kwargs):
        phase.samples_processed_this_epoch=0

    def phase_ended(self, phase, **kwargs):
        phase.loader.reset()
