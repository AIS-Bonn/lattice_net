from callback import *

class StateCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self, phase, **kwargs):
        phase.iter_nr+=1

    def epoch_ended(self, phase, **kwargs):
        phase.epoch_nr+=1

    # def phase_ended(self, main, **kwargs):
    #     print("got main")
