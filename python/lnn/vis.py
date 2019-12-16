import torchnet
import numpy as np
import torch

node_name="lnn"
port=8097
logger_iou = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_iou'}, port=port, env='train_'+node_name)


class Vis():
    def __init__(self):
        self.port=8097
        self.env="train_lnn"

        self.name_dict=dict()
        self.logger_dict=dict()
        self.exp_alpha=0.03 #the lower the value the smoother the plot is

    def update_val(self, val, name):
        if name not in self.name_dict:
            self.name_dict[name]=val
        else:
            self.name_dict[name]= self.name_dict[name] + self.exp_alpha*(val-self.name_dict[name])
        
        return self.name_dict[name]

    def update_logger(self, x_axis, val, name_window, name_plot):
        if name_window not in self.logger_dict:
            self.logger_dict[name_window]=torchnet.logger.VisdomPlotLogger('line', opts={'title': name_window}, port=self.port, env=self.env)
        else:
            self.logger_dict[name_window].log(x_axis, val, name=name_plot)

    def log(self, x_axis, val, name_window, name_plot):
        smoothed_val=self.update_val(val,name_plot)
        self.update_logger(x_axis, smoothed_val, name_window, name_plot)

