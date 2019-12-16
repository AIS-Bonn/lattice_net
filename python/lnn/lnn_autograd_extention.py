#!/usr/bin/env python3.6


import torch
from torch.autograd import Function
from torch import Tensor

import sys
# sys.path.append('/media/rosu/Data/phd/c_ws/devel/lib/') #contains the modules of pycom
sys.path.append('/media/rosu/Data/phd/c_ws/build/surfel_renderer/') #contains the modules of pycom
from DataLoaderTest  import *
import numpy as np
import time
import math

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=5000)
torch.set_printoptions(threshold=50000)
# torch.set_printoptions(profile="full")


config_file="lnn_autograd_extension.cfg"


#Just to have something close to the macros we have in c++
def profiler_start(name):
    torch.cuda.synchronize()
    Profiler.start(name)
def profiler_end(name):
    torch.cuda.synchronize()
    Profiler.end(name)
TIME_START = lambda name: profiler_start(name)
TIME_END = lambda name: profiler_end(name)


def run():
    view=Viewer(config_file)
    loader=DataLoaderSemanticKitti(config_file)

    first_time=True
    iter_nr=0
    lnn=LNN.create(config_file)

    while True:
        view.update()

        if(loader.has_data()): 
            print("\n ITER----", iter_nr)
            cloud=loader.get_cloud()
            Scene.show(cloud,"cloud")

            lnn.forward()


            iter_nr=iter_nr+1
def main():
    run()



if __name__ == "__main__":
     main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')