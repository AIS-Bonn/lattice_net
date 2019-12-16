#!/usr/bin/env python3.6

import os
import numpy as np
import sys
from easypbr  import *
from dataloaders import *
np.set_printoptions(threshold=sys.maxsize)

config_file="test_loader.cfg"

config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
view=Viewer.create(config_path) #first because it needs to init context
loader=DataLoaderShapeNetPartSeg(config_path)
loader.start()


while True:
    if(loader.has_data() ): 
        cloud=loader.get_cloud()
    #     # cloud = Mesh()
    #     # help(cloud)
        # print(cloud.V)
        Scene.show(cloud,"cloud")
    view.update()

