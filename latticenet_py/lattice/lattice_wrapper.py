import torch
# from torch import Tensor

# import sys
# sys.path.append('/media/rosu/Data/phd/c_ws/build/surfel_renderer/') #contains the modules of pycom
from latticenet  import Lattice
from easypbr  import Mesh
from easypbr  import Scene
import torch_scatter

#Pytorch functions require that the return values are torch Variables or a class derived from them so when we return a lattice we wrap it with this first
class LatticeWrapper(torch.Tensor): 
    @staticmethod
    def wrap(lattice):
        ls_wrap=LatticeWrapper()
        ls_wrap.lattice=lattice
        return ls_wrap

