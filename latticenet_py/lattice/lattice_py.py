import torch
# from torch import Tensor

# import sys
# sys.path.append('/media/rosu/Data/phd/c_ws/build/surfel_renderer/') #contains the modules of pycom
from latticenet  import Lattice
from easypbr  import Mesh
from easypbr  import Scene
import torch_scatter



# # https://discuss.pytorch.org/t/subclassing-torch-tensor/23754
# class LatticePy(torch._C._TensorBase): 
class LatticePy(torch.Tensor): 
# class LatticePy(torch.Tensor): 
# class LatticePy(torch.autograd.Variable): 

    def create(self, config_file, name):
        self.lattice=Lattice.create(config_file, name)

    def begin_splat(self):
        self.lattice.begin_splat()
    def begin_splat_modify_only_values(self):
        self.lattice.begin_splat_modify_only_values()
    def splat_standalone(self, positions, values ):
       return self.lattice.splat_standalone(positions, values )
    def distribute(self, positions, values):
        return self.lattice.distribute(positions, values)
    # def blur_standalone(self):
    #     blurred_lattice=self.lattice.blur_standalone()
    #     blurred_lattice_py=LatticePy()
    #     blurred_lattice_py.lattice=blurred_lattice

    #     return blurred_lattice_py
    # def convolve_standalone(self, filter_bank):
    #     convolved_lattice=self.lattice.convolve_standalone(filter_bank)
    #     convolved_lattice_py=LatticePy()
    #     convolved_lattice_py.lattice=convolved_lattice

    #     #this tensor needs to have a shape so that the sizes of the forwards and backard pass match
    #     values=selfvalues()
    #     self.data=values
    #     values=convolved_lattice_py.values()
    #     convolved_lattice_py.data=values

    #     return convolved_lattice_py

    # def convolve_im2row_standalone(self, filter_bank, dilation, with_homogeneous_coord):
    #     # print("running convolve")
    #     convolved_lattice=self.lattice.convolve_im2row_standalone(filter_bank, dilation, with_homogeneous_coord)
    #     # print("finished convolve")
    #     convolved_lattice_py=LatticePy()
    #     convolved_lattice_py.lattice=convolved_lattice

    #     return convolved_lattice_py

    # def depthwise_convolve(self, filter_bank, dilation, lattice_neighbours=None, use_center_vertex_from_lattice_neighbours=True, flip_neighbours=False):
    #     # print("running convolve")
    #     # if lattice_neighbours is not None:
    #         # print("lattice py doing convolve with lattic eneighbours")
    #     # else:
    #         # print("lattice_py, doing convolve with lattice neighbous being none ")

    #     if lattice_neighbours is None:
    #         convolved_lattice=self.lattice.depthwise_convolve(filter_bank, dilation, self.lattice, use_center_vertex_from_lattice_neighbours, flip_neighbours)
    #     else:
    #         convolved_lattice=self.lattice.depthwise_convolve(filter_bank, dilation, lattice_neighbours.lattice, use_center_vertex_from_lattice_neighbours, flip_neighbours)
    #     # print("finished convolve")
    #     convolved_lattice_py=LatticePy()
    #     convolved_lattice_py.lattice=convolved_lattice

    #     return convolved_lattice_py
    #     # return LatticePy(blurred_lattice)

    def convolve_im2row_standalone(self, filter_bank, dilation, lattice_neighbours=None, use_center_vertex_from_lattice_neighbours=True, flip_neighbours=False):
        # print("running convolve")
        # if lattice_neighbours is not None:
            # print("lattice py doing convolve with lattic eneighbours")
        # else:
            # print("lattice_py, doing convolve with lattice neighbous being none ")

        if lattice_neighbours is None:
            convolved_lattice=self.lattice.convolve_im2row_standalone(filter_bank, dilation, self.lattice, use_center_vertex_from_lattice_neighbours, flip_neighbours)
        else:
            convolved_lattice=self.lattice.convolve_im2row_standalone(filter_bank, dilation, lattice_neighbours.lattice, use_center_vertex_from_lattice_neighbours, flip_neighbours)
        # print("finished convolve")
        convolved_lattice_py=LatticePy()
        convolved_lattice_py.lattice=convolved_lattice

        return convolved_lattice_py
        # return LatticePy(blurred_lattice)

    def im2row(self, filter_extent, lattice_neighbours=None,  dilation=1, use_center_vertex_from_lattice_neighbours=True, flip_neighbours=False):


        if lattice_neighbours is not None:
            lattice_rowified = self.lattice.im2row(lattice_neighbours.lattice, filter_extent, dilation, use_center_vertex_from_lattice_neighbours,  flip_neighbours)
        else:
            lattice_rowified = self.lattice.im2row(self.lattice, filter_extent, dilation, use_center_vertex_from_lattice_neighbours,  flip_neighbours)

        return lattice_rowified


    def just_create_verts(self, positions):
        splatting_indices, splatting_weights=  self.lattice.just_create_verts(positions, False) #false is for the with_homogeneous coord
        return splatting_indices, splatting_weights

    def create_coarse_verts(self):
        coarsened_lattice=self.lattice.create_coarse_verts()
        coarsened_lattice_py=LatticePy()
        coarsened_lattice_py.lattice=coarsened_lattice
        # print("lattice_py. created a coarsened lattice with values of shape ", coarsened_lattice_py.values().shape)
        return coarsened_lattice_py

    def create_coarse_verts_naive(self, positions):
        coarsened_lattice=self.lattice.create_coarse_verts_naive(positions)
        coarsened_lattice_py=LatticePy()
        coarsened_lattice_py.lattice=coarsened_lattice
        # print("lattice_py. created a coarsened lattice with values of shape ", coarsened_lattice_py.values().shape)
        return coarsened_lattice_py
    
    def slice_standalone_no_precomputation(self, positions ):
        return self.lattice.slice_standalone_no_precomputation(positions )

    # def slice_elevated_verts(self, lattice_to_slice_from):
    #     sliced_lattice = self.lattice.slice_elevated_verts(lattice_to_slice_from.lattice) 
    #     sliced_lattice_py=LatticePy()
    #     sliced_lattice_py.lattice=sliced_lattice
    #     return sliced_lattice_py

    def gather_standalone_no_precomputation(self, positions):
        return self.lattice.gather_standalone_no_precomputation(positions)

    def gather_standalone_with_precomputation(self, positions, splatting_indices, splatting_weights):
        return self.lattice.gather_standalone_with_precomputation(positions, splatting_indices, splatting_weights)

    def gather_elevated_standalone_no_precomputation(self, lattice_to_gather_from):
        return self.lattice.gather_elevated_standalone_no_precomputation(lattice_to_gather_from.lattice)

    def keys_to_mesh(self):
        # if(self.lattice.m_pos_dim is not 2):
            # sys.exit("In order to show the keys as a mesh the pos_dim has to be 2 because only then the keys will be in 3D space and not in something bigger")
        # keys=self.lattice.m_hash_table.m_keys_tensor
        # keys=keys.unsqueeze(0)
        V=self.lattice.keys_to_verts()
        mesh=Mesh()
        mesh.V=V
        mesh.m_vis.m_show_points=True

        return mesh

    #elevate some positions into the hyperplane and returns the tensor of the m_pos_dim+1 positions
    def elevate_mesh(self, positions_raw):
        elevated_eigen=self.lattice.elevate(positions_raw)
        mesh=Mesh()
        mesh.V=elevated_eigen
        mesh.m_vis.m_show_points=True
        return mesh

    def show_lattice(self, name):
        points_deelevated=self.lattice.deelevate(self.lattice.hash_table().m_keys_tensor)

        #set in red the vertices that no neigbhours, the one for which lattice rowified has a row of zero
        color_no_neighbours=self.lattice.color_no_neighbours()

        mesh_deelevated=Mesh()
        mesh_deelevated.V=points_deelevated
        mesh_deelevated.C=color_no_neighbours
        mesh_deelevated.m_vis.m_show_points=True
        mesh_deelevated.m_vis.set_color_pervertcolor()
        Scene.show(mesh_deelevated,name)       

    #getters 
    def val_dim(self):
        return self.lattice.val_dim()
    # def val_full_dim(self):
        # return self.lattice.val_full_dim()
    def pos_dim(self):
        return self.lattice.pos_dim()
    def keys(self):
        return self.lattice.hash_table().m_keys_tensor
    def values(self):
        return self.lattice.hash_table().m_values_tensor
    def sliced_values_hom(self):
        return self.lattice.m_sliced_values_hom_tensor
    def lattice_rowified(self):
        return self.lattice.m_lattice_rowified
    def distributed(self):
        return self.lattice.m_distributed_tensor
    def splatting_indices(self):
        return self.lattice.m_splatting_indices_tensor
    def splatting_weights(self):
        return self.lattice.m_splatting_weights_tensor
    def nr_lattice_vertices(self):
        return self.lattice.nr_lattice_vertices()
    def capacity(self):
        return self.lattice.capacity()
    def positions(self):
        return self.lattice.positions()

    #setters
    # def set_val_dim(self, val_dim):
        # self.lattice.set_val_dim(val_dim)
    # def set_val_full_dim(self, val_dim):
        # self.lattice.set_val_full_dim(val_dim)
    def set_values(self, new_values):
        # self.lattice.m_hash_table.set_values(new_values)
        # if not new_values.is_contiguous():
            # new_values=new_values.contiguous()
        self.lattice.hash_table().set_values(new_values)
        # self.set_val_dim(new_values.shape[1])
        self.lattice.hash_table().update_impl()
        #this tensor needs to have a shape so that the sizes of the forwards and backard pass match
    # def set_positions(self, positions):
        # self.lattice.m_positions=positions
    def set_splatting_indices(self, indices):
        self.lattice.m_splatting_indices_tensor=indices
    def set_splatting_weights(self, weights):
        self.lattice.m_splatting_weights_tensor=weights
    # def clone_lattice(self):
    #     cloned_lattice = LatticePy()
    #     cloned_lattice.lattice=self.lattice.clone_lattice()

    #     self.data=self.values()
    #     cloned_lattice.data=cloned_lattice.values()

    #     return cloned_lattice

    def clone_lattice(self):
        convolved_lattice=self.lattice.clone_lattice()
        convolved_lattice_py=LatticePy()
        convolved_lattice_py.lattice=convolved_lattice

        #this tensor needs to have a shape so that the sizes of the forwards and backard pass match
        convolved_lattice_py.data=convolved_lattice_py.values()

        return convolved_lattice_py

    def compute_nr_points_per_lattice_vertex(self):
        indices=self.splatting_indices() #this is size nr_positions*(m_pos_dim+1) indicating for each position the m_pos_dim+1 simplexes onto which it splats
        indices_long=indices.long()
        #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long<0]=0

        ones=torch.ones(indices.shape[0]).to("cuda")

        # print("compute_nr_points: indices is ", indices_long)

        nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
        nr_points_per_simplex=nr_points_per_simplex[1:] #the invalid simplex is at zero, the one in which we accumulate or the splatting indices that are -1
        mean_points=nr_points_per_simplex.mean()
        max_points=nr_points_per_simplex.max()
        min_points=nr_points_per_simplex.min()
        median_points=nr_points_per_simplex.median()

        print("lattice has mean points per simplex: ", mean_points.item(), " min: ", min_points.item(), " max: ", max_points.item(), "median: ", median_points.item())

        return mean_points, max_points, min_points, median_points, nr_points_per_simplex

