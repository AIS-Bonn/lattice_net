#include "lattice_net/PyBridge.h"

#include <torch/extension.h>
#include "torch/torch.h"
#include "torch/csrc/utils/pybind.h"

//my stuff 
// #include "data_loaders/DataLoaderShapeNetPartSeg.h"
// #include "easy_pbr/Mesh.h"
// #include "easy_pbr/LabelMngr.h"
#include "lattice_net/Lattice.cuh"
#include "lattice_net/HashTable.cuh"
#include "lattice_net/TrainParams.h"
#include "lattice_net/ModelParams.h"
#include "lattice_net/EvalParams.h"


// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work 
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;




PYBIND11_MODULE(latticenet, m) {


    //Lattice
    // py::module::import("torch");
    // py::object variable = (py::object) py::module::import("torch").attr("autograd").attr("Variable"); //from here but it segment faults https://pybind11.readthedocs.io/en/stable/advanced/misc.html
    py::class_<HashTable, std::shared_ptr<HashTable>   > (m, "HashTable")
    .def_readonly("m_values_tensor", &HashTable::m_values_tensor) //careful when using this because setting it and not using update_impl is a big bug
    .def_readonly("m_keys_tensor", &HashTable::m_keys_tensor) //careful when using this because setting it and not using update_impl is a big bug
    .def_readonly("m_nr_filled_tensor", &HashTable::m_nr_filled_tensor) ////careful when using this because setting it and not using update_impl is a big bug
    .def("update_impl", &HashTable::update_impl)
    .def("set_values", &HashTable::set_values)
    ;

    py::class_<Lattice, std::shared_ptr<Lattice>   > (m, "Lattice")
    // py::class_<Lattice, std::shared_ptr<Lattice>   > (m, "Lattice", variable)
    // py::class_<Lattice, at::Tensor, std::shared_ptr<Lattice>   > (m, "Lattice")
    // py::class_<Lattice, torch::autograd::Variable, std::shared_ptr<Lattice>   > (m, "Lattice")
    // py::class_<Lattice, torch::autograd::Variable > (m, "Lattice")
    .def_static("create", &Lattice::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_static("create", &Lattice::create<const std::string, const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("begin_splat",  &Lattice::begin_splat )
    // .def("begin_splat_modify_only_values",  &Lattice::begin_splat_modify_only_values )
    .def("splat_standalone",  &Lattice::splat_standalone )
    .def("distribute",  &Lattice::distribute )
    // .def("create_splatting_mask",  &Lattice::create_splatting_mask )
    // .def("blur_standalone",  &Lattice::blur_standalone )
    // .def("convolve_standalone",  &Lattice::convolve_standalone )
    // .def("depthwise_convolve",  &Lattice::depthwise_convolve )
    .def("convolve_im2row_standalone",  &Lattice::convolve_im2row_standalone )
    .def("im2row",  &Lattice::im2row )
    .def("row2im",  &Lattice::row2im )
    .def("just_create_verts",  &Lattice::just_create_verts )
    .def("create_coarse_verts",  &Lattice::create_coarse_verts )
    .def("create_coarse_verts_naive",  &Lattice::create_coarse_verts_naive )
    .def("slice_standalone_no_precomputation", &Lattice::slice_standalone_no_precomputation )
    // .def("slice_elevated_verts", &Lattice::slice_elevated_verts )
    .def("slice_classify_no_precomputation", &Lattice::slice_classify_no_precomputation )
    .def("slice_classify_with_precomputation", &Lattice::slice_classify_with_precomputation )
    .def("gather_standalone_no_precomputation", &Lattice::gather_standalone_no_precomputation )
    .def("gather_standalone_with_precomputation", &Lattice::gather_standalone_with_precomputation )
    // .def("gather_elevated_standalone_no_precomputation", &Lattice::gather_elevated_standalone_no_precomputation )
    .def("slice_backwards_standalone_with_precomputation", &Lattice::slice_backwards_standalone_with_precomputation )
    .def("slice_backwards_standalone_with_precomputation_no_homogeneous", &Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous )
    // .def("slice_backwards_elevated_verts_with_precomputation", &Lattice::slice_backwards_elevated_verts_with_precomputation )
    .def("slice_classify_backwards_with_precomputation", &Lattice::slice_classify_backwards_with_precomputation )
    .def("gather_backwards_standalone_with_precomputation", &Lattice::gather_backwards_standalone_with_precomputation )
    // .def("gather_backwards_elevated_standalone_with_precomputation", &Lattice::gather_backwards_elevated_standalone_with_precomputation )
    // .def("row2im", &Lattice::row2im )
    // .def("to_tensors", &Lattice::to_tensors )
    // .def("from_tensors", &Lattice::from_tensors )
    .def("get_filter_extent", &Lattice::get_filter_extent )
    .def("val_dim", &Lattice::val_dim )
    // .def("val_full_dim", &Lattice::val_full_dim )
    .def("pos_dim", &Lattice::pos_dim )
    .def("name", &Lattice::name )
    .def("nr_lattice_vertices", &Lattice::nr_lattice_vertices )
    // .def("set_nr_lattice_vertices", &Lattice::set_nr_lattice_vertices )
    .def("capacity", &Lattice::capacity )
    .def("positions", &Lattice::positions )
    .def("sigmas_tensor", &Lattice::sigmas_tensor)
    .def("hash_table", &Lattice::hash_table)
    // .def_readwrite("m_hash_table", &Lattice::m_hash_table )
    // .def_readwrite("m_sliced_values_hom_tensor", &Lattice::m_sliced_values_hom_tensor )
    // .def_readwrite("m_lattice_rowified", &Lattice::m_lattice_rowified )
    // .def_readwrite("m_distributed_tensor", &Lattice::m_distributed_tensor)
    // .def_readwrite("m_splatting_indices_tensor", &Lattice::m_splatting_indices_tensor)
    // .def_readwrite("m_splatting_weights_tensor", &Lattice::m_splatting_weights_tensor)
    // .def_readwrite("m_positions", &Lattice::m_positions)
    // .def_readwrite("m_name", &Lattice::m_name )
    // .def("set_val_dim", &Lattice::set_val_dim)
    // .def("set_val_full_dim", &Lattice::set_val_full_dim)
    .def("clone_lattice", &Lattice::clone_lattice)
    // .def("keys_to_verts", &Lattice::keys_to_verts)
    // .def("elevate", &Lattice::elevate)
    // .def("deelevate", &Lattice::deelevate)
    // .def("color_no_neighbours", &Lattice::color_no_neighbours)
    .def("increase_sigmas", &Lattice::increase_sigmas)
    .def("set_sigma", &Lattice::set_sigma)
    ;

      //TrainParams
    py::class_<TrainParams, std::shared_ptr<TrainParams>   > (m, "TrainParams", py::module_local())
    .def_static("create", &TrainParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("dataset_name",  &TrainParams::dataset_name )
    .def("with_viewer",  &TrainParams::with_viewer )
    .def("with_visdom",  &TrainParams::with_visdom )
    .def("lr",  &TrainParams::lr )
    .def("weight_decay",  &TrainParams::weight_decay )
    .def("save_checkpoint",  &TrainParams::save_checkpoint )
    .def("checkpoint_path",  &TrainParams::checkpoint_path )
    ;

    //EvalParams
    py::class_<EvalParams, std::shared_ptr<EvalParams>   > (m, "EvalParams", py::module_local())
    .def_static("create", &EvalParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("dataset_name",  &EvalParams::dataset_name )
    .def("with_viewer",  &EvalParams::with_viewer )
    .def("checkpoint_path",  &EvalParams::checkpoint_path )
    .def("do_write_predictions",  &EvalParams::do_write_predictions )
    .def("output_predictions_path",  &EvalParams::output_predictions_path )
    ;

    //ModelParams
    py::class_<ModelParams, std::shared_ptr<ModelParams>   > (m, "ModelParams", py::module_local())
    .def_static("create", &ModelParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("positions_mode",  &ModelParams::positions_mode )
    .def("values_mode",  &ModelParams::values_mode )
    .def("pointnet_layers",  &ModelParams::pointnet_layers )
    .def("pointnet_start_nr_channels",  &ModelParams::pointnet_start_nr_channels )
    .def("nr_downsamples",  &ModelParams::nr_downsamples )
    .def("nr_blocks_down_stage",  &ModelParams::nr_blocks_down_stage )
    .def("nr_blocks_bottleneck",  &ModelParams::nr_blocks_bottleneck )
    .def("nr_blocks_up_stage",  &ModelParams::nr_blocks_up_stage )
    .def("nr_levels_down_with_normal_resnet",  &ModelParams::nr_levels_down_with_normal_resnet )
    .def("nr_levels_up_with_normal_resnet",  &ModelParams::nr_levels_up_with_normal_resnet )
    .def("compression_factor",  &ModelParams::compression_factor )
    .def("dropout_last_layer",  &ModelParams::dropout_last_layer )
    .def("experiment",  &ModelParams::experiment )
    ;


}