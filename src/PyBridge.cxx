#include "lattice_net/PyBridge.h"

// #include <torch/extension.h>
// #include "torch/torch.h"
// #include "torch/csrc/utils/pybind.h"

//my stuff 
// #include "data_loaders/DataLoaderShapeNetPartSeg.h"
// #include "easy_pbr/Mesh.h"
// #include "easy_pbr/LabelMngr.h"


// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work 
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;




PYBIND11_MODULE(latticenet, m) {

 
    // //DataLoader ShapeNetPartSeg
    // py::class_<DataLoaderShapeNetPartSeg> (m, "DataLoaderShapeNetPartSeg")
    // .def(py::init<const std::string>())
    // .def("start", &DataLoaderShapeNetPartSeg::start )
    // .def("get_cloud", &DataLoaderShapeNetPartSeg::get_cloud )
    // .def("has_data", &DataLoaderShapeNetPartSeg::has_data ) 
    // .def("is_finished", &DataLoaderShapeNetPartSeg::is_finished ) 
    // .def("is_finished_reading", &DataLoaderShapeNetPartSeg::is_finished_reading ) 
    // .def("reset", &DataLoaderShapeNetPartSeg::reset ) 
    // .def("nr_samples", &DataLoaderShapeNetPartSeg::nr_samples ) 
    // .def("set_mode_train", &DataLoaderShapeNetPartSeg::set_mode_train ) 
    // .def("set_mode_test", &DataLoaderShapeNetPartSeg::set_mode_test ) 
    // .def("set_mode_validation", &DataLoaderShapeNetPartSeg::set_mode_validation ) 
    // .def("get_object_name", &DataLoaderShapeNetPartSeg::get_object_name ) 
    // .def("set_object_name", &DataLoaderShapeNetPartSeg::set_object_name ) 
    // ;


}