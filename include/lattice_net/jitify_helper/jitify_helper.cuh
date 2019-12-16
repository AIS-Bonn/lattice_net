#pragma once

//from https://github.com/lattice/quda/blob/develop/include/jitify_helper.cuh
/**
   @file jitify_helper.cuh
   @brief Helper file when using jitify run-time compilation.  This
   file should be included in source code, and not jitify.hpp
   directly.
*/

#include "lattice_net/jitify_helper/jitify_options.hpp" //Needs to be added BEFORE jitify because this defined the include paths so that the kernels can find each other
#include <jitify/jitify.hpp>


static jitify::JitCache *kernel_cache = nullptr;
// static jitify::Program *program = nullptr;
static bool jitify_init = false;

static jitify::Program create_jitify_program(const std::string file, const std::vector<std::string> extra_options = {}) {

    if (!jitify_init) {
        jitify_init = true;
        kernel_cache = new jitify::JitCache;
    }

    // std::vector<std::string> options = {"-std=c++11", "--generate-line-info", "-ftz=true", "-prec-div=false", "-prec-sqrt=false"};
    // std::vector<std::string> options = {"-std=c++11", "--generate-line-info"}; //https://stackoverflow.com/questions/41672506/how-do-i-direct-all-accesses-to-global-memory-in-cuda
    std::vector<std::string> options = {"-std=c++11", "-ftz=true", "-prec-div=false", "-prec-sqrt=false"};

//   options.push_back(std::string("-G"));

    // add an extra compilation options specific to this instance
    for (auto option : extra_options) options.push_back(option);

    // program = new jitify::Program(kernel_cache->program(file, 0, options));
    return kernel_cache->program(file, 0, options);
}
