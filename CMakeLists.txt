cmake_minimum_required(VERSION 2.8 FATAL_ERROR) # we need at least 3.11 because we use the FetchContent feature to manage dependencies and at least 3.14 because we use FetchContent_MakeAvailable

project(lattice_net)


###   VARIABLES   ##############################################################
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wall -O3") 
set(PYTORCH_PATH "/opt/pytorch")
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/ros/melodic/\;${PYTORCH_PATH}"
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${PYTORCH_PATH}")
set(CMAKE_CXX_STANDARD 11) 

set(CMAKE_VERBOSE_MAKEFILE ON)


add_subdirectory( ${CMAKE_SOURCE_DIR}/deps/pybind11)

#pybind attempt 3
# pybind11_add_module(DataLoaderTest ${CMAKE_SOURCE_DIR}/src/pycom/PyCom.cxx )
pybind11_add_module(latticenet ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )


####   GLOBAL OPTIONS   ###https://stackoverflow.com/questions/15201064/cmake-conditional-preprocessor-define-on-code


######   PACKAGES   ############################################################
# find_package(catkin REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(OpenCV REQUIRED)
find_package(EasyPBR REQUIRED)
find_package(DataLoaders REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
# add_subdirectory(${PROJECT_SOURCE_DIR}/deps/pybind11)
# find_package(catkin REQUIRED COMPONENTS roscpp std_msgs cv_bridge pcl_ros image_transport )


###   INCLUDES   #########################################################
# include_directories(${catkin_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/deps)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EASYPBR_INCLUDE_DIR})
include_directories(${DATALOADERS_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
# catkin_package( INCLUDE_DIRS include  CATKIN_DEPENDS roscpp std_msgs cv_bridge pcl_ros image_transport )


###   SOURCES   #################################################################
set(MY_SRC  
    # ${PROJECT_SOURCE_DIR}/src/Frame.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderVolRef.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderImgRos.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderScanNet.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderShapeNetPartSeg.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataTransformer.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderSemanticKitti.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderStanfordIndoor.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderToyExample.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderCloudRos.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderRueMonge.cxx
    # ${PROJECT_SOURCE_DIR}/src/RosBagPlayer.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderPNG.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderModelNet40.cxx
)
set(DEPS_SRC 
    # ${PROJECT_SOURCE_DIR}/deps/json11/json11.cpp
    )


###   SET ALL THE GLOBAL OPTIONS #########################################


#lib 
# add_library( latticenet_cpp SHARED ${MY_SRC}  ${DEPS_SRC}  )

##    CUDA library  ######################
#configure the include paths of jitify so that the kernels can properly include other ones
configure_file(${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp.in ${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp)
install(FILES "${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/jitify_options.hpp" DESTINATION ${PROJECT_SOURCE_DIR}/include/lattice_net/jitify_helper/)
# set(CUDA_NVCC_FLAGS "-gencode arch=compute_30 -code=sm_30" CACHE STRING "nvcc flags" FORCE)
set(CUDA_NVCC_FLAGS "-arch=sm_60" CACHE STRING "nvcc flags" FORCE)
message("NVCC flags are ", ${CUDA_NVCC_FLAGS})
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE) 
# set(LIB_TYPE SHARED)
cuda_add_library(latticenet_cu  SHARED  ${CMAKE_SOURCE_DIR}/src/Lattice.cu 
                                        ${CMAKE_SOURCE_DIR}/src/HashTable.cu  
                                        ${CMAKE_SOURCE_DIR}/src/ModelParams.cxx
                                        ${CMAKE_SOURCE_DIR}/src/TrainParams.cxx
                                        )
# set(LIBS ${LIBS} latticenet_cu)


##pybind
# pybind11_add_module(dataloaders ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )

###   EXECUTABLE   #######################################
# add_executable(run_data_loaders ${PROJECT_SOURCE_DIR}/src/main.cxx  )


###   LIBS   ###############################################
set(LIBS ${LIBS} ${catkin_LIBRARIES} ${EASYPBR_LIBRARY} ${DATALOADERS_LIBRARY} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS})






target_link_libraries(latticenet_cu ${LIBS} )
target_link_libraries(latticenet PRIVATE latticenet_cu )
# target_link_libraries(run_data_loaders PRIVATE data_loaders_cpp  )


#definitions for cmake variables that are necesarry during runtime
target_compile_definitions(latticenet_cu PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}") #point to the cmakelist folder of the easy_pbr
target_compile_definitions(latticenet_cu PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}") # points to the CMakeList folder of whichever project included easy_pbr


