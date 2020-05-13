# Dockerfiles

Repository by Peer Schütt, peerschuett1996@gmail.com .

## Workflow is: 

```./build.sh DOCKER_NAME```

```./run.sh DOCKER_NAME ```

The other .sh-files are copied into the docker during the build and executed.

## In the docker: 
After starting the docker, you have to ```source /.bashrc``` .

## Building data_loaders: 

The catkin workspace "/workspace" has to be initialized and ```source devel/setup.bash``` must have been called. Afterwards data_loaders can be successfully built. ```source devel/setup.bash``` should again be called after the installation.

## Building lattice_net

The pytorch path in CMakeslist has to be changed from ```/opt/pytorch``` to the actual path. It can be found by opening a python shell, importing torch and calling ```print(torch.__path__)```.
Additionally all steps in the README of lattice_net have to be performed (installing torchscatter, too).



