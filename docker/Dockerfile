# FROM ubuntu:bionic

# FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display,video

# Arguments
ARG user
ARG uid
ARG home
ARG workspace
ARG shell

# switch to non interactive https://github.com/phusion/baseimage-docker/issues/319
ENV DEBIAN_FRONTEND noninteractive

# copy the cleanup script so that its accesible in docker https://stackoverflow.com/a/53167819
COPY cleanup.sh ./
RUN chmod +x ./cleanup.sh

# Basic Utilities (dirmngr and gpg-agent is needed so that apt-get adv works)
RUN apt-get -y update && apt-get install -y --no-install-recommends sudo ssh tmux git meld cmake cmake-curses-gui htop vim iputils-ping apt-utils apt-transport-https software-properties-common wget build-essential gdb valgrind silversearcher-ag curl dirmngr gpg-agent checkinstall locate ninja-build libncurses5-dev pkg-config && sh "./cleanup.sh"

#Make SSH keys work inside the container
RUN  echo "    IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config

#fixing locales in docker https://daten-und-bass.io/blog/fixing-missing-locale-setting-in-ubuntu-docker-image/
RUN sudo apt-get update \
    && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y locales locales-all \
    && sudo sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && sudo dpkg-reconfigure --frontend=noninteractive locales \
    && sudo update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8 
ENV LC_ALL en_US.UTF-8
RUN sudo locale-gen

#-------------------------------------------------------------------------------


#python3
COPY cleanup.sh ./
RUN chmod +x ./cleanup.sh
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends python3-dev python3-pip python3-setuptools && sh "./cleanup.sh"

#switch to python3  https://linuxconfig.org/how-to-change-from-default-to-alternative-python-version-on-debian-linux
#this is needed so that catkin uses python3 and therefore pybind uses python3 which is the only one that can use spconv..
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

#packages that were previously installed by ros but now we need the expliticly
RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get -y install libopencv-dev libboost-all-dev libpcl-dev

#ROS and install also catkin for python3
# COPY cleanup.sh ./
# RUN chmod +x ./cleanup.sh
# RUN sudo add-apt-repository main
# RUN sudo add-apt-repository universe
# RUN sudo add-apt-repository multiverse
# RUN sudo add-apt-repository restricted
# RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ros-melodic-desktop-full && sh "./cleanup.sh"
# RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ros-melodic-nmea-msgs && sh "./cleanup.sh"
# RUN sudo python3 -m pip install catkin-tools rosdep
# RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends python3-wstool && sh "./cleanup.sh"
# RUN sudo rosdep init
# RUN rosdep update
# RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends python-catkin-tools && sh "./cleanup.sh"

#GRAPHICS STUFF TODO
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends libglfw3-dev && sh "./cleanup.sh"

RUN wget http://mirrors.edge.kernel.org/ubuntu/pool/universe/e/eigen3/libeigen3-dev_3.3.7-1_all.deb
RUN sudo dpkg -i libeigen3-dev_3.3.7-1_all.deb
RUN rm libeigen3-dev_3.3.7-1_all.deb

# source the basrc we provide for the docker
WORKDIR /
# COPY .bashrc ./



#install conda 
RUN apt-get -y update && apt-get install -y --no-install-recommends curl && sh "./cleanup.sh"
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh 
ENV PATH /opt/conda/bin:$PATH
WORKDIR /


COPY lattice_env.yaml ./


# Create conda environment
RUN conda env create -f lattice_env.yaml
ENV PATH /opt/conda/envs/lattice/bin:$PATH
#RUN echo "/opt/conda/envs/lattice/bin:$PATH" >> /.bashrc
RUN echo "source activate lattice" >> /.bashrc
RUN conda config --add channels conda-forge
ENV PATH /opt/conda/envs/lattice/bin:$PATH

COPY cleanup.sh ./
RUN chmod +x ./cleanup.sh

#nvvp crashes under the default java11 so we need java8 as explained here https://bugs.launchpad.net/ubuntu/+source/nvidia-cuda-toolkit/+bug/1766948
# RUN sudo apt-get update && sudo apt-get install  -y --no-install-recommends  openjdk-8-jre && sh "./cleanup.sh"
# RUN sudo update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
# RUN sudo apt-get update && apt-get install -y --no-install-recommends cuda-command-line-tools-$CUDA_PKG_VERSION cuda-visual-tools-$CUDA_PKG_VERSION && sh "./cleanup.sh"

#python3 libs 
RUN sudo apt-get update
# RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip python3-tk python3-pyqt5 python3-crypto python3-gnupg python3-dbg && sh "./cleanup.sh"
RUN sudo python3 -m pip install --upgrade pip
# RUN sudo apt-get update

# create conda env and get all needed libraries
RUN pip install ipdb numpy jupyter h5py scipy ipython pillow matplotlib opencv-python wheel trollius rospkg asyncio netifaces visdom torchnet sympy tqdm dlutils sklearn scikit-image twine natsort termcolor && sh "./cleanup.sh"
RUN sudo python3 -m pip install --upgrade setuptools setuptools-git  && sh "./cleanup.sh" 

#install libtorch 
# WORKDIR / 
# RUN wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcu101.zip


#torch scatter in order to have scatter_max which is useful for a pointnet architecture
#RUN pip install  --verbose --no-cache-dir  torch-scatter==1.4.0 

WORKDIR /

# RUN /bin/bash -c "source ./.bashrc"

# default command to run when running the container 
#COPY echo_to_file.sh /
#RUN chmod +x /echo_to_file.sh

#COPY setup.sh /
#RUN chmod +x /setup.sh

#catkin build for the workspace -> it is needed for the data_loaders package
# WORKDIR ${workspace}/src
#WORKDIR ${workspace}
#RUN /bin/sh -c "catkin build -c"
#RUN /bin/sh -c "source devel/setup.bash"

# clone the needed repositories into the docker
# WORKDIR ${workspace}/src
WORKDIR ${workspace}
# RUN git clone --recursive https://github.com/RaduAlexandru/easy_pbr.git
# RUN git clone --recursive https://github.com/RaduAlexandru/data_loaders
# after every data_loaders build the devel/setup.bash of the catkin workspace has to be sourced again

# #default command to run when running the container 
# ENTRYPOINT ./setup.sh &&  /bin/bash

#·-------------------------------------------------------------------------------

#Cleanup
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update
# RUN rm cleanup.sh

# Make SSH available
EXPOSE 22
EXPOSE 42421
# TensorBoard https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 6006
# IPython https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 8888

ENV CATKIN_TOPLEVEL_WS="${workspace}/devel"
# Switch to the workspace
WORKDIR ${workspace}

# switch to interactive
ENV DEBIAN_FRONTEND teletype
