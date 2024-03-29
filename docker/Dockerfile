# FROM ubuntu:bionic

# FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
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
RUN apt-get -y update && apt-get install -y --no-install-recommends sudo ssh tmux git meld cmake cmake-curses-gui htop vim iputils-ping apt-utils apt-transport-https software-properties-common wget build-essential gdb valgrind silversearcher-ag curl dirmngr gpg-agent checkinstall locate baobab ninja-build libncurses5-dev pkg-config && sh "./cleanup.sh"

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


#also needed cmake 3.13 and above for pytorch
#CMAKE above 3.19  https://askubuntu.com/a/8652941
# https://gitlab.kitware.com/cmake/cmake/-/issues/22245
RUN sudo apt-get update
RUN sudo apt-get install software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN sudo apt update
RUN sudo apt install -y cmake


#python3
COPY cleanup.sh ./
RUN chmod +x ./cleanup.sh
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends python3-dev python3-pip python3-setuptools && sh "./cleanup.sh"

#switch to python3  https://linuxconfig.org/how-to-change-from-default-to-alternative-python-version-on-debian-linux
#this is needed so that catkin uses python3 and therefore pybind uses python3 which is the only one that can use spconv..
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

#packages that were previously installed by ros but now we need the expliticly
RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get -y install libopencv-dev libboost-all-dev libpcl-dev


# RUN wget http://mirrors.edge.kernel.org/ubuntu/pool/universe/e/eigen3/libeigen3-dev_3.3.7-3_all.deb
# RUN sudo dpkg -i libeigen3-dev_3.3.7-3_all.deb
# RUN rm libeigen3-dev_3.3.7-3_all.deb
#update to eigen 3.4 https://ubuntu.pkgs.org/22.04/ubuntu-universe-amd64/libeigen3-dev_3.4.0-2ubuntu2_all.deb.html
COPY data/libeigen3-dev_3.4.0-2ubuntu2_all.deb ./
RUN sudo dpkg -i libeigen3-dev_3.4.0-2ubuntu2_all.deb



# #ROS and install also catkin for python3
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



#install conda 
# RUN apt-get -y update && apt-get install -y --no-install-recommends curl && sh "./cleanup.sh"
# RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh 
# ENV PATH /opt/conda/bin:$PATH
# WORKDIR /

# RUN conda install python=3.6 pip=20.0.2
# # RUN conda install -y pandas scikit-image cython -c pytorch
# RUN conda install pytorch=1.5.0 torchvision=0.6.0 -c pytorch

# #libtorch 
# WORKDIR /opt/libtorch
# RUN wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.0.zip
# RUN sudo apt-get update && sudo apt-get install -y unzip
# RUN unzip libtorch-cxx11-abi-shared-with-deps-1.5.0.zip
# WORKDIR ${workspace}


# source the basrc we provide for the docker
# WORKDIR /
# COPY .bashrc ./

# COPY lattice_env.yaml ./

# Create conda environment
# RUN conda env create -f lattice_env.yaml
# ENV PATH /opt/conda/envs/lattice/bin:$PATH
# #RUN echo "/opt/conda/envs/lattice/bin:$PATH" >> /.bashrc
# RUN echo "source activate lattice" >> /.bashrc
# RUN conda config --add channels conda-forge
# ENV PATH /opt/conda/envs/lattice/bin:$PATH


#pytorch
# RUN conda config --env --add channels pytorch
# RUN conda install -n pt numpy pyyaml mkl mkl-include setuptools cmake cffi typing magma-cuda100 && conda clean -ya 
# ENV PATH /opt/conda/envs/pt/bin:$PATH
WORKDIR /
# RUN GIT_TRACE=1 GIT_CURL_VERBOSE=1 git clone --verbose --recursive https://github.com/pytorch/pytorch /opt/pytorch
# RUN git clone --verbose --recursive git@github.com:pytorch/pytorch.git /opt/pytorch
#RUN git clone --verbose --recursive https://github.com/pytorch/pytorch  /opt/pytorch
RUN git clone  https://github.com/pytorch/pytorch  /opt/pytorch
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends python3-pip python3-setuptools && sh "./cleanup.sh"
# RUN pip3 install numpy pyyaml mkl mkl-include typing dataclasses
RUN sudo python3 -m pip install wheel
RUN sudo python3 -m pip install numpy pyyaml mkl mkl-include typing dataclasses typing-extensions 
WORKDIR /opt/pytorch
RUN git checkout tags/v1.7.1
RUN git submodule sync
RUN git submodule update --init --recursive
# RUN mkdir /opt/pytorch_installed
# ENV CMAKE_PREFIX_PATH /opt/pytorch_installed
ENV BUILD_TORCH ON 
# ENV FULL_CAFFE2 1
ENV BUILD_CAFFE2_OPS OFF
ENV BUILD_CAFFE2_MOBILE OFF
ENV MAX_JOBS 4
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
RUN MAX_JOBS=8 USE_NINJA=1 BUILD_TEST=0 BUILD_JNI=0 TORCH_CUDA_ARCH_LIST="5.2;5.3;6.0;6.1;6.2;7.0;7.5;8.0+PTX" python3 setup.py install 
WORKDIR /
# #need to make it own by myself and not by root because otherwise when we run python it will not access it
WORKDIR /
RUN git clone --recursive https://github.com/pytorch/vision /opt/vision
WORKDIR /opt/vision
RUN git checkout tags/v0.8.2
RUN git submodule sync
RUN git submodule update --init --recursive 
# ENV CMAKE_PREFIX_PATH /opt/conda/envs/pt
RUN MAX_JOBS=8 USE_NINJA=1 BUILD_TEST=0 BUILD_JNI=0 TORCH_CUDA_ARCH_LIST="5.2;5.3;6.0;6.1;6.2;7.0;7.5;8.0+PTX" python3 setup.py install
WORKDIR /

#torch scatter in order to have scatter_max which is useful for a pointnet architecture. It needs the cuda_arch list in order to find nvidia drivers and build the gpu part of the torch-scatter https://github.com/rusty1s/pytorch_scatter/pull/79
#FORCE_CUDA may be needed for cases like this when the cuda runtime is not exposed during docker build but is actually needed https://github.com/facebookresearch/maskrcnn-benchmark/issues/167
ARG CUDA_HOME="/usr/local/cuda"
ARG TORCH_CUDA_ARCH_LIST="5.2 5.3 6.0 6.1 6.2 7.0 8.0+PTX"
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ARG CPATH="/usr/local/cuda/include" 
RUN git clone --recursive https://github.com/rusty1s/pytorch_scatter /opt/torch_scatter
WORKDIR /opt/torch_scatter
RUN git checkout tags/2.0.4
RUN git submodule sync
RUN git submodule update --init --recursive
RUN FORCE_CUDA=1 python3 setup.py install 
WORKDIR /

#python3 libs 
RUN sudo apt-get update && sudo apt-get -y install python3-pip python3-tk python3-pyqt5 python3-crypto python3-gnupg python3-dbg && sh "./cleanup.sh"
RUN sudo python3 -m pip install --upgrade pip
RUN sudo apt-get update && sudo python3 -m pip install ipdb numpy jupyter h5py scipy ipython pillow matplotlib opencv-python wheel trollius rospkg asyncio netifaces visdom torchnet sympy tqdm dlutils sklearn scikit-image twine natsort termcolor && sh "./cleanup.sh"
RUN sudo python3 -m pip install --upgrade setuptools setuptools-git  && sh "./cleanup.sh"
#downgrade PIL to solve this issue https://github.com/RaduAlexandru/lattice_net/issues/1 
RUN sudo python3 -m pip install 'pillow<9' 



RUN apt-get -y update && apt-get install -y --no-install-recommends unzip && sh "./cleanup.sh"


#nvvp crashes under the default java11 so we need java8 as explained here https://bugs.launchpad.net/ubuntu/+source/nvidia-cuda-toolkit/+bug/1766948
# RUN sudo apt-get update && sudo apt-get install  -y --no-install-recommends  openjdk-8-jre && sh "./cleanup.sh"
# RUN sudo update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
# RUN sudo apt-get update && apt-get install -y --no-install-recommends cuda-command-line-tools-$CUDA_PKG_VERSION cuda-visual-tools-$CUDA_PKG_VERSION && sh "./cleanup.sh"

#python3 libs 
# RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip python3-tk python3-pyqt5 python3-crypto python3-gnupg python3-dbg && sh "./cleanup.sh"
# RUN sudo python3 -m pip install --upgrade pip
# RUN sudo apt-get update

# # create conda env and get all needed libraries
# RUN pip install ipdb numpy jupyter h5py scipy ipython pillow matplotlib opencv-python wheel trollius rospkg asyncio netifaces visdom torchnet sympy tqdm dlutils  twine natsort termcolor && sh "./cleanup.sh"
# RUN sudo python3 -m pip install --upgrade setuptools setuptools-git  && sh "./cleanup.sh" 

#torch scatter in order to have scatter_max which is useful for a pointnet architecture
#RUN pip install  --verbose --no-cache-dir  torch-scatter==1.4.0 

WORKDIR /

# RUN /bin/bash -c "source ./.bashrc"

# default command to run when running the container 
#COPY echo_to_file.sh /
#RUN chmod +x /echo_to_file.sh

#COPY setup.sh /
#RUN chmod +x /setup.sh
#ENTRYPOINT /setup.sh &&  /bin/sh

#catkin build for the workspace -> it is needed for the data_loaders package
#WORKDIR ${workspace}/src
#WORKDIR ${workspace}
#RUN /bin/sh -c "catkin build -c"
#RUN /bin/sh -c "source devel/setup.bash"

# clone the needed repositories into the docker
# WORKDIR ${workspace}
# RUN git clone --recursive https://github.com/RaduAlexandru/easy_pbr.git
# WORKDIR easy_pbr
# RUN make

# WORKDIR ${workspace}
# RUN git clone --recursive https://github.com/RaduAlexandru/data_loaders
# WORKDIR data_loaders
# RUN make
# after every data_loaders build the devel/setup.bash of the catkin workspace has to be sourced again


#·-------------------------------------------------------------------------------

#Cleanup
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update
WORKDIR / 
RUN rm cleanup.sh

# Make SSH available
EXPOSE 22
EXPOSE 42421
# TensorBoard https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 6006
# IPython https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 8888

# ENV CATKIN_TOPLEVEL_WS="${workspace}/devel"
# Switch to the workspace
WORKDIR ${workspace}

# Clone user into docker image and set up X11 sharing
RUN \
  echo "${user}:x:${uid}:${uid}:${user},,,:${home}:${shell}" >> /etc/passwd && \
  echo "${user}:x:${uid}:" >> /etc/group && \
  echo "${user} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${user}" && \
  chmod 0440 "/etc/sudoers.d/${user}"


#Set the user in the approapriate groups
RUN usermod -a -G dialout ${user}
RUN usermod -a -G video ${user}
RUN usermod -a -G audio ${user}
RUN usermod -a -G plugdev ${user}

#own some stuff because they were created by the root user in the docker container
# RUN chown -R ${user} /opt
# USER ${user}
# USER root

# Switch to user
RUN chown -R ${user} ${workspace}
# RUN echo ${workspace}
USER "${user}"

# switch to interactive
ENV DEBIAN_FRONTEND teletype
