# LatticeNet 

<p align="middle">
  <img src="imgs/teaser.png" width="550" />
</p>

This is the official PyTorch implementation of [LatticeNet: Fast Point Cloud Segmentation Using Permutohedral Lattices](https://arxiv.org/abs/1912.05905)

### Dependencies 
```sh
$ sudo python3 -m pip install  --verbose --no-cache-dir  torch-scatter==1.4.0 
```
You will also need to install both [EasyPBR] and [DataLoaders] with the intructions on the respective pages.


### Build and install: 
```sh
$ git clone --recursive https://git.ais.uni-bonn.de/rosu/lattice_net.git
$ cd lattice_net
$ make
```

### Training 
For training start the script: 
```sh
$ python/lnn_train.py 
```






