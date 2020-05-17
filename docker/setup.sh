echo "Performing setup of the system by writing some config files"
#ros sourcing
/echo_to_file.sh /.bashrc "source /opt/ros/melodic/setup.bash"

#activate conda environment which is the same in which we install stuff in the Dockerfile
# ./echo_to_file.sh ~/.bashrc "source activate pt"
#make it so that conda doesnt modify the terminal line when we activate the conda environment
#touch ~/.condarc
#./echo_to_file.sh ~/.condarc "changeps1: false"
