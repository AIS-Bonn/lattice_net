all:
	echo "Building LatticeNet"
	python3 -m pip install -v --user --editable ./ 

clean:
	python3 -m pip uninstall latticenet
	rm -rf build *.egg-info build latticenet*.so liblatticenet_cpp.so

        


        
        