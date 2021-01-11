if [ ! -z "$1" ]; then

    if [ -d "$1" ]; then

        cwd=$(pwd) &&
        cd $1 &&
        git clone \
		--single-branch \
		--branch develop \
		--recursive \
		--recurse-submodules \
		https://github.com/PetaVision/OpenPV.git &&
	cd OpenPV && 
	mkdir build && 
	cd build && 
	cmake .. \
		-DPV_BUILD_SHARED=ON \
		-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
		-DCUDA_NVCC_FLAGS="-Xcompiler -fPIC" && 
	make -j $(nproc) && 
	ctest &&
	cd $cwd

    else
    
        echo "OpenPV install directory does not exist."

    fi

else

    echo "OpenPV install directory not given."

fi
