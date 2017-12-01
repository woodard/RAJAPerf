CUDA_HOME = /usr/tcetmp/packages/cuda-8.0/
CUDA_INCLUDES = -I${CUDA_HOME}/include
PERF_INCLUDES = -I/g/g0/holger/workspace/RAJAPerf_github/src  
RAJA_INCLUDES = -I/g/g0/holger/workspace/RAJAPerf_github/tpl/RAJA/include 
RAJA_INCLUDES_SPECIFIC = -I/g/g0/holger/workspace/RAJAPerf_github/build_blueos_nvcc8.0_clang-coral-2017.10.13/tpl/RAJA/include
RAJA_INCLUDES_SPECIFIC_CLANG = -I/g/g0/holger/workspace/RAJAPerf_github/build_blueos_clang-coral-2017.10.13_omptarget-nvcc8.0/tpl/RAJA/include
INCLUDES = ${CUDA_INCLUDES} ${PERF_INCLUDES} ${RAJA_INCLUDES} ${RAJA_INCLUDES_SPECIFIC} 
CUDA_COMPILE = ${CUDA_HOME}/bin/nvcc
CUDA_COMPILE += -x=cu -dc
CUDA_COMPILE += -m64 -DUSE_OPENMP -DUSE_CUDA
CUDA_COMPILE += -O3 -restrict
CUDA_COMPILE += -arch sm_60
CUDA_COMPILE += --std c++11
CUDA_COMPILE += --expt-extended-lambda
CUDA_COMPILE += -ccbin /usr/tce/packages/clang/clang-coral-2017.10.13/bin/clang++ -Xcompiler -O3 -Xcompiler -fopenmp 
CUDA_COMPILE += -DNVCC
CUDA_COMPILE += ${INCLUDES}


CLANG_COMPILE =  /usr/tce/packages/clang/clang-coral-2017.10.13/bin/clang++  
CLANG_COMPILE += -O3 -fPIC -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-implicit-declare-target -std=c++11 
CLANG_COMPILE += -DUSE_OPENMP 
CLANG_COMPILE += ${PERF_INCLUDES} ${RAJA_INCLUDES} ${RAJA_INCLUDES_SPECIFIC_CLANG}


LINK = /usr/tce/packages/clang/clang-coral-2017.10.13/bin/clang++  
LINK += -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-implicit-declare-target 
LINK_SUFFIX =  -rdynamic /usr/tcetmp/packages/cuda-8.0/lib64/libcudart_static.a -lpthread -ldl /usr/lib64/librt.so /g/g0/holger/workspace/RAJAPerf_github/build_blueos_nvcc8.0_clang-coral-2017.10.13/lib/libRAJA.a 

CUDA_LINK = ${CUDA_HOME}/bin/nvcc -arch=sm_60 -dlink 


all: 
  $(info $$CUDA_COMPILE = ${CUDA_COMPILE})
  $(info $$LINK = ${LINK})

build: raja-perf.exe



DataUtils.o:../src/common/DataUtils.cpp
	${CLANG_COMPILE} -o $@ -c $<
Executor.o:../src/common/Executor.cpp
	${CLANG_COMPILE} -o $@ -c $<
KernelBase.o:../src/common/KernelBase.cpp
	${CLANG_COMPILE} -o $@ -c $<
OutputUtils.o:../src/common/OutputUtils.cpp
	${CLANG_COMPILE} -o $@ -c $<
RAJAPerfSuite.o:../src/common/RAJAPerfSuite.cpp
	${CLANG_COMPILE} -o $@ -c $<
RunParams.o:../src/common/RunParams.cpp
	${CLANG_COMPILE} -o $@ -c $<
INIT3_separate.o:../src/basic/INIT3_separate.cpp
	${CLANG_COMPILE}  -o $@ -c $<
RAJAPerfSuiteDriver.o:../src/RAJAPerfSuiteDriver.cpp
	${CLANG_COMPILE} -o $@ -c $<
INIT3_cuda.o:../src/basic/INIT3_cuda.cpp
	${CUDA_COMPILE} -o $@ -c $<
INIT3_cuda_dlink.o:INIT3_cuda.o
	${CUDA_LINK} $< -o $@	

raja-perf.exe: DataUtils.o Executor.o KernelBase.o OutputUtils.o RAJAPerfSuite.o RunParams.o INIT3_separate.o INIT3_cuda.o INIT3_cuda_dlink.o RAJAPerfSuiteDriver.o
	${LINK} -o $@ $+ ${LINK_SUFFIX}


run: build
	./raja-perf.exe

clean:
	rm -f raja-perf.exe DataUtils.o Executor.o KernelBase.o OutputUtils.o RAJAPerfSuite.o RunParams.o INIT3_separate.o INIT3_cuda.o RAJAPerfSuiteDriver.o

clobber: clean
