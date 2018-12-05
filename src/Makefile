MACRO=-DDEBUG_DF
CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp $(MACRO)
NVCC=nvcc
NVCCFLAGS=--gpu-architecture=compute_60 --gpu-code=sm_60 -Xcompiler -O3 -std=c++11 $(MACRO)

all : tricount tricount_cpu

tricount : main_gpu.o tricount_cu.o gpu.o
	$(CXX) $(CXXFLAGS) -o tricount main_gpu.o tricount_cu.o gpu.o -L /usr/local/cuda/lib64 -lcudart

tricount_cpu : main_cpu.o bsr.o
	$(CXX) $(CXXFLAGS) -o tricount_cpu main_cpu.o bsr.o

tricount_cu.o : tricount.cu
	$(NVCC) $(NVCCFLAGS) -dc tricount.cu -o tricount_cu.o

gpu.o : tricount_cu.o
	$(NVCC) $(NVCCFLAGS) -dlink tricount_cu.o -o gpu.o

main_gpu.o : main_gpu.cpp mapfile.hpp
	$(CXX) $(CXXFLAGS) -c main_gpu.cpp

main_cpu.o : main_cpu.cpp mapfile.hpp bsr.cpp bsr.h
	$(CXX) $(CXXFLAGS) -c main_cpu.cpp bsr.cpp

.PHONY : clean
clean :
	rm tricount tricount_cpu *.o

