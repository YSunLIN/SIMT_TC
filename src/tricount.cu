#include "tricount.h"
#include "cuda_runtime.h"
#include <iostream>

using namespace std;

const int WARPSIZE = 16;
const int BLOCKSIZE = 32;
uint64_t edgeBlockSize = 1024 * 1024 * 1024 / sizeof(uint64_t) * 8;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void initGPU(const uint64_t edge_num, const uint32_t N){
    cudaDeviceProp deviceProp;
    gpuErrchk( cudaGetDeviceProperties(&deviceProp, 0) );
    
    // 64MB for redundancy
    uint64_t usageMem = edge_num / 2 * 8 + (N + 1) * 4 * 3 + 1024 * 1024 * 64;
    if(usageMem > deviceProp.totalGlobalMem){
        cerr << "Global memory(" 
                << deviceProp.totalGlobalMem / 1024 / 1024 
                << "MB) is not enough. Require " 
                << usageMem / 1024 / 1024 << "MB" << endl;
        exit(2);
    }

    edgeBlockSize = (deviceProp.totalGlobalMem - usageMem) / sizeof(Edge_t);
    edgeBlockSize = min(edge_num, edgeBlockSize);

#ifdef DEBUG_DF
    cout << "Global memory: " 
                << deviceProp.totalGlobalMem / 1024 / 1024 
                << "MB; Require: " 
                << (usageMem + edgeBlockSize * sizeof(Edge_t)) / 1024 / 1024 << "MB." << endl;
#endif
}


__device__ void intersection16(const uint32_t* lbases, const uint32_t* rbases, uint32_t ln, uint32_t rn, 
                                unsigned long long* p_mysum){
    __shared__ uint32_t lblock[BLOCKSIZE];
    __shared__ uint32_t rblock[BLOCKSIZE];

    const int warpBegin = threadIdx.x & (~(WARPSIZE - 1));
    const int threadLane = threadIdx.x & (WARPSIZE - 1);

    uint32_t i = 0, j = 0, sum = 0;
    uint32_t lsize = WARPSIZE, rsize = WARPSIZE;

    while (i < ln && j < rn) {

        lsize = min(ln - i, WARPSIZE);
        rsize = min(rn - j, WARPSIZE);

        if(i + threadLane < ln) lblock[threadIdx.x] = lbases[i + threadLane];
        if(j + threadLane < rn) rblock[threadIdx.x] = rbases[j + threadLane];

        __threadfence_block();

        for(int k = 0; k < rsize; ++k)
            sum += (threadLane < lsize) & (lblock[threadIdx.x] == rblock[warpBegin + k]);

        uint32_t llast = lblock[warpBegin + lsize - 1];
        uint32_t rlast = rblock[warpBegin + rsize - 1];

        if(llast >= rlast) j += rsize;
        if(llast <= rlast) i += lsize;
    }

    (*p_mysum) += sum;
}


__device__ unsigned long long dev_sum;
__device__ unsigned int dev_nowNode;


__global__ void __tricount(uint32_t N, const uint32_t* __restrict__ nodeIndex, const uint32_t* __restrict__ adjList){
    __shared__ unsigned long long sdata[BLOCKSIZE];
    unsigned long long mysum = 0;

    const int warpLane = threadIdx.x / WARPSIZE;
    const int warpNum = blockDim.x / WARPSIZE;

    __shared__ unsigned int nodeI;
    __shared__ unsigned int nodeEnd;
    

    while(true){
        if(threadIdx.x == 0){
            if(++nodeI >= nodeEnd){
                nodeI = atomicAdd(&dev_nowNode, 256);
                nodeEnd = min(N, nodeI + 256);
            }
        }

        __syncthreads();

        unsigned int i = nodeI;
        if(i >= N) break;

        uint32_t lb = nodeIndex[i];
        uint32_t le = nodeIndex[i + 1];
        uint32_t ln = le - lb;

        for(uint32_t j = lb + warpLane; j < le; j += warpNum){
            uint32_t ri = adjList[j];
            uint32_t rn = nodeIndex[ri+1] - nodeIndex[ri];
            uint32_t rb = nodeIndex[ri];

            intersection16(adjList + lb, adjList + rb, ln, rn, &mysum);
        }
    }

    sdata[threadIdx.x] = mysum;
    __syncthreads();

    for (int s=1; s < blockDim.x; s *=2){
        int index = 2 * s * threadIdx.x;

        if (index < blockDim.x){
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&dev_sum, sdata[0]);
}


unsigned long long tricount(uint32_t N, const uint32_t* nodeIndex, const uint32_t* adjList){
    int numBlocks = 2048;

    uint32_t* dev_nodeIndex;
    uint32_t* dev_adjList;

    uint64_t size_nodeIndex = sizeof(uint32_t) * (N + 1);
    uint64_t size_adjList = sizeof(uint32_t) * nodeIndex[N];

    gpuErrchk( cudaMalloc((void**)&dev_nodeIndex, size_nodeIndex) );
    gpuErrchk( cudaMalloc((void**)&dev_adjList, size_adjList) );

    // copy inputs to device
    gpuErrchk( cudaMemcpy(dev_nodeIndex, nodeIndex, size_nodeIndex, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_adjList, adjList, size_adjList, cudaMemcpyHostToDevice) );


    unsigned long long sum = 0;
    gpuErrchk( cudaMemcpyToSymbol(dev_sum, &sum, sizeof(unsigned long long)) );
    gpuErrchk( cudaMemcpyToSymbol(dev_nowNode, &sum, sizeof(unsigned int)) );

    __tricount<<<numBlocks, BLOCKSIZE>>>(N, dev_nodeIndex, dev_adjList);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    return sum;
}


// degree collect on gpu
__global__ void __alldegreeCollect(const Edge_t* __restrict__ edgeList, const uint64_t __restrict__ edge_num, 
                                    uint32_t* deg){

    int blockSize = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i=tid; i<edge_num; i += blockSize){
        const Edge_t &e = edgeList[i];

        int res = (e.u != e.v);
        atomicAdd(deg + e.u, res);
        atomicAdd(deg + e.v, res);
    }
}

__global__ void __outdegreeCollect(const Edge_t* __restrict__ edgeList, const uint64_t __restrict__ edge_num, 
                                    uint32_t* deg){

    int blockSize = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i=tid; i<edge_num; i += blockSize){
        const Edge_t &e = edgeList[i];

        atomicAdd(deg + e.u, e.u != e.v);
    }
}

void degreeCollect(const int type, const Edge_t* edgeList, const uint64_t edge_num, 
                    uint32_t* deg, uint32_t N){
    int numBlocks = 2048;

    Edge_t* dev_edgeList;
    uint32_t* dev_deg;

    gpuErrchk( cudaMalloc((void**)&dev_edgeList, edgeBlockSize * sizeof(Edge_t)) );
    gpuErrchk( cudaMalloc((void**)&dev_deg, N * sizeof(uint32_t)) );

    auto func = __alldegreeCollect;
    if(type == 1){
        func = __outdegreeCollect;
    }

    for(uint64_t i = 0; i < edge_num; i += edgeBlockSize){
        uint64_t copySize = min(edge_num - i, edgeBlockSize);

        gpuErrchk( cudaMemcpy(dev_edgeList, edgeList + i, copySize * sizeof(Edge_t), cudaMemcpyHostToDevice) );
        func<<<numBlocks, BLOCKSIZE>>>(dev_edgeList, copySize, dev_deg);
        gpuErrchk( cudaDeviceSynchronize() );
    }

    gpuErrchk( cudaMemcpy(deg, dev_deg, N * sizeof(uint32_t), cudaMemcpyDeviceToHost) );

    cudaFree(dev_edgeList);
    cudaFree(dev_deg);
}


// adjList(CSR) construct on gpu
__global__ void __adjListConstruct(const Edge_t* edgeList, const uint64_t edge_num, 
                        const uint32_t* nbr_u, uint32_t* nbr_size, uint32_t* nbr_arr){

    int blockSize = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i=tid; i<edge_num; i += blockSize){
        const uint32_t &u = edgeList[i].u;
        const uint32_t &v = edgeList[i].v;
        
        if(u == v) continue;

        uint32_t j = atomicAdd(nbr_size + u, 1);
        nbr_arr[nbr_u[u] + j] = v;
    }
}

void adjListConstruct(const Edge_t* edgeList, const uint64_t edge_num, 
                        const uint32_t* nbr_u, uint32_t* nbr_arr, const uint32_t N){
    int numBlocks = 2048;

    Edge_t* dev_edgeList;
    uint32_t* dev_nbr_arr;
    uint32_t* dev_nbr_u;
    uint32_t* dev_nbr_size;

    gpuErrchk( cudaMalloc((void**)&dev_edgeList, edgeBlockSize * sizeof(Edge_t)) );
    gpuErrchk( cudaMalloc((void**)&dev_nbr_arr, edge_num * sizeof(uint32_t)) );
    gpuErrchk( cudaMalloc((void**)&dev_nbr_u, N * sizeof(uint32_t)) );
    gpuErrchk( cudaMalloc((void**)&dev_nbr_size, N * sizeof(uint32_t)) );

    gpuErrchk( cudaMemcpy(dev_nbr_u, nbr_u, N * sizeof(uint32_t), cudaMemcpyHostToDevice) );

    for(uint64_t i = 0; i < edge_num; i += edgeBlockSize){
        uint64_t copySize = min(edge_num - i, edgeBlockSize);

        gpuErrchk( cudaMemcpy(dev_edgeList, edgeList + i, copySize * sizeof(Edge_t), cudaMemcpyHostToDevice) );
        __adjListConstruct<<<numBlocks, BLOCKSIZE>>>(dev_edgeList, copySize, 
                                                        dev_nbr_u, dev_nbr_size, dev_nbr_arr);
        gpuErrchk( cudaDeviceSynchronize() );
    }

    gpuErrchk( cudaMemcpy(nbr_arr, dev_nbr_arr, edge_num * sizeof(uint32_t), cudaMemcpyDeviceToHost) );

    cudaFree(dev_edgeList);
    cudaFree(dev_nbr_arr);
    cudaFree(dev_nbr_u);
    cudaFree(dev_nbr_size);
}

