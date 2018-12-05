#ifndef TRICOUNT_H
#define TRICOUNT_H

#include <cstdint>
#include <unistd.h>

// size must be 8
struct Edge_t {
    uint32_t u;
    uint32_t v;
} __attribute__ ((aligned (4)));


void initGPU(const uint64_t edge_num, const uint32_t N);

unsigned long long tricount(uint32_t N, const uint32_t* nodeIndex, const uint32_t* adjList);

// type(0: all; 1: out degree)
void degreeCollect(const int type, const Edge_t* edgeList, const uint64_t edge_num, 
                        uint32_t* deg, const uint32_t N);
void adjListConstruct(const Edge_t* edgeList, const uint64_t edge_num, 
                        const uint32_t* nbr_u, uint32_t* nbr_arr, const uint32_t N);

#endif
