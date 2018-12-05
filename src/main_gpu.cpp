#include <iostream>
#include <fstream>
#include <cstring>
#include <omp.h>

#include <algorithm>
#include <random>
#include <chrono>

#include "mapfile.hpp"
#include "tricount.h"

using namespace std;
// #define DEBUG_DF


#ifdef DEBUG_DF

# include <sys/time.h>
class TimeInterval{
public:
    TimeInterval(){
        check();
    }

    void check(){
        gettimeofday(&tp, NULL);
    }

    void print(const char* title){
        struct timeval tp_end, tp_res;
        gettimeofday(&tp_end, NULL);
        timersub(&tp_end, &tp, &tp_res);
        cout << title << ": " << tp_res.tv_sec << " s " << tp_res.tv_usec << " us.\n";
    }
private:
    struct timeval tp;
};

TimeInterval allTime;
TimeInterval preProcessTime;
TimeInterval tmpTime;

#endif


uint32_t N = 0;

int main(int argc, char *argv[]) {
    if (argc != 3 || strcmp(argv[1], "-f") != 0) {
        cerr << "Usage: -f [data_file_path]" << endl;
        exit(1);
    }
    const char *filepath = argv[2];

#ifdef DEBUG_DF
    cout << "Start reading file" << endl;
    tmpTime.check();
#endif

    // load data file via mmap
    MapFile mapfile(filepath);

    if (mapfile.getLen() % sizeof(Edge_t)) {
        cerr << "file size error: cannot be divided by 8" << endl;
        exit(-1);
    }

    Edge_t *edgeFile = (Edge_t *) mapfile.getAddr();

    Edge_t *edgeList = (Edge_t *) (new char[mapfile.getLen()]);
    memcpy(edgeList, edgeFile, mapfile.getLen());
    mapfile.release();

    // get edge number
    uint64_t edge_num = mapfile.getLen() / sizeof(Edge_t);

#ifdef DEBUG_DF
    tmpTime.print("Mapfile Time Cost");
    
    preProcessTime.check();
    tmpTime.check();
#endif

    // find number of vertex
#pragma omp parallel for reduction(max : N)
    for (off_t i = 0; i < edge_num; ++i) {
        Edge_t &e = edgeList[i];

        N = max(N, e.u);
        N = max(N, e.v);
    }
    N++;

    // init GPU parameters
    initGPU(edge_num, N);

    // degree of each vertex.
    uint32_t* deg = new uint32_t[N];
    degreeCollect(0, edgeList, edge_num, deg, N);

#ifdef DEBUG_DF
    cout << "Edge Num: " << edge_num << endl;
    cout << "Vertex Num: " << N << endl;
    tmpTime.print("Degree Calculate Time Cost");
    tmpTime.check();
#endif

    // redirect edges
#pragma omp parallel for
    for (off_t i = 0; i < edge_num; ++i) {
        uint32_t u = edgeList[i].u;
        uint32_t v = edgeList[i].v;

        if (deg[u] > deg[v] || (deg[u] == deg[v] && u > v)) {
            edgeList[i].u = v;
            edgeList[i].v = u;
        }
    }

#ifdef DEBUG_DF
    tmpTime.print("Edge Redirect Time Cost");
    tmpTime.check();
#endif

    // construct adj list
    uint32_t* nbr_arr = new uint32_t[edge_num];
    uint32_t* nbr_u = new uint32_t[N];
    uint32_t* nbr_size = deg;
    
    // calculate out degree and the index of adj list
    degreeCollect(1, edgeList, edge_num, deg, N);

    uint64_t nbr_index = 0;
    for (off_t i = 0; i < N; ++i) {
        nbr_u[i] = nbr_index;
        nbr_index += deg[i];
    }

    // construct Raw AdjList(unordered)
    adjListConstruct(edgeList, edge_num, nbr_u, nbr_arr, N);

#ifdef DEBUG_DF
    tmpTime.print("Raw AdjList construct Time Cost");
    tmpTime.check();
#endif

    // release memory (deg == nbr_size, donot free again)
    // delete[] deg;

    // sort and deduplicate
#pragma omp parallel for schedule(dynamic, 256)
    for (uint32_t i = 0; i < N; ++i) {
        auto arr_begin = nbr_arr + nbr_u[i];
        auto arr_end = arr_begin + nbr_size[i];

        sort(arr_begin, arr_end);
        auto result = unique(arr_begin, arr_end);

        nbr_size[i] = result - arr_begin;
    }

#ifdef DEBUG_DF
    tmpTime.print("Sort-Deduplicate Time Cost");
    tmpTime.check();
#endif

    // statistics node index
    uint64_t edge_index = 0;
    uint32_t* nodeIndex = new uint32_t[N + 1];
    uint32_t* adjList = (uint32_t*) edgeList;

    for (uint32_t i = 0; i < N; ++i){
        nodeIndex[i] = edge_index;
        // nodeSize[i] = nbr_vec[i].size();
        edge_index += nbr_size[i];
    }
    nodeIndex[N] = edge_index;

#ifdef DEBUG_DF
    tmpTime.print("Node Index Time Cost");
    tmpTime.check();
#endif

#pragma omp parallel for schedule(dynamic, 256)
    for (uint32_t i = 0; i < N; ++i){
        uint32_t adjSize = nbr_size[i];
        uint32_t adjBegin = nodeIndex[i];
        for(uint32_t j = 0; j < adjSize; ++j){
            adjList[j + adjBegin] = nbr_arr[nbr_u[i] + j];
        }
    }

    // release memory
    delete[] nbr_arr;
    delete[] nbr_size;
    delete[] nbr_u;

#ifdef DEBUG_DF
    tmpTime.print("adjList build Time Cost");
    preProcessTime.print("Preprocessing Time Cost");
    tmpTime.check();
#endif

    /*---------count triangle---------*/
    // Global aggregator
    unsigned long long sum = tricount(N, (const uint32_t*) nodeIndex, (const uint32_t*) adjList);

    cout << "There are " << sum << " triangles in the input graph.\n";

#ifdef DEBUG_DF
    tmpTime.print("Counting Time Cost");
    allTime.print("All Time Cost");
#endif

    return 0;
}
