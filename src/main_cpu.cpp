#include <iostream>
#include <fstream>
#include <cstring>
#include <omp.h>

#include <algorithm>
#include <random>
#include <chrono>

#include "bsr.h"
#include "mapfile.hpp"

using namespace std;
// #define DEBUG_DF

// size must be 8
struct Edge_t {
    uint32_t u;
    uint32_t v;
} __attribute__ ((aligned (4)));

typedef vector<uint32_t> nbr_t;

#ifdef DEBUG_DF
# include <sys/time.h>
struct timeval tp, tp_end, tp_res;
int e = gettimeofday(&tp, NULL);
#endif

uint32_t N = 0;


int main(int argc, char *argv[]) {
    if (argc != 3 || strcmp(argv[1], "-f") != 0) {
        cout << "Usage: -f [data_file_path]" << endl;
        exit(1);
    }
    const char *filepath = argv[2];

#ifdef DEBUG_DF
    cout << "Start reading file" << endl;
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

    uint64_t edge_num = mapfile.getLen() / sizeof(Edge_t);

#ifdef DEBUG_DF
    cout << "Edge Num: " << edge_num << endl;
#endif

#ifdef DEBUG_DF
    struct timeval build_edge_bt, build_edge_et;
    e = gettimeofday(&build_edge_bt, NULL);
#endif

    // find number of vertex
#pragma omp parallel for reduction(max : N)
    for (off_t i = 0; i < edge_num; ++i) {
        Edge_t &e = edgeList[i];

        N = max(N, e.u);
        N = max(N, e.v);
    }
    N++;

    // degree of each vertex.
    vector<uint32_t> deg(N);
#pragma omp parallel for
    for (off_t i = 0; i < edge_num; ++i) {
        Edge_t &e = edgeList[i];
        deg[e.u]++;
        deg[e.v]++;
    }

    /*---------build edges list---------*/
    // filter edge
#pragma omp parallel for
    for (off_t i = 0; i < edge_num; ++i) {
        uint32_t u = edgeList[i].u;
        uint32_t v = edgeList[i].v;

        if (deg[u] > deg[v] || (deg[u] == deg[v] && u > v)) {
            edgeList[i].u = v;
            edgeList[i].v = u;
        }
    }

    // release memory
    deg.clear();

    // construct adj list
    std::vector<nbr_t> nbr_vec(N);

    for (off_t i = 0; i < edge_num; ++i) {
        uint32_t &u = edgeList[i].u;
        uint32_t &v = edgeList[i].v;
        if (u == v) {
            continue;
        }
        nbr_vec[u].push_back(v);
    }

    // deduplicate
#pragma omp parallel for schedule(dynamic, 256)
    for (uint32_t i = 0; i < N; ++i) {
        auto &vec = nbr_vec[i];

        sort(vec.begin(), vec.end());
        auto result = unique(vec.begin(), vec.end());

        vec.resize(distance(vec.begin(), result));
    }

    uint64_t edge_index = 0;
    for (uint32_t i = 0; i < N; ++i) {
        for (auto j: nbr_vec[i]) {
            edgeList[edge_index].u = i;
            edgeList[edge_index].v = j;
            ++edge_index;
        }
    }

    uint64_t filter_edge_num = edge_index;


#ifdef DEBUG_DF
    cout << N << endl;
    cout << "End reading file" << endl;
#endif

    // construct bsr format
    std::vector<BSR> nbr_bsr(N);

#pragma omp parallel for schedule(dynamic, 256)
    for (uint32_t i = 0; i < N; ++i) {
        nbr_bsr[i].init(nbr_vec[i]);
        // release nbr_vec memory
        nbr_vec[i].clear();
    }

#ifdef DEBUG_DF
    e = gettimeofday(&build_edge_et, NULL);
    cout << edge_index << " " << filter_edge_num << endl;
    timersub(&build_edge_et, &build_edge_bt, &tp_res);
    cout << "Loading Time Cost: " << tp_res.tv_sec << " s " << tp_res.tv_usec << " us.\n";
#endif

    /*---------count triangle---------*/
    // Global aggregator
    unsigned long long sum = 0;

#pragma omp parallel for reduction(+:sum) schedule(dynamic, 1024)
    for (uint64_t i = 0; i < filter_edge_num; ++i) {
        auto &e = edgeList[i];
        sum += nbr_bsr[e.u] & nbr_bsr[e.v];
    }

    cout << "There are " << sum << " triangles in the input graph.\n";
#ifdef DEBUG_DF
    e = gettimeofday(&tp_end, NULL);
    
    timersub(&tp_end, &build_edge_et, &tp_res);
    cout << "Counting Time Cost: " << tp_res.tv_sec << " s " << tp_res.tv_usec << " us.\n";
    
    timersub(&tp_end, &tp, &tp_res);
    cout << "Total Time Cost: " << tp_res.tv_sec << " s " << tp_res.tv_usec << " us.\n";
#endif

    return 0;
}
