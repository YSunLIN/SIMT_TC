#include <iostream>
#include "bsr.h"

using namespace std;

BSR::BSR(){
    bases = NULL;
    n = 0;
}

void BSR::init(const std::vector<uint32_t>& vec){
    this->release();
    this->n = vec.size();
    this->bases = new uint32_t[this->n];
    for (int i=0;i<n;i++)
        this->bases[i] = vec[i];
}


void BSR::release(){
    if (!bases) return;

    delete[] bases;
    bases = NULL;
    n = 0;
}

uint32_t BSR::operator&(const BSR& b){
    uint32_t i = 0, j = 0, res = 0;
    
    while (i < this->n && j < b.n) {
        res += (this->bases[i] == b.bases[j]);
        uint32_t tmp = (this->bases[i] >= b.bases[j]);
        i += (this->bases[i] <= b.bases[j]);
        j += tmp;
    }

    return res;
}