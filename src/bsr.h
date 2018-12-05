#ifndef BSR_H
#define BSR_H

#include <x86intrin.h>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <algorithm>
#include <map>
#include <vector>

// BSR Format of neighbor list of a vertex. 

class BSR{
public:
    BSR();

    void init(const std::vector<uint32_t>& vec);
    uint32_t operator&(const BSR& b);

    void release();

    ~BSR(){
        release();
    }

private:
    uint32_t n;
    uint32_t* bases;
};

#endif
