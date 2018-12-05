#ifndef MAPFILE_HPP
#define MAPFILE_HPP

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

class MapFile{
public:
    MapFile(const char* filepath){
        struct stat statbuf;

        if((fdin = open(filepath, O_RDONLY)) < 0){
            perror("open file error");
            exit(-1);
        }

        if((fstat(fdin, &statbuf)) < 0){
            perror("get file length error");
            exit(-1);
        }

        len = statbuf.st_size;

        if((src = mmap(0, len, PROT_READ, MAP_SHARED, fdin, 0)) == \
            (void *) -1){
            perror("mmap file error");
            exit(-1);
        }
    }

    void release(){
        if(fdin != -1){
            close(fdin);
            munmap(src, len);
        }
        fdin = -1;
        src = NULL;
    }

    ~MapFile(){
        release();
    }

    off_t getLen(){
        return len;
    }

    void* getAddr(){
        return src;
    }

private:
    int fdin;
    void* src;
    off_t len;
};

#endif
