#include "tiered_reader.h"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc,char ** argv) {
    if(argc!=5) return 2;
    bmoe::TieredReader reader;
    bool opened=reader.open(argv[1],argv[2],2,true,{{0,144}});
    if(std::string(argv[4])=="reject") return opened ? 1 : 0;
    if(!opened || !reader.contains(0) || reader.contains(1)) return 3;
    std::ifstream f(argv[3],std::ios::binary);
    std::vector<unsigned char> expected((std::istreambuf_iterator<char>(f)),{}), actual(144);
    for(int lane=0;lane<2;++lane) {
        if(reader.read(lane,actual.data(),0,144)<0 || actual!=expected) return 4;
    }
    if(reader.read(0,actual.data(),0,143)>=0) return 5;
    if(reader.read(2,actual.data(),0,144)>=0) return 6;
    if(!reader.reopen() || reader.read(0,actual.data(),0,144)<0 || actual!=expected) return 7;
    std::puts("tiered reader gate passed");
}
