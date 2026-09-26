// Native layer replay using saved real inputs/routes. No whole-model speed claim.
#ifndef NATIVE_TILES_HEADER
#define NATIVE_TILES_HEADER "native_tiles.h"
#endif
#include NATIVE_TILES_HEADER
#include <filesystem>
#include <iostream>
using json=nlohmann::json;
int main(int argc,char **argv) try {
    if(argc!=6)throw std::runtime_error("usage: layer-benchmark PACK TRACE NEW_OUTPUT ROUNDS sync|overlap");
    std::filesystem::path pack=argv[1],trace=argv[2],out=argv[3];
    int rounds=std::stoi(argv[4]);std::string mode=argv[5];
    if(rounds<1||(mode!="sync"&&mode!="overlap"))throw std::runtime_error("invalid benchmark settings");
    if(!std::filesystem::create_directory(out))throw std::runtime_error("output must be new");
    ggml_cpu_init();
#ifdef TILE_ASYNC_SUPPORTED
    NativeTiles tiles(pack.string(),6,true,mode=="overlap");
#else
    if(mode!="sync")throw std::runtime_error("control only supports sync");
    NativeTiles tiles(pack.string(),6,true);
#endif
    struct Group {std::vector<float>x;std::vector<int32_t>ids;int n,k;};
    std::vector<Group> groups;std::ifstream f(trace);std::string line;
    while(std::getline(f,line)) {
        auto r=json::parse(line);
        if(r.value("layer",-1)!=tiles.layer||!r.contains("input_file"))continue;
        Group g;g.n=r.at("input_shape").at(2);g.ids=r.at("ids").get<std::vector<int32_t>>();
        if(g.n<=0||g.ids.size()%g.n)throw std::runtime_error("invalid trace geometry");
        g.k=g.ids.size()/g.n;g.x.resize(size_t(g.n)*tiles.d);
        std::ifstream data(trace.parent_path()/r.at("input_file").get<std::string>(),std::ios::binary);
        data.read(reinterpret_cast<char*>(g.x.data()),g.x.size()*sizeof(float));
        if(!data||data.peek()!=std::char_traits<char>::eof())throw std::runtime_error("invalid captured input length");
        groups.push_back(std::move(g));
    }
    if(groups.empty())throw std::runtime_error("no captured groups");
    json report={{"mode",mode},{"rounds",rounds},{"groups",groups.size()},{"seconds",json::array()},
                 {"scope","native single-layer replay; captured readiness; O_DIRECT reads; not end-to-end inference"}};
    std::vector<std::vector<float>> reference;
    for(int round=0;round<rounds;++round) {
        auto start=std::chrono::steady_clock::now();
        for(size_t i=0;i<groups.size();++i) {
            auto & g=groups[i];auto y=tiles.evaluate(g.x,g.ids,g.n,g.k);
            for(float v:y)if(!std::isfinite(v))throw std::runtime_error("nonfinite output");
            if(!round)reference.push_back(std::move(y));
            else if(y!=reference[i])throw std::runtime_error("repeated output changed");
        }
        report["seconds"].push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    }
    std::ofstream values(out/"outputs.f32",std::ios::binary);
    for(auto & y:reference)values.write(reinterpret_cast<const char*>(y.data()),y.size()*sizeof(float));
    if(!values)throw std::runtime_error("output write failed");
    report["native_tiles"]=tiles.stats();std::ofstream(out/"result.json")<<report.dump(2)<<'\n';
    std::cout<<report.dump()<<'\n';return 0;
} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
