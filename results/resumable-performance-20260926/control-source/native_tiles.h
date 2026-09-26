// Experimental ready-row executor using the existing CPU quantized dot kernels.
// Only current, routed rows are admitted; no predicted future routes are used.
#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

class NativeTiles {
    using json = nlohmann::json;
    struct Block { size_t offset, length[3]; int start; };
    json manifest;
    std::vector<std::vector<Block>> blocks;
    ggml_type types[3];
    int fd=-1, threads;
    bool preserve_reduction;
    void * buffer=nullptr; size_t capacity=0;
    static constexpr size_t work_limit=256ULL<<20;
    void read(const Block & b) {
        size_t size=b.length[0]+b.length[1]+b.length[2], count=(size+4095)/4096*4096;
        if(count>capacity) {
            free(buffer);buffer=nullptr;capacity=0;
            if(posix_memalign(&buffer,4096,count))throw std::runtime_error("tile I/O allocation failed");
            capacity=count;
        }
        ssize_t got=pread(fd,buffer,count,b.offset);
        if(got<0 || size_t(got)<size)throw std::runtime_error("short direct tile read");
        read_bytes+=got;++read_calls;
    }
    static size_t row_bytes(ggml_type type,int n) { return ggml_row_size(type,n); }
    static void quant(ggml_type weight,const float * x,int n,void * out) {
        auto * traits=ggml_get_type_traits_cpu(weight);
        ggml_get_type_traits_cpu(traits->vec_dot_type)->from_float(x,out,n);
    }
    static void activate(float * gate,float * up,float * result,int width,int rows) {
        // Use the same SIMD activation as the native graph, rather than scalar exp.
        ggml_init_params ip{ggml_graph_overhead_custom(8,false)+4*ggml_tensor_overhead()+1024,nullptr,true};
        auto * ctx=ggml_init(ip);if(!ctx)throw std::runtime_error("activation graph allocation failed");
        auto * g=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows);
        auto * u=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows);
        g->data=gate;u->data=up;
        auto * h=ggml_swiglu_split(ctx,g,u);h->data=result;
        auto * graph=ggml_new_graph_custom(ctx,8,false);ggml_build_forward_expand(graph,h);
        auto plan=ggml_graph_plan(graph,1,nullptr);
        std::vector<uint8_t> work(plan.work_size);plan.work_data=work.data();
        auto status=ggml_graph_compute(graph,&plan);ggml_free(ctx);
        if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("native activation failed");
    }
    static float dot(ggml_type type,int n,const void * weight,const void * x) {
        float value=0;ggml_get_type_traits_cpu(type)->vec_dot(n,&value,0,weight,0,x,0,1);return value;
    }
public:
    int layer,d,m,experts,tile;
    uint64_t read_bytes=0,read_calls=0,expert_visits=0,routed_rows=0;
    size_t peak_work_bytes=0;
    double compute_seconds=0;
    explicit NativeTiles(const std::string & directory,int nthreads=6,bool staged_down=true):threads(nthreads),preserve_reduction(staged_down) {
        std::ifstream(directory+"/manifest.json")>>manifest;
        if(manifest.at("format")!="jev-executable-tiles-v1")throw std::runtime_error("unsupported tile format");
        layer=manifest.at("layer");d=manifest.at("hidden");m=manifest.at("intermediate");
        experts=manifest.at("experts");tile=manifest.at("tile");
        if(d<=0||m<=0||experts<=0||experts>65536||tile<=0||m%tile||tile%256||d%256||threads<1)
            throw std::runtime_error("unsupported tile geometry");
        const char * names[]={"gate","up","down"};
        for(int i=0;i<3;++i) {
            int t=manifest.at("tensors").at(names[i]).at("qtype");
            if(t!=GGML_TYPE_Q4_K&&t!=GGML_TYPE_Q5_K&&t!=GGML_TYPE_Q6_K)
                throw std::runtime_error("unsupported quantization type");
            types[i]=ggml_type(t);
            const auto *tr=ggml_get_type_traits_cpu(types[i]);
            if(!tr->vec_dot||!ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float)
                throw std::runtime_error("missing quantized CPU kernel");
        }
        blocks.resize(experts);
        for(const auto & b:manifest.at("blocks")) {
            int e=b.at("expert");Block out{};out.offset=b.at("offset");out.start=b.at("start");
            if(e<0||e>=experts||out.offset%4096)throw std::runtime_error("invalid tile address");
            for(int i=0;i<3;++i) {
                out.length[i]=b.at("lengths").at(i);
                size_t expected=i==2?d*row_bytes(types[i],tile):tile*row_bytes(types[i],d);
                if(out.length[i]!=expected)throw std::runtime_error("tile geometry/length mismatch");
            }
            blocks[e].push_back(out);
        }
        for(const auto & list:blocks) {
            if(list.size()!=size_t(m/tile))throw std::runtime_error("missing tiles");
            for(size_t j=0;j<list.size();++j)if(list[j].start!=int(j)*tile)throw std::runtime_error("unordered tiles");
        }
        fd=open((directory+"/weights.bin").c_str(),O_RDONLY|O_DIRECT);
        if(fd<0)throw std::runtime_error("cannot open direct tile reader");
    }
    void validate_model_path(const std::string & model) const {
        if(!manifest.contains("model")||!std::filesystem::equivalent(model,manifest.at("model").get<std::string>()))
            throw std::runtime_error("tile pack source model differs");
    }
    ggml_type weight_type(int projection) const { return types[projection]; }
    NativeTiles(const NativeTiles&)=delete;
    ~NativeTiles() { if(fd>=0)close(fd);free(buffer); }
    std::vector<float> evaluate(const std::vector<float> & x,const std::vector<int32_t> & ids,int n,int k) {
        auto start=std::chrono::steady_clock::now();
        if(n<=0||k<=0||k>experts||n>4096||x.size()!=size_t(n)*d||ids.size()!=size_t(n)*k)
            throw std::runtime_error("invalid ready input geometry");
        size_t output_bytes=size_t(n)*k*d*sizeof(float);
        size_t qxbytes[2]={row_bytes(ggml_get_type_traits_cpu(types[0])->vec_dot_type,d),
                           row_bytes(ggml_get_type_traits_cpu(types[1])->vec_dot_type,d)};
        size_t qhbytes=row_bytes(ggml_get_type_traits_cpu(types[2])->vec_dot_type,tile);
        // Conservative: count caller input/routes, output, admission directory,
        // both quantized inputs, worst-case expert temporaries and direct I/O.
        size_t io=0;for(auto & list:blocks)for(auto & b:list)io=std::max(io,(b.length[0]+b.length[1]+b.length[2]+4095)/4096*4096);
        size_t work=(1ULL<<20)+size_t(n)*k*tile*8+x.size()*4+ids.size()*8+output_bytes+size_t(n)*(qxbytes[0]+qxbytes[1])+size_t(n)*k*(tile*4+qhbytes)+io;
        if(preserve_reduction)work+=size_t(n)*k*(m*4+row_bytes(ggml_get_type_traits_cpu(types[2])->vec_dot_type,m))+d*row_bytes(types[2],m);
        if(work>work_limit)throw std::runtime_error("ready-work memory budget exceeded");
        peak_work_bytes=std::max(peak_work_bytes,work);
        std::vector<std::vector<int>> ready(experts);
        for(size_t i=0;i<ids.size();++i) {
            if(ids[i]<0||ids[i]>=experts)throw std::runtime_error("invalid routed expert");
            ready[ids[i]].push_back(i);
        }
        std::vector<uint8_t> qx[2];
        for(int t=0;t<2;++t) {
            qx[t].resize(size_t(n)*qxbytes[t]);
            #pragma omp parallel for num_threads(threads)
            for(int row=0;row<n;++row)quant(types[t],x.data()+size_t(row)*d,d,qx[t].data()+row*qxbytes[t]);
        }
        std::vector<float> output(size_t(n)*k*d,0);
        for(int e=0;e<experts;++e) {
            const auto & rows=ready[e];if(rows.empty())continue;
            ++expert_visits;routed_rows+=rows.size();
            std::vector<float> hidden(rows.size()*tile),gates(rows.size()*tile),ups(rows.size()*tile);
            std::vector<uint8_t> qhidden(rows.size()*qhbytes);
            size_t down_full_stride=row_bytes(types[2],m);
            size_t qfull_stride=row_bytes(ggml_get_type_traits_cpu(types[2])->vec_dot_type,m);
            std::vector<float> full_hidden(preserve_reduction?rows.size()*m:0);
            std::vector<uint8_t> full_down(preserve_reduction?d*down_full_stride:0);
            std::vector<uint8_t> qfull(preserve_reduction?rows.size()*qfull_stride:0);
            for(const auto & b:blocks[e]) {
                read(b);
                auto * gate=static_cast<const uint8_t*>(buffer);
                const auto *up=gate+b.length[0], *down=up+b.length[1];
                size_t gate_stride=row_bytes(types[0],d),up_stride=row_bytes(types[1],d),down_stride=row_bytes(types[2],tile);
                #pragma omp parallel for num_threads(threads)
                for(size_t r=0;r<rows.size();++r) {
                    int input_row=rows[r]/k;
                    for(int j=0;j<tile;++j) {
                        float g=dot(types[0],d,gate+j*gate_stride,qx[0].data()+input_row*qxbytes[0]);
                        float u=dot(types[1],d,up+j*up_stride,qx[1].data()+input_row*qxbytes[1]);
                        gates[r*tile+j]=g;ups[r*tile+j]=u;
                    }
                }
                activate(gates.data(),ups.data(),hidden.data(),tile,rows.size());
                if(!preserve_reduction) {
                    #pragma omp parallel for num_threads(threads)
                    for(size_t r=0;r<rows.size();++r)quant(types[2],hidden.data()+r*tile,tile,qhidden.data()+r*qhbytes);
                }
                if(preserve_reduction) {
                    for(size_t r=0;r<rows.size();++r)
                        std::memcpy(full_hidden.data()+r*m+b.start,hidden.data()+r*tile,tile*sizeof(float));
                    for(int j=0;j<d;++j)
                        std::memcpy(full_down.data()+j*down_full_stride+(b.start/tile)*down_stride,down+j*down_stride,down_stride);
                } else {
                    #pragma omp parallel for num_threads(threads)
                    for(size_t r=0;r<rows.size();++r)for(int j=0;j<d;++j)
                        output[size_t(rows[r])*d+j]+=dot(types[2],tile,down+j*down_stride,qhidden.data()+r*qhbytes);
                }
            }
            if(preserve_reduction) {
                #pragma omp parallel for num_threads(threads)
                for(size_t r=0;r<rows.size();++r) {
                    quant(types[2],full_hidden.data()+r*m,m,qfull.data()+r*qfull_stride);
                    for(int j=0;j<d;++j)output[size_t(rows[r])*d+j]=dot(types[2],m,full_down.data()+j*down_full_stride,qfull.data()+r*qfull_stride);
                }
            }
        }
        compute_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return output;
    }
    json stats() const { return {{"direct_read_bytes",read_bytes},{"read_calls",read_calls},
        {"expert_visits",expert_visits},{"routed_rows",routed_rows},{"peak_accounted_work_bytes",peak_work_bytes},
        {"work_limit_bytes",work_limit},{"decoded_weight_bytes",0},{"preserve_down_reduction",preserve_reduction},{"seconds",compute_seconds}}; }
};
