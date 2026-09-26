#include "native_tiles.h"
#include "native_bypass.h"
#include <filesystem>
#include <iostream>
#include <random>
using json=nlohmann::json;
static void require(bool ok,const char * reason) { if(!ok)throw std::runtime_error(reason); }
static std::vector<uint8_t> quantize(ggml_type type,const std::vector<float>& data,int width) {
    std::vector<uint8_t> out(data.size()/width*ggml_row_size(type,width));
    for(size_t row=0;row<data.size()/width;++row)
        ggml_get_type_traits(type)->from_float_ref(data.data()+row*width,out.data()+row*ggml_row_size(type,width),width);
    return out;
}
int main(int argc,char **argv) try {
    if(argc!=2)throw std::runtime_error("usage: native-tests NEW_DIRECTORY");
    std::filesystem::path root=argv[1];require(std::filesystem::create_directory(root),"new output required");
    ggml_cpu_init();int tests=0;
    for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K}) {
        const int d=256,m=512,e=2,n=4;
        auto dir=root/std::to_string(type);std::filesystem::create_directory(dir);
        std::mt19937 rng(17);std::uniform_real_distribution<float> distribution(-0.05f,0.05f);
        std::vector<uint8_t> weights[2][3];
        for(int expert=0;expert<e;++expert)for(int kind=0;kind<3;++kind) {
            std::vector<float> v(d*m);for(float &x:v)x=distribution(rng);
            weights[expert][kind]=quantize(type,v,kind==2?m:d);
        }
        json manifest={{"format","jev-executable-tiles-v1"},{"layer",0},{"hidden",d},{"intermediate",m},{"experts",e},{"tile",256},{"blocks",json::array()}};
        for(auto name:{"gate","up","down"})manifest["tensors"][name]["qtype"]=int(type);
        std::ofstream file(dir/"weights.bin",std::ios::binary);
        for(int expert=0;expert<e;++expert)for(int start:{0,256}) {
            size_t off=file.tellp(),padding=(-off)%4096;file.write(std::string(padding,'\0').data(),padding);off+=padding;
            json lengths=json::array();
            for(int kind=0;kind<3;++kind) {
                const auto & w=weights[expert][kind];size_t rowbytes=ggml_row_size(type,256);size_t size=256*rowbytes;lengths.push_back(size);
                if(kind<2)file.write(reinterpret_cast<const char*>(w.data()+start*rowbytes),size);
                else for(int row=0;row<d;++row)file.write(reinterpret_cast<const char*>(w.data()+row*2*rowbytes+(start/256)*rowbytes),rowbytes);
            }
            manifest["blocks"].push_back({{"expert",expert},{"start",start},{"offset",off},{"lengths",lengths}});
        }
        size_t padding=(-size_t(file.tellp()))%4096;file.write(std::string(padding,'\0').data(),padding);file.close();
        std::ofstream(dir/"manifest.json")<<manifest;
        std::vector<float>x(n*d);for(float&v:x)v=distribution(rng);
        std::vector<int32_t>ids={1,0,1,0};
        NativeTiles grouped(dir.string(),2,true);auto output=grouped.evaluate(x,ids,n,1);
        NativeTiles serial(dir.string(),2,true);
        for(int row=0;row<n;++row) {
            std::vector<float>one(x.begin()+row*d,x.begin()+(row+1)*d);
            auto y=serial.evaluate(one,{ids[row]},1,1);
            require(std::equal(y.begin(),y.end(),output.begin()+row*d),"ready grouping changed values");
        }
        require(grouped.read_calls*2==serial.read_calls,"ready grouping did not reuse tiles");++tests;
        // Reuse the same activation graph across shrinking/growing row counts,
        // and across invocations with different backing-buffer addresses.
        auto again=grouped.evaluate(x,ids,n,1);require(again==output,"repeat changed values");
        std::vector<float> first(x.begin(),x.begin()+d);
        auto small=grouped.evaluate(first,{ids[0]},1,1);
        require(std::equal(small.begin(),small.end(),output.begin()),"shrunk graph changed values");
        auto twice_x=x;twice_x.insert(twice_x.end(),x.begin(),x.end());
        auto twice_ids=ids;twice_ids.insert(twice_ids.end(),ids.begin(),ids.end());
        auto large=grouped.evaluate(twice_x,twice_ids,2*n,1);
        require(std::equal(output.begin(),output.end(),large.begin())&&std::equal(output.begin(),output.end(),large.begin()+output.size()),"grown graph changed values");
        require(grouped.activation_graph_builds==1&&grouped.input_quantized_rows==uint64_t(4*n+1),"setup was not reused");++tests;
        NativeTiles pipeline(dir.string(),2,true,true);
        require(pipeline.evaluate(x,ids,n,1)==output,"overlap changed values");
        require(pipeline.evaluate(twice_x,twice_ids,2*n,1)==large,"overlap reuse changed values");
        require(pipeline.read_calls==8,"overlap read count differs");++tests;
        // Native whole-expert graph is an independent reference for both gate
        // activation and the down reduction; use F32 input so ggml quantizes it.
        for(int row=0;row<n;++row) {
            ggml_init_params ip{ggml_graph_overhead_custom(16,false)+16*ggml_tensor_overhead()+1024,nullptr,true};
            auto *ctx=ggml_init(ip);auto * input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d,1);input->data=x.data()+row*d;
            ggml_tensor *w[3];
            for(int k=0;k<3;++k) {w[k]=ggml_new_tensor_2d(ctx,type,k==2?m:d,k==2?d:m);w[k]->data=weights[ids[row]][k].data();}
            auto *g=ggml_mul_mat(ctx,w[0],input),*u=ggml_mul_mat(ctx,w[1],input);
            auto *h=ggml_swiglu_split(ctx,g,u),*o=ggml_mul_mat(ctx,w[2],h);
            std::vector<float>gv(m),uv(m),hv(m),ov(d);g->data=gv.data();u->data=uv.data();h->data=hv.data();o->data=ov.data();
            auto *graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,o);
            auto plan=ggml_graph_plan(graph,1,nullptr);std::vector<uint8_t>work(plan.work_size);plan.work_data=work.data();
            require(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS,"reference graph failed");
            for(int j=0;j<d;++j)require(std::abs(ov[j]-output[row*d+j])<1e-7f,"native reference differs");
            ggml_free(ctx);
        }++tests;
        bool rejected=false;try {grouped.evaluate(x,{2,0,1,0},n,1);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"invalid expert accepted");++tests;
        // The interception seam must restore operators on normal cleanup and
        // reject unsupported graphs before mutating their operations.
        auto metadata_path=(dir/"model.gguf").string();auto * meta=gguf_init_empty();
        require(gguf_write_to_file(meta,metadata_path.c_str(),true),"metadata fixture write failed");gguf_free(meta);
        ggml_init_params mip{8*ggml_tensor_overhead()+1024,nullptr,true};auto * mc=ggml_init(mip);
        auto * mw=ggml_new_tensor_3d(mc,type,d,m,e);ggml_set_name(mw,"blk.0.ffn_gate_exps.weight");
        auto * mx=ggml_new_tensor_3d(mc,GGML_TYPE_F32,d,1,n);
        auto * mi=ggml_new_tensor_2d(mc,GGML_TYPE_I32,1,n);
        auto * mg=ggml_mul_mat_id(mc,mw,mx,mi);ggml_set_name(mg,"ffn_moe_gate-0");
        {
            NativeBypass bypass(grouped,metadata_path);
            require(bypass.interested(mg),"target operation missed");
            require(bypass.visit(mg,true)&&mg->op==GGML_OP_NONE,"operation not suppressed");
            bypass.restore();require(mg->op==GGML_OP_MUL_MAT_ID&&bypass.idle(),"explicit restore failed");
            bypass.visit(mg,true);
        }
        require(mg->op==GGML_OP_MUL_MAT_ID,"destructor did not restore operation");++tests;
        {
            NativeBypass bypass(grouped,metadata_path);mg->op=GGML_OP_SQR;bool failed=false;
            try {bypass.visit(mg,true);}catch(const std::runtime_error&){failed=true;}
            require(failed&&mg->op==GGML_OP_SQR&&bypass.idle(),"unsupported graph was mutated");
        }
        ggml_free(mc);++tests;
        manifest["blocks"].erase(0);std::ofstream(dir/"manifest.json")<<manifest;
        rejected=false;try {NativeTiles invalid(dir.string());}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"missing tile accepted");++tests;
        // A producer error must wake a waiting consumer and join cleanly.
        std::filesystem::resize_file(dir/"weights.bin",4096);
        rejected=false;try {pipeline.evaluate(x,ids,n,1);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"short asynchronous read accepted");++tests;
    }
    std::cout<<json({{"checks",tests},{"passed",true}}).dump()<<'\n';return 0;
} catch(const std::exception&e) {std::cerr<<e.what()<<'\n';return 1;}
