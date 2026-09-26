// Standalone CPU experiment. Bypass uses version-sensitive graph interception.
// Does not patch llama.cpp source or the streamer.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "native_tiles.h"
#include "native_bypass.h"
#include <memory>
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
static uint64_t reads() {
    std::ifstream f("/proc/self/io"); std::string k; uint64_t v;
    while (f >> k >> v) if (k == "read_bytes:") return v;
    throw std::runtime_error("no process I/O accounting");
}
template<class T> static std::vector<T> values(const ggml_tensor * t, ggml_type type) {
    if (t->type != type) throw std::runtime_error("unexpected capture dtype");
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<T> out; out.reserve(ggml_nelements(t));
    for(int64_t i3=0;i3<t->ne[3];++i3) for(int64_t i2=0;i2<t->ne[2];++i2)
    for(int64_t i1=0;i1<t->ne[1];++i1) for(int64_t i0=0;i0<t->ne[0];++i0) {
        size_t off=i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3];
        if(off+sizeof(T)>raw.size()) throw std::runtime_error("strided capture overflow");
        T v; std::memcpy(&v,raw.data()+off,sizeof(T)); out.push_back(v);
    }
    return out;
}
struct Trace {
    bool diagnostics=true;
    std::ofstream file; std::string dir, phase, error; int call=0, record=0, capture_layer=-1;
    Clock::time_point mark; uint64_t before=0;
    std::unique_ptr<NativeTiles> tiles;
    std::unique_ptr<NativeBypass> bypass;
    bool replace_tiles=false;
    std::vector<float> ready_x;
    std::vector<int32_t> ready_ids;
    int ready_n=0,ready_k=0,detail_record=0;
    json tile_checks=json::array();
    static bool callback(ggml_tensor * t, bool ask, void * p) {
        auto & s=*static_cast<Trace*>(p);
        try {
            if(s.bypass&&s.bypass->interested(t))return s.bypass->visit(t,ask);
        } catch(const std::exception & e) {
            if(s.bypass)s.bypass->restore();
            s.error=e.what();return false;
        }
        // Bypass control is required; route/tensor diagnostics are optional.
        // check/replace still needs its validation captures.
        if(!s.diagnostics && (!s.tiles || s.bypass))return !ask;
        // The up projection exposes normalized expert inputs and actual route IDs.
        bool wanted=t->op==GGML_OP_MUL_MAT_ID && t->src[0] &&
                    std::strstr(t->src[0]->name,".ffn_up_exps.weight");
        int node_layer=-1;
        if(t->src[0])std::sscanf(t->src[0]->name,"blk.%d.",&node_layer);
        bool tile_down=s.tiles && node_layer==s.tiles->layer && t->op==GGML_OP_MUL_MAT_ID &&
                       t->src[0] && std::strstr(t->src[0]->name,".ffn_down_exps.weight");
        std::string node_name=t->name;
        bool detail=s.capture_layer==-3&&node_name.size()>2&&node_name.substr(node_name.size()-2)=="-3"&&
                    t->type==GGML_TYPE_F32&&ggml_nelements(t)<2*1024*1024;
        if(ask) return wanted||tile_down||detail;
        if(!wanted&&!tile_down&&!detail) return true;
        try {
            if(detail) {
                auto data=values<float>(t,GGML_TYPE_F32);
                std::string name="tensor-"+std::to_string(s.detail_record++)+".f32";
                std::ofstream f(s.dir+"/"+name,std::ios::binary);
                f.write(reinterpret_cast<char*>(data.data()),data.size()*sizeof(float));
                if(!f)throw std::runtime_error("detail capture failed");
                s.file<<json({{"kind","tensor"},{"name",node_name},{"phase",s.phase},
                    {"shape",std::vector<int64_t>(t->ne,t->ne+4)},{"input_file",name}}).dump()<<'\n';
                if(!wanted&&!tile_down)return true;
            }
            if(tile_down) {
                if(t->ne[0]!=s.tiles->d||t->ne[1]!=s.ready_k||t->ne[2]!=s.ready_n||t->ne[3]!=1||!ggml_is_contiguous(t))
                    throw std::runtime_error("tile down geometry differs");
                if(values<int32_t>(t->src[2],GGML_TYPE_I32)!=s.ready_ids)throw std::runtime_error("routes changed before down");
                auto original=values<float>(t,GGML_TYPE_F32);
                auto replacement=s.tiles->evaluate(s.ready_x,s.ready_ids,s.ready_n,s.ready_k);
                double max_error=0,squared=0,energy=0;
                for(size_t i=0;i<original.size();++i) {
                    if(!std::isfinite(replacement[i])||!std::isfinite(original[i]))throw std::runtime_error("nonfinite tile comparison");
                    double delta=double(replacement[i])-original[i];
                    max_error=std::max(max_error,std::abs(delta));squared+=delta*delta;energy+=double(original[i])*original[i];
                }
                s.tile_checks.push_back({{"phase",s.phase},{"rows",s.ready_n},{"max_abs_error",max_error},
                    {"relative_l2_error",std::sqrt(squared/std::max(energy,1e-30))},{"replaced",s.replace_tiles}});
                if(s.replace_tiles && max_error!=0.0)throw std::runtime_error("tile replacement refused: native output differs");
                if(s.replace_tiles)ggml_backend_tensor_set(t,replacement.data(),0,replacement.size()*sizeof(float));
                s.ready_x.clear();s.ready_ids.clear();s.ready_n=0;
                return true;
            }
            int layer=-1; std::sscanf(t->src[0]->name,"blk.%d.",&layer);
            auto ids=values<int32_t>(t->src[2],GGML_TYPE_I32);
            if(s.tiles&&layer==s.tiles->layer) {
                if(t->src[1]->ne[0]!=s.tiles->d||t->src[1]->ne[1]!=1||t->src[1]->ne[3]!=1||s.ready_n)
                    throw std::runtime_error("unsupported ready-row geometry");
                s.ready_x=values<float>(t->src[1],GGML_TYPE_F32);s.ready_ids=ids;
                s.ready_n=t->src[1]->ne[2];s.ready_k=t->src[2]->ne[0];
            }
            auto now=reads();
            json j={{"phase",s.phase},{"call",s.call},{"record",s.record++},{"layer",layer},
                    {"input_shape",std::vector<int64_t>(t->src[1]->ne,t->src[1]->ne+4)},
                    {"ids_shape",std::vector<int64_t>(t->src[2]->ne,t->src[2]->ne+4)},
                    {"ids",ids},{"seconds_since_decode_start",seconds(s.mark)},
                    {"process_read_bytes_since_decode_start",now-s.before}};
            if(layer==s.capture_layer||s.capture_layer==-2) {
                auto x=values<float>(t->src[1],GGML_TYPE_F32);
                std::string name="input-"+std::to_string(s.record-1)+".f32";
                std::ofstream f(s.dir+"/"+name,std::ios::binary);
                f.write(reinterpret_cast<char*>(x.data()),x.size()*sizeof(float));
                if(!f) throw std::runtime_error("capture write failed");
                j["input_file"]=name;
            }
            s.file << j.dump() << '\n';
            if(!s.file) throw std::runtime_error("trace write failed");
        } catch(const std::exception & e) { s.error=e.what(); return false; }
        return true;
    }
};
int main(int argc,char **argv) try {
    if(argc<5) throw std::runtime_error("usage: probe MODEL INPUT_JSON OUTPUT_DIR full|serial|shared|full-padded|shared-padded|split-serial|restore-serial [trace_layer|-1|-2|-3] [threads] [pack check|replace|bypass staged|tiled]");
    std::string mode=argv[4];
    if(mode!="full"&&mode!="serial"&&mode!="shared"&&mode!="full-padded"&&mode!="shared-padded"&&mode!="split-serial"&&mode!="restore-serial") throw std::runtime_error("unknown mode");
    bool padded=mode=="full-padded"||mode=="shared-padded";
    std::filesystem::path dir=argv[3];
    if(!std::filesystem::create_directory(dir)) throw std::runtime_error("output directory must be new");
    json input; std::ifstream(argv[2])>>input;
    const auto & rows=input.at("rows"); int n=rows.size(), prefix=input.at("prefix_tokens");
    if(n<1||n>32||prefix<1) throw std::runtime_error("invalid fixture geometry");
    std::vector<std::vector<llama_token>> tokens;
    int total=0;
    for(const auto & row:rows) { tokens.push_back(row.at("tokens").get<std::vector<llama_token>>()); total+=tokens.back().size(); }
    for(const auto & t:tokens) if(prefix>=(int)t.size()||!std::equal(t.begin(),t.begin()+prefix,tokens[0].begin()))
        throw std::runtime_error("invalid prefix");
    Trace trace; trace.dir=dir.string(); bool tracing=argc>5 && input.value("diagnostics",true);
    trace.diagnostics=tracing;
    if(tracing) { trace.capture_layer=std::stoi(argv[5]); trace.file.open(dir/"routes.jsonl"); }
    if(argc>7) {
        if(argc>9&&std::string(argv[9])!="staged"&&std::string(argv[9])!="tiled")
            throw std::runtime_error("tile reduction must be staged or tiled");
        trace.tiles=std::make_unique<NativeTiles>(argv[7],argc>6?std::stoi(argv[6]):6,argc<=9||std::string(argv[9])=="staged");
        trace.tiles->validate_model_path(argv[1]);
        if(argc<9||(std::string(argv[8])!="check"&&std::string(argv[8])!="replace"&&std::string(argv[8])!="bypass"))
            throw std::runtime_error("tile action must be check, replace or bypass");
        trace.replace_tiles=std::string(argv[8])=="replace";
        if(std::string(argv[8])=="bypass") {
            if(argc>9&&std::string(argv[9])!="staged")throw std::runtime_error("bypass requires staged reduction");
            trace.bypass=std::make_unique<NativeBypass>(*trace.tiles,argv[1],input.value("protect_bypassed_weights",false));
        }
    }
    llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=0; mp.use_extra_bufts=false; mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    auto started=Clock::now(); auto rb=reads();
    auto * model=llama_model_load_from_file(argv[1],mp);
    if(!model) throw std::runtime_error("model load failed");
    if(trace.tiles) {
        char architecture[128];
        llama_model_meta_val_str(model,"general.architecture",architecture,sizeof(architecture));
        if(std::string(architecture)!="qwen35moe")throw std::runtime_error("live tile validation currently supports qwen35moe only");
    }
    const auto * vocab=llama_model_get_vocab(model);
    for(int i=0;i<n;++i) {
        std::string text=rows[i].at("prompt"); std::vector<llama_token> check(tokens[i].size()+32);
        int written=llama_tokenize(vocab,text.data(),text.size(),check.data(),check.size(),false,true);
        if(written!=(int)tokens[i].size()||!std::equal(tokens[i].begin(),tokens[i].end(),check.begin()))
            throw std::runtime_error("tokenizer parity failed");
    }
    int longest=0;for(const auto&t:tokens)longest=std::max(longest,(int)t.size());
    auto cp=llama_context_default_params(); cp.n_ctx=std::max(4096,longest*n+256); cp.n_seq_max=n;
    cp.n_outputs_max=n; cp.n_outputs_max_per_seq=1;
    cp.n_batch=cp.n_ubatch=std::max(1024,longest*n); cp.n_threads=cp.n_threads_batch=argc>6?std::stoi(argv[6]):6;
    cp.offload_kqv=false; cp.op_offload=false;
    std::string flash=input.value("flash_attention",std::string("auto"));
    if(flash!="auto"&&flash!="disabled")throw std::runtime_error("invalid flash_attention experiment setting");
    if(flash=="disabled")cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if(tracing||trace.tiles) { cp.cb_eval=Trace::callback; cp.cb_eval_user_data=&trace; }
    auto * ctx=llama_init_from_model(model,cp);
    if(!ctx) throw std::runtime_error("context load failed");
    json report={{"mode",mode},{"tracing",tracing},{"flash_attention",flash},{"load_seconds",seconds(started)},
                 {"load_read_bytes",reads()-rb},{"prefix_tokens",prefix},{"branches",n},
                 {"n_ctx",llama_n_ctx(ctx)},{"n_ubatch",llama_n_ubatch(ctx)},
                 {"phases",json::array()},{"answers",json::array()}};
    auto begin=Clock::now(); rb=reads();
    auto decode=[&](const std::vector<int>& seqs,int offset,bool logits,const std::string & phase) {
        int count=0; for(int seq:seqs) count+=logits?(padded?longest:(int)tokens[seq].size())-offset:prefix;
        auto batch=llama_batch_init(count,0,1); batch.n_tokens=count; int q=0;
        for(int seq:seqs) {
            int real_end=logits?tokens[seq].size():prefix;
            int end=logits&&padded?longest:real_end;
            for(int j=offset;j<end;++j,++q) {
                // Right padding is causally AFTER the requested readout. It aligns
                // recurrent batch lengths; it is not a masked input substitution.
                batch.token[q]=j<real_end?tokens[seq][j]:llama_vocab_eos(vocab); batch.pos[q]=j;
                batch.n_seq_id[q]=1; batch.seq_id[q][0]=seq;
                batch.logits[q]=logits&&j==real_end-1;
            }
        }
        auto t=Clock::now(); auto r=reads(); trace.phase=phase; ++trace.call; trace.mark=t; trace.before=r;
        int rc=llama_decode(ctx,batch);
        report["phases"].push_back({{"phase",phase},{"tokens",count},{"seconds",seconds(t)},{"read_bytes",reads()-r}});
        if(trace.bypass&&!trace.bypass->idle()) {
            trace.bypass->restore();if(trace.error.empty())trace.error="incomplete bypass group";
        }
        if(rc||!trace.error.empty()) { llama_batch_free(batch); throw std::runtime_error("decode failed: "+std::to_string(rc)+" "+trace.error); }
        if(logits) { int index=0;
            for(int seq:seqs) {
                int answer_index=index+tokens[seq].size()-offset-1;
                index+=(padded?longest:(int)tokens[seq].size())-offset;
                const float * z=llama_get_logits_ith(ctx,answer_index);
                if(!z) throw std::runtime_error("missing logits");
                std::vector<double> chosen;
                for(int slot:rows[seq].at("slots")) chosen.push_back(z[slot]);
                double high=*std::max_element(chosen.begin(),chosen.end()),sum=0; std::vector<double> p;
                for(double v:chosen) { if(!std::isfinite(v)) throw std::runtime_error("nonfinite logits"); p.push_back(std::exp(v-high)); sum+=p.back(); }
                for(double &v:p) v/=sum;
                report["answers"].push_back({{"id",rows[seq].at("id")},{"logits",chosen},{"probabilities",p}});
            }
        }
        llama_batch_free(batch);
    };
    std::vector<int> seqs; for(int i=0;i<n;++i)seqs.push_back(i);
    int repetitions=input.value("repetitions",1);
    if(repetitions<1||repetitions>8)throw std::runtime_error("invalid repetition count");
    report["iterations"]=json::array();
    for(int repetition=0;repetition<repetitions;++repetition) {
    if(repetition)llama_memory_clear(llama_get_memory(ctx),true);
    report["answers"]=json::array();
    if(mode=="full"||mode=="full-padded") decode(seqs,0,true,"full");
    else if(mode=="serial") {
        for(int i=0;i<n;++i) { llama_memory_clear(llama_get_memory(ctx),true); decode({i},0,true,"serial-"+std::to_string(i)); }
    } else if(mode=="split-serial"||mode=="restore-serial") {
        for(int i=0;i<n;++i) {
            llama_memory_clear(llama_get_memory(ctx),true);
            decode({i},0,false,"prefix-"+std::to_string(i));
            if(mode=="restore-serial") {
                size_t size=llama_state_seq_get_size(ctx,i); std::vector<uint8_t> saved(size);
                if(!size||llama_state_seq_get_data(ctx,saved.data(),size,i)!=size) throw std::runtime_error("snapshot failed");
                llama_memory_clear(llama_get_memory(ctx),true);
                if(llama_state_seq_set_data(ctx,saved.data(),size,i)!=size) throw std::runtime_error("restore failed");
            }
            decode({i},prefix,true,"suffix-"+std::to_string(i));
        }
    } else {
        decode({0},0,false,"prefix");
        auto t=Clock::now(); size_t size=llama_state_seq_get_size(ctx,0);
        if(!size)throw std::runtime_error("empty snapshot");
        std::vector<uint8_t> saved(size);
        if(llama_state_seq_get_data(ctx,saved.data(),size,0)!=size)throw std::runtime_error("snapshot failed");
        llama_memory_clear(llama_get_memory(ctx),true);
        for(int seq:seqs) if(llama_state_seq_set_data(ctx,saved.data(),size,seq)!=size)
            throw std::runtime_error("restore failed for sequence "+std::to_string(seq));
        report["snapshot_bytes"]=size; report["snapshot_and_branch_seconds"]=seconds(t);
        // Release the serialized temporary before suffix compute.
        std::vector<uint8_t>().swap(saved);
        decode(seqs,prefix,true,"suffix");
    }
    report["iterations"].push_back(report["answers"]);
    }
    if(trace.tiles) { report["native_tiles"]=trace.tiles->stats();report["tile_checks"]=trace.tile_checks;
        report["tile_validation_only"]=!trace.bypass;
        if(trace.bypass)report["native_bypass"]=trace.bypass->stats(); }
    report["scoring_seconds"]=seconds(begin); report["scoring_read_bytes"]=reads()-rb;
    std::ofstream(dir/"result.json")<<report.dump(2)<<'\n';
    std::cout<<report.dump()<<'\n';
    if(trace.bypass)trace.bypass->unprotect();
    llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
} catch(const std::exception &e) { std::cerr<<"probe: "<<e.what()<<'\n'; return 1; }
