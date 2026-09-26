// Version-sensitive CPU graph interception experiment, NOT a supported skip API.
// Every temporarily suppressed operator is restored at its synchronized callback.
#pragma once
#include "native_tiles.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <array>
#include <sys/mman.h>

class NativeBypass {
    NativeTiles & tiles;
    enum Role { Gate, Up, Activation, Down, Other };
    ggml_tensor * pending=nullptr;
    ggml_op saved_op=GGML_OP_NONE;
    Role saved_role=Other;
    ggml_tensor * projections[2]={nullptr,nullptr};
    ggml_tensor * activation=nullptr;
    ggml_tensor * input=nullptr;
    ggml_tensor * routes=nullptr;
    std::vector<float> result;
    int n=0,k=0;
    std::array<uint64_t,4> skipped{};
    uint64_t groups=0,protected_bytes=0;
    bool protect_original;
    std::vector<std::pair<void*,size_t>> protected_ranges;
    void protect(const ggml_tensor * weight) {
        if(!protect_original)return;
        if(!weight->data||!weight->buffer||!ggml_backend_buffer_is_host(weight->buffer))
            throw std::runtime_error("weight guard requires host-mapped tensors");
        size_t page=sysconf(_SC_PAGESIZE);uintptr_t begin=reinterpret_cast<uintptr_t>(weight->data);
        uintptr_t end=(begin+ggml_nbytes(weight))/page*page;begin=(begin+page-1)/page*page;
        if(end<=begin)throw std::runtime_error("no whole weight pages to protect");
        for(auto &r:protected_ranges)if(r.first==reinterpret_cast<void*>(begin))return;
        if(mprotect(reinterpret_cast<void*>(begin),end-begin,PROT_NONE))throw std::runtime_error("weight access guard failed");
        protected_ranges.push_back({reinterpret_cast<void*>(begin),end-begin});protected_bytes+=end-begin;
    }
    template<class T> static std::vector<T> read(const ggml_tensor * t,ggml_type type) {
        if(t->type!=type)throw std::runtime_error("bypass input dtype mismatch");
        // Router IDs are often a strided view into a larger sorted index array.
        std::vector<uint8_t> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());
        std::vector<T> out;out.reserve(ggml_nelements(t));
        for(int64_t i3=0;i3<t->ne[3];++i3)for(int64_t i2=0;i2<t->ne[2];++i2)
        for(int64_t i1=0;i1<t->ne[1];++i1)for(int64_t i0=0;i0<t->ne[0];++i0) {
            size_t offset=i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3];
            if(offset+sizeof(T)>raw.size())throw std::runtime_error("bypass input stride overflow");
            T value;std::memcpy(&value,raw.data()+offset,sizeof(T));out.push_back(value);
        }
        return out;
    }
    Role role(const ggml_tensor * t) const {
        std::string suffix="-"+std::to_string(tiles.layer),name=t->name;
        if(name=="ffn_moe_gate"+suffix)return Gate;
        if(name=="ffn_moe_up"+suffix)return Up;
        if(name=="ffn_moe_swiglu"+suffix)return Activation;
        if(name=="ffn_moe_down"+suffix)return Down;
        return Other;
    }
    void validate(ggml_tensor * t,Role r) {
        if(t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t)||t->ne[3]!=1)
            throw std::runtime_error("unsupported bypass tensor layout");
        if(r==Gate||r==Up||r==Down) {
            if(t->op!=GGML_OP_MUL_MAT_ID||!t->src[0]||!t->src[1]||!t->src[2])
                throw std::runtime_error("bypass requires direct expert projection");
            const char * part=r==Gate?"gate":r==Up?"up":"down";
            std::string expected="blk."+std::to_string(tiles.layer)+".ffn_"+part+"_exps.weight";
            if(t->src[0]->name!=expected||t->src[0]->ne[2]!=tiles.experts||
               t->src[0]->type!=tiles.weight_type(r==Down?2:int(r))||
               t->src[0]->ne[0]!=(r==Down?tiles.m:tiles.d)||t->src[0]->ne[1]!=(r==Down?tiles.d:tiles.m))
                throw std::runtime_error("bypass weight identity mismatch");
            if(r==Down) {
                if(t->src[1]!=activation||t->src[2]!=routes||t->ne[0]!=tiles.d||t->ne[1]!=k||t->ne[2]!=n||result.empty())
                    throw std::runtime_error("bypass down dependency mismatch");
            } else {
                if(t->src[1]->ne[0]!=tiles.d||t->src[1]->ne[1]!=1||t->src[1]->ne[3]!=1||t->ne[0]!=tiles.m)
                    throw std::runtime_error("bypass projection geometry mismatch");
                if(projections[r]||(input&&(input!=t->src[1]||routes!=t->src[2])))
                    throw std::runtime_error("bypass projection inputs changed");
            }
        } else if(r==Activation) {
            if(t->op!=GGML_OP_GLU||ggml_get_glu_op(t)!=GGML_GLU_OP_SWIGLU||
               !projections[Gate]||!projections[Up]||t->src[0]!=projections[Gate]||t->src[1]!=projections[Up])
                throw std::runtime_error("bypass activation dependency mismatch");
        }
    }
public:
    explicit NativeBypass(NativeTiles & executor,const std::string & model,bool protect_weights=false):tiles(executor),protect_original(protect_weights) {
        auto * metadata=gguf_init_from_file(model.c_str(),{true,nullptr});
        if(!metadata)throw std::runtime_error("cannot inspect model metadata");
        bool unsupported=false;
        // The current seam supports plain, split SwiGLU projections only.
        for(const char * part:{"gate","up","down"})for(const char * extra:{"scale","input_scale","bias"}) {
            std::string name="blk."+std::to_string(tiles.layer)+".ffn_"+part+"_exps."+extra;
            unsupported|=gguf_find_tensor(metadata,name.c_str())>=0;
        }
        gguf_free(metadata);
        if(unsupported)throw std::runtime_error("bypass does not support expert scales or biases");
    }
    ~NativeBypass() { restore();unprotect(); }
    void unprotect() {
        for(auto &r:protected_ranges)mprotect(r.first,r.second,PROT_READ);
        protected_ranges.clear();
    }
    bool interested(const ggml_tensor * t) const { return pending==t||role(t)!=Other; }
    void restore() { if(pending)pending->op=saved_op;pending=nullptr;saved_role=Other; }
    bool visit(ggml_tensor * t,bool ask) {
        if(ask) {
            if(pending)throw std::runtime_error("overlapping bypass callbacks");
            Role r=role(t);validate(t,r);
            if(r==Gate||r==Up||r==Down)protect(t->src[0]);
            pending=t;saved_op=t->op;saved_role=r;
            t->op=GGML_OP_NONE; // CPU dispatch returns without touching weights.
            return true;       // End this graph segment and synchronize.
        }
        if(pending!=t)throw std::runtime_error("missing synchronized bypass callback");
        Role r=saved_role;restore();++skipped[r];
        if(r==Gate||r==Up) {
            projections[r]=t;
            if(!input) {
                input=t->src[1];routes=t->src[2];n=input->ne[2];k=routes->ne[0];
                result=tiles.evaluate(read<float>(input,GGML_TYPE_F32),read<int32_t>(routes,GGML_TYPE_I32),n,k);
            }
        } else if(r==Activation)activation=t;
        else if(r==Down) {
            if(result.size()!=size_t(ggml_nelements(t)))throw std::runtime_error("bypass result size mismatch");
            for(float value:result)if(!std::isfinite(value))throw std::runtime_error("nonfinite bypass result");
            ggml_backend_tensor_set(t,result.data(),0,result.size()*sizeof(float));
            ++groups;projections[0]=projections[1]=nullptr;activation=input=routes=nullptr;
            std::vector<float>().swap(result);n=k=0;
        }
        return true;
    }
    bool idle() const { return !pending&&!input&&result.empty(); }
    nlohmann::json stats() const { return {{"groups",groups},{"skipped_gate",skipped[Gate]},
        {"skipped_up",skipped[Up]},{"skipped_activation",skipped[Activation]},
        {"skipped_down",skipped[Down]},{"operators_restored",idle()},
        {"experimental_graph_interception",true},{"protected_original_weight_bytes",protected_bytes}}; }
};
