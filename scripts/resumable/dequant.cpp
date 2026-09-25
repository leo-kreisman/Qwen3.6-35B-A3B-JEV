#include "ggml.h"
#include <cstdint>
extern "C" int resume_dequant(int type, const void *src, int64_t n, float *out) {
    if (type<0 || type>=GGML_TYPE_COUNT || n<=0) return 0;
    auto *traits=ggml_get_type_traits(static_cast<ggml_type>(type));
    if(!traits || !traits->to_float || n%traits->blck_size) return 0;
    traits->to_float(src,out,n); return 1;
}
