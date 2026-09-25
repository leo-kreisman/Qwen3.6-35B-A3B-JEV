#include "codec.h"
extern "C" int jev_pack(const uint8_t * src, size_t n, uint8_t * dst) {
    if(n%144) return 0;
    jev_tiered::pack(src,n/144,dst); return 1;
}
extern "C" int jev_expand(const uint8_t * src, size_t n, uint8_t * dst, size_t raw) {
    return jev_tiered::expand(src,n,dst,raw);
}
