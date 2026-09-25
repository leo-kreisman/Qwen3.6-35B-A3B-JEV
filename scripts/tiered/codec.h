#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <array>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <tmmintrin.h>
#endif

// Q4_K metadata is retained verbatim. Each block chooses one of these nibble
// codebooks; 256 two-bit indices replace 256 four-bit indices. 81 vs 144 bytes.
// This is a lossy STORAGE codec, not a native Q2_K arithmetic format.
namespace jev_tiered {
inline constexpr uint8_t books[][4] = {
    {0,5,10,15}, {1,5,9,13}, {2,6,10,14}, {3,6,9,12},
    {4,6,8,10}, {5,7,9,11}, {6,8,10,12}, {4,7,10,13},
    {0,3,6,9}, {6,9,12,15}, {0,2,4,6}, {9,11,13,15},
    {2,5,8,11}, {4,7,10,15}, {0,5,8,11}, {5,8,11,14}
};
inline constexpr auto pair_luts = [] {
    std::array<std::array<uint8_t,16>,16> result{};
    for(unsigned b=0;b<16;++b) for(unsigned j=0;j<16;++j)
        result[b][j]=books[b][j&3] | (books[b][j>>2]<<4);
    return result;
}();
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("ssse3"))) inline bool expand_simd(const uint8_t * src, uint8_t * dst, size_t blocks) {
    const __m128i mask=_mm_set1_epi8(15);
    for(size_t b=0;b<blocks;++b,src+=81,dst+=144) {
        if(src[16]>=16) return false;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst),_mm_loadu_si128(reinterpret_cast<const __m128i *>(src)));
        __m128i lut=_mm_loadu_si128(reinterpret_cast<const __m128i *>(pair_luts[src[16]].data()));
        for(size_t j=0;j<64;j+=16) {
            __m128i v=_mm_loadu_si128(reinterpret_cast<const __m128i *>(src+17+j));
            __m128i lo=_mm_shuffle_epi8(lut,_mm_and_si128(v,mask));
            __m128i hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(v,4),mask));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst+16+2*j),_mm_unpacklo_epi8(lo,hi));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst+32+2*j),_mm_unpackhi_epi8(lo,hi));
        }
    }
    return true;
}
#endif
inline bool expand(const uint8_t * src, size_t stored, uint8_t * dst, size_t raw) {
    if (raw % 144 || stored != raw / 144 * 81) return false;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if(__builtin_cpu_supports("ssse3")) return expand_simd(src,dst,raw/144);
#endif
    for (size_t b = 0; b < raw / 144; ++b, src += 81, dst += 144) {
        if (src[16] >= 16) return false;
        std::memcpy(dst, src, 16);
        const auto & lut=pair_luts[src[16]];
        for (size_t j=0;j<64;++j) {
            dst[16+2*j] = lut[src[17+j]&15];
            dst[17+2*j] = lut[src[17+j]>>4];
        }
    }
    return true;
}
inline void pack(const uint8_t * src, size_t blocks, uint8_t * dst) {
    uint8_t maps[16][16];
    uint8_t errors[16][16];
    for (unsigned c=0;c<16;++c) for (int x=0;x<16;++x) {
        unsigned best=0; int cost=1000;
        for (unsigned k=0;k<4;++k) {
            int d=x-books[c][k];
            if (d*d < cost) {cost=d*d; best=k;}
        }
        maps[c][x]=best; errors[c][x]=cost;
    }
    for (size_t b=0;b<blocks;++b,src+=144,dst+=81) {
        unsigned hist[16]={};
        for (size_t j=16;j<144;++j) {++hist[src[j]&15]; ++hist[src[j]>>4];}
        unsigned best=0, cost=~0u;
        for (unsigned c=0;c<16;++c) {
            unsigned e=0;
            for (unsigned x=0;x<16;++x) e+=hist[x]*errors[c][x];
            if(e<cost){cost=e;best=c;}
        }
        std::memcpy(dst,src,16); dst[16]=best;
        const auto * m=maps[best];
        for(size_t j=0;j<64;++j) {
            uint8_t a=src[16+2*j], d=src[17+2*j];
            dst[17+j]=m[a&15] | (m[a>>4]<<2) | (m[d&15]<<4) | (m[d>>4]<<6);
        }
    }
}
}
