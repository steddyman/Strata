// src/kernels/cpu/arm64_stubs.cpp - the x86 CPU expert kernels' entry points on an arm64 Linux build (NVIDIA DGX Spark,
// GB10).  expert.cpp, q2_avx*.cpp, iq_avx*.cpp, kq_avx*.cpp, nvfp4_avx*.cpp and tq_avx*.cpp are AVX / AVX-512 code and
// are not compiled there (CMakeLists.txt, STRATA_ARM64).  include/arm64_compat/cpuid.h makes every x86 feature probe
// read "no", so the engine takes ggml-cpu's own rows (NEON) for a native pack's CPU experts and never calls these.
//
// The probes a dispatcher asks return "no"; every kernel says which one was reached and stops, so a run never goes on
// with outputs nobody wrote.  The canonical (non-native) Q2_0 pack still needs an AVX-512 CPU and is refused at start
// (generate.cpp, cpu_require_expert_support), as on any x86 CPU without it.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <cstdio>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#error "arm64_stubs.cpp is for the arm64 build only: an x86 build compiles the real kernels"
#endif

namespace {
[[noreturn]] void x86_only(const char* name) {
    std::fprintf(stderr,
                 "strata: %s is an x86 (AVX) CPU kernel and this is an arm64 build.\n"
                 "        Use a native pack (setup prepares one) or keep every expert on the GPU.\n",
                 name);
    std::fflush(stderr);
    std::_Exit(86);
}
}  // namespace

namespace strata::kernels::cpu {

// ---- expert.cpp (AVX-512) and q2_avx2.cpp / q2_avx1.cpp: the canonical Q2_0 pack's rows
void act_quant_q8_1(const float*, int, ActQ&) { x86_only("act_quant_q8_1"); }
void act_quant_q8_1_avx2(const float*, int, ActQ&) { x86_only("act_quant_q8_1_avx2"); }
void act_quant_q8_1_avx1(const float*, int, ActQ&) { x86_only("act_quant_q8_1_avx1"); }
void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { x86_only("s2_expert_vnni"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { x86_only("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { x86_only("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { x86_only("s2_expert_down_rows"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    x86_only("s2_expert_gu_rows_multi");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    x86_only("s2_expert_down_rows_multi");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    x86_only("s2_expert_vnni_multi");
}
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { x86_only("s2_expert_scalar"); }
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi_avx2");
}
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi_avx2_legacy");
}
void q2_0_gguf_rows_multi_avx2_v(bool, const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi_avx2_v");
}
void q2_0_gguf_rows_multi_avx1(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi_avx1");
}
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi_avx1_legacy");
}
bool q2_bitplane_enabled() { return false; }

// ---- iq_avx1.cpp / iq_avx2.cpp / iq_avx512.cpp: the i-quant rows (iq*_supported live in native_expert.cpp)
bool iq128_supported(int) noexcept { return false; }
bool iq128_down_supported(int) noexcept { return false; }
void iq128_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq128_gu_rows");
}
void iq128_down_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq128_down_rows");
}
void q8k_quant_avx1(const float*, void*, int64_t) { x86_only("q8k_quant_avx1"); }
void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    x86_only("iq256_gu_rows_exact_one");
}
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_gu_rows");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_rows");
}
int iq256_variant() noexcept { return 0; }
int iq256_variants() noexcept { return 0; }
void iq256_gu_rows_v(int, int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_gu_rows_v");
}
void iq256_rows_v(int, int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_rows_v");
}
void q8k_quant_avx2(const float*, void*, int64_t) { x86_only("q8k_quant_avx2"); }
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq4nl256_down_rows");
}
void iq4nl256_down_rows_v(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq4nl256_down_rows_v");
}
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq512_gu_rows");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq512_rows");
}

// ---- kq_avx1.cpp / kq_avx2.cpp: the K-quant rows and the router lookahead's bf16 dot
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("kq256_gu_rows");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("kq256_rows");
}
void bf16_rows_dot(const uint16_t*, int, int, const float*, float*) { x86_only("bf16_rows_dot"); }
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) { x86_only("bf16_rows_dot_multi"); }
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    x86_only("bf16_rows_dot_multi_avx1");
}

// ---- nvfp4_avx2.cpp / nvfp4_avx512.cpp: the NVFP4 rows
bool nvfp4_256_fits(int) { return false; }
bool nvfp4_512_fits(int) { return false; }
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    x86_only("nvfp4_256_gu_rows");
}
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    x86_only("nvfp4_256_rows");
}
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    x86_only("nvfp4_512_gu_rows");
}
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    x86_only("nvfp4_512_rows");
}

// ---- tq_avx2.cpp / tq_avx512.cpp: the trellis rows (tq_cpu.cpp takes its scalar rows off x86)
namespace tq_detail {
void tq_rows_avx2(int, const uint8_t*, size_t, int, const TqAct* const*, int, float* const*, int, int) {
    x86_only("tq_rows_avx2");
}
void tq_rows_avx512(int, const uint8_t*, size_t, int, const TqAct* const*, int, float* const*, int, int) {
    x86_only("tq_rows_avx512");
}
}  // namespace tq_detail

}  // namespace strata::kernels::cpu
