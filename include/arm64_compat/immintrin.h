// include/arm64_compat/immintrin.h - the x86 intrinsics the host loops use, on arm64 Linux (NVIDIA DGX Spark / GB10,
// Grace): the spin-wait hint and the store fence before a host flag is raised (verify.cpp, session.cpp, mtp.cpp,
// graph.cpp, pool.cpp).  CMakeLists.txt puts this folder first on the include path only for an aarch64 build; the x86
// kernel files that need the real header are left out of that build (STRATA_ARM64).
//
// The fence is "dmb oshst", not the Metal build's "dmb ishst": the flag it orders is read by the GPU, and on Grace the
// GPU is in the outer-shareable domain.
#pragma once
static inline void _mm_pause() { __asm__ __volatile__("yield"); }
static inline void _mm_sfence() { __asm__ __volatile__("dmb oshst" ::: "memory"); }
