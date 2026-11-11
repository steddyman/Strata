// src/kernels/cpu/iq_avx512_clx.cpp - iq_avx512.cpp built for the AVX-512 CPUs without VBMI / VPOPCNTDQ (Intel Skylake-SP and
// Cascade Lake Xeons: the Xeon W-3200 of a Mac Pro 2019): the same rows, every function renamed *_clx, the sign masks' parity without
// popcount (STRATA_IQ512_CLX), compiled with exactly F / DQ / BW / VL / VNNI + BMI2 (cmake/backends/cpu.cmake) so no newer
// instruction can appear in it.  iq512x_rows_clx is ggml-cpu's dot bit for bit like iq512x_rows (the DeepSeek lane checks it, own_ok).
#define STRATA_IQ512_CLX 1
#define iq512_supported iq512_supported_clx
#define iq512_gather_on iq512_gather_on_clx
#define iq512_gu_rows iq512_gu_rows_clx
#define iq512_rows iq512_rows_clx
#define iq512x_supported iq512x_supported_clx
#define iq512x_rows iq512x_rows_clx
#define iq512_reference_gu_rows iq512_reference_gu_rows_clx
#define iq512f_supported iq512f_supported_clx
#define iq512_fused_gu_rows iq512_fused_gu_rows_clx
#define q50_512_down_rows q50_512_down_rows_clx
#include "iq_avx512.cpp"
