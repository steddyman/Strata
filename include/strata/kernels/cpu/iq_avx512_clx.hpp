// include/strata/kernels/cpu/iq_avx512_clx.hpp - the Cascade Lake build of the i-quant rows (src/kernels/cpu/iq_avx512_clx.cpp)
#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels::cpu {
bool iq512x_supported_clx(int ggml_type) noexcept;   // this CPU has what the build needs (AVX-512 F/DQ/BW/VL/VNNI, BMI2) and the type is covered
void iq512x_rows_clx(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                     int r0, int r1);
}  // namespace strata::kernels::cpu
