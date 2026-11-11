// src/qwen35/q36_g906.cuh - gfx906 (Vega 20, wave64) decode kernels of the Qwen3.6 engine path.
//
// The Qwen3.8 gfx906 lessons applied to q36_engine.cu's small decode launches: a 64-lane wavefront a row, every load of
// a lane issued before the first use (an empty asm takes the loaded registers at once; without it the compiler moves
// each load beside its use and waits on it there), and the decode columns (a verify window's tokens) inside the kernel
// so a weight row is read once for all of them.  Each column's sum is computed the same way whatever the column count,
// so a verify window's column equals the one-token result (the MTP drafts keep the greedy text).  The order differs
// from the warp-per-row kernels', so outputs match them to float rounding, not bitwise.
#pragma once
#if defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>
#include <cstdlib>

namespace q36g {

__device__ __forceinline__ float wave_sum64(float v) {
#pragma unroll
    for (int m = 32; m > 0; m >>= 1) v += __shfl_xor(v, m, 64);
    return v;
}

inline bool on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_Q36_G906"); return !(e && e[0] == '0'); }();
    return v;
}

/// y[c][o] = W[o] . x[c] (f32 W [n_out][NI]), NC columns: a wavefront a row, 4 rows a block, NI / 256 float4 loads a
/// lane.  EPI 0: plain; 1: the GDN alpha / beta epilogue (rows [0, HV) softplus(acc + dt) * ssm_a into y, rows
/// [HV, 2 HV) sigmoid(acc) into y2; W rows [HV, 2 HV) are Wb's).
template<int NC, int NI, int EPI>
__global__ void __launch_bounds__(256) k_gemv_f32(const float* __restrict__ W, const float* __restrict__ Wb,
                                                  const float* __restrict__ x, float* __restrict__ y, float* __restrict__ y2,
                                                  int n_out, const float* __restrict__ dt, const float* __restrict__ ssm_a,
                                                  int HV) {
    constexpr int PL = NI / 256;   // float4 a lane
    const int lane = int(threadIdx.x) & 63;
    const int o = int(blockIdx.x) * 4 + int(threadIdx.x >> 6);
    if (o >= n_out) return;
    const float* wrow = (EPI == 1 && o >= HV) ? Wb + (size_t) (o - HV) * NI : W + (size_t) o * NI;
    const float4* wr = (const float4*) wrow;
    float4 w[PL];
#pragma unroll
    for (int i = 0; i < PL; ++i) w[i] = wr[lane + 64 * i];
#pragma unroll
    for (int i = 0; i < PL; ++i) asm volatile("" : "+v"(w[i].x), "+v"(w[i].y), "+v"(w[i].z), "+v"(w[i].w));
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float4* xr = (const float4*) (x + (size_t) c * NI);
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            const float4 b = xr[lane + 64 * i];
            acc += w[i].x * b.x + w[i].y * b.y + w[i].z * b.z + w[i].w * b.w;
        }
        acc = wave_sum64(acc);
        if (lane == 0) {
            if (EPI == 0) y[(size_t) c * n_out + o] = acc;
            else if (o >= HV) y2[(size_t) c * HV + (o - HV)] = 1.0f / (1.0f + expf(-acc));
            else {
                const float v = acc + dt[o];
                y[(size_t) c * HV + o] = (v > 20.0f ? v : log1pf(expf(v))) * ssm_a[o];
            }
        }
    }
}

template<int EPI>
inline bool gemv_f32(const float* W, const float* Wb, const float* x, float* y, float* y2, int n_in, int n_out, int T,
                     const float* dt, const float* ssm_a, int HV, cudaStream_t st) {
    if (!on() || n_in != 2048 || T < 1 || T > 8) return false;
    const unsigned g = (unsigned) ((n_out + 3) / 4);
#define Q36G_GO(NC) k_gemv_f32<NC, 2048, EPI><<<g, 256, 0, st>>>(W, Wb, x, y, y2, n_out, dt, ssm_a, HV)
    switch (T) {
        case 1: Q36G_GO(1); break;
        case 2: Q36G_GO(2); break;
        case 3: Q36G_GO(3); break;
        case 4: Q36G_GO(4); break;
        case 5: Q36G_GO(5); break;
        case 6: Q36G_GO(6); break;
        case 7: Q36G_GO(7); break;
        default: Q36G_GO(8); break;
    }
#undef Q36G_GO
    return true;
}

/// Two q8_1 elements a lane (elements i0 and i0 + 1024 of a row, blocks i >> 5) with their shuffle chains interleaved:
/// q36k::q8_store's arithmetic for each.
__device__ __forceinline__ void q8_store2(q36k::QB* __restrict__ y, int i0, float v0, float v1, int lane) {
    float a0 = fabsf(v0), a1 = fabsf(v1), s0 = v0, s1 = v1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o, 32));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o, 32));
        s0 += __shfl_xor_sync(0xffffffffu, s0, o, 32);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o, 32);
    }
    const float amax[2] = {a0, a1}, sum[2] = {s0, s1}, v[2] = {v0, v1};
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int i = i0 + 1024 * k;
        const float d = strata::kernels::q8_1_finite(q36k::fdiv_fm(amax[k], 127.0f));
        int8_t q = 0;
        if (amax[k] != 0.0f) {
            const float qf = roundf(q36k::fdiv_fm(v[k], d));
            q = (int8_t) (qf > 127.0f ? 127.0f : qf < -127.0f ? -127.0f : qf);
        }
        y[i >> 5].qs[lane] = q;
        if (lane == 0) y[i >> 5].ds = strata::kernels::q8_1_ds(d, sum[k]);
    }
}

/// The block's sum of one value a thread (1024 threads): wavefront sums, then the 16 in a fixed order in every thread.
__device__ __forceinline__ float block_sum1024(float v, float* ws) {
    v = wave_sum64(v);
    if ((threadIdx.x & 63) == 0) ws[threadIdx.x >> 6] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) t += ws[w];
    return t;
}

/// q36k::k_add_rms_q for n == 2048: x += y; h = rms(x) * w; hq = q8_1(h).  Grid T, block 1024 (two elements a thread).
__global__ void __launch_bounds__(1024) k_add_rms_q(float* x, const float* y, const float* w, float* h, q36k::QB* hq, float eps) {
    constexpr int n = 2048;
    __shared__ float ws[16];
    const int t = threadIdx.x;
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    const float v0 = xr[t] + yr[t], v1 = xr[t + 1024] + yr[t + 1024];
    xr[t] = v0;
    xr[t + 1024] = v1;
    const float r = rsqrtf(block_sum1024(v0 * v0 + v1 * v1, ws) / (float) n + eps);
    const float h0 = __fmul_rn(__fmul_rn(v0, r), w[t]), h1 = __fmul_rn(__fmul_rn(v1, r), w[t + 1024]);
    float* hr = h + (size_t) blockIdx.x * n;
    hr[t] = h0;
    hr[t + 1024] = h1;
    q8_store2(hq + (size_t) blockIdx.x * (n >> 5), t, h0, h1, t & 31);
}

/// q36k::k_moe_out_small for n == 2048: x += sum_k w_k part_k + sigmoid(gate) * shared; with wn, h = rms(x) * wn and,
/// with hq, its q8_1 image.  Grid T, block 1024 (two elements a thread).
__global__ void __launch_bounds__(1024) k_moe_out_small(float* x, const float* parts, const int32_t* prow, const float* wts,
                                                        int K_, const float* shared, const float* sgate, int sg_stride,
                                                        const float* wn, float* h, float eps, q36k::QB* hq) {
    constexpr int n = 2048;
    __shared__ float ws[16];
    __shared__ const float* pk[16];
    const int tk = blockIdx.x, t = threadIdx.x;
    if (t < K_) pk[t] = parts + (size_t) (prow ? prow[tk * K_ + t] : tk * K_ + t) * n;
    __syncthreads();
    const float* wr = wts + (size_t) tk * K_;
    const float gs = q36k::sigm(sgate[(size_t) tk * sg_stride]);
    float* xr = x + (size_t) tk * n;
    float a0 = wr[0] * pk[0][t], a1 = wr[0] * pk[0][t + 1024];
    for (int k = 1; k < K_; ++k) {
        a0 = fmaf(wr[k], pk[k][t], a0);
        a1 = fmaf(wr[k], pk[k][t + 1024], a1);
    }
    a0 += gs * shared[(size_t) tk * n + t];
    a1 += gs * shared[(size_t) tk * n + t + 1024];
    const float v0 = xr[t] + a0, v1 = xr[t + 1024] + a1;
    xr[t] = v0;
    xr[t + 1024] = v1;
    if (wn == nullptr) return;
    const float r = rsqrtf(block_sum1024(v0 * v0 + v1 * v1, ws) / (float) n + eps);
    const float h0 = __fmul_rn(__fmul_rn(v0, r), wn[t]), h1 = __fmul_rn(__fmul_rn(v1, r), wn[t + 1024]);
    h[(size_t) tk * n + t] = h0;
    h[(size_t) tk * n + t + 1024] = h1;
    if (hq) q8_store2(hq + (size_t) tk * (n >> 5), t, h0, h1, t & 31);
}


}  // namespace q36g
#endif
