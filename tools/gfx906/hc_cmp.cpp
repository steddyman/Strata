// fused_gr_read_multi outputs for T = 1..4 (with and without the pending write and the inject rows) dumped to a file
#include "strata/kernels/fused_gr.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
using namespace strata::kernels;
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
static uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
template <class T> T* up(const std::vector<T>& h) { T* d; hipMalloc(&d, h.size() * sizeof(T)); hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice); return d; }
int main(int argc, char** argv) {
    FILE* out = std::fopen(argv[1], "wb");
    std::mt19937 rng(7); std::normal_distribution<float> nd(0, 1);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& v : wd) v = bf(nd(rng) * 0.02f); for (auto& v : wu) v = bf(nd(rng) * 0.05f); for (auto& v : wi) v = bf(nd(rng) * 0.02f);
    std::vector<float> wn(D); for (auto& v : wn) v = 1.0f + 0.1f * nd(rng);
    uint16_t *dwd = up(wd), *dwu = up(wu), *dwi = up(wi); float* dwn = up(wn);
    hipStream_t s; hipStreamCreate(&s);
    for (int apply = 0; apply < 2; ++apply) for (int inj = 0; inj < 2; ++inj) for (int T = 1; T <= 4; ++T) {
        std::vector<FusedGrArgs> a(T); std::vector<float*> R(T), lo(T), rs(T), io(T), mx(T);
        for (int k = 0; k < T; ++k) {
            std::vector<float> r(D), bo(N), ip(HC); for (auto& v : r) v = nd(rng); for (auto& v : bo) v = nd(rng); for (auto& v : ip) v = nd(rng);
            R[k] = up(r); hipMalloc(&lo[k], LR * 4); hipMalloc(&rs[k], HC * 4); hipMalloc(&io[k], HC * 4); hipMalloc(&mx[k], N * 4);
            a[k].R = R[k]; a[k].R_out = R[k]; a[k].apply = apply; a[k].bo_prev = up(bo); a[k].inj_prev = up(ip);
            a[k].w_norm = dwn; a[k].w_down = dwd; a[k].w_up = dwu; a[k].w_inject = inj ? dwi : nullptr;
            a[k].lo = lo[k]; a[k].rs = rs[k]; a[k].inject_out = io[k]; a[k].mixed = mx[k];
        }
        float* xn; hipMalloc(&xn, (size_t) T * D * 4);
        fused_gr_read_multi(a.data(), T, xn, s); hipStreamSynchronize(s);
        for (int k = 0; k < T; ++k) {
            std::vector<float> h; auto dump = [&](float* p, int n) { h.resize(n); hipMemcpy(h.data(), p, n * 4, hipMemcpyDeviceToHost); std::fwrite(h.data(), 4, n, out); };
            dump(lo[k], LR); dump(rs[k], HC); dump(io[k], inj ? HC : 0); dump(mx[k], N); dump(R[k], D);
        }
    }
    std::fclose(out); std::printf("dumped\n");
}
