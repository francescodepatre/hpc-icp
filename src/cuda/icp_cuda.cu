#include "icp_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "metrics.h"

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess)                                                                \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " (" + \
                                     #call + ", line " + std::to_string(__LINE__) + ")");     \
    } while (0)

namespace {

template <typename T>
struct DevBuf {
    T* p = nullptr;
    DevBuf() = default;
    explicit DevBuf(size_t n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (p) cudaFree(p); }
};

struct GridParams {
    float minx, miny, minz, inv;
    int nx, ny, nz;
};

constexpr int NV = 17;


__global__ void apply_kernel(int n, const float4* __restrict__ src, float4* __restrict__ moved, Rigid T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 p = src[i];
    const double x = p.x, y = p.y, z = p.z;
    float4 o;
    o.x = static_cast<float>(T.R[0] * x + T.R[1] * y + T.R[2] * z + T.t[0]);
    o.y = static_cast<float>(T.R[3] * x + T.R[4] * y + T.R[5] * z + T.t[1]);
    o.z = static_cast<float>(T.R[6] * x + T.R[7] * y + T.R[8] * z + T.t[2]);
    o.w = 0.f;
    moved[i] = o;
}

__global__ void nn_kernel(int n, const float4* __restrict__ moved, const int* __restrict__ cellStart,
                          const float4* __restrict__ gpts, GridParams g, float dmax,
                          int* __restrict__ match, float* __restrict__ d2out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float4 q = moved[i];
    const float fx = floorf((q.x - g.minx) * g.inv);
    const float fy = floorf((q.y - g.miny) * g.inv);
    const float fz = floorf((q.z - g.minz) * g.inv);

    int best_j = -1;
    float best = dmax * dmax;

    if (!(fx < -1.f || fx > static_cast<float>(g.nx) || fy < -1.f || fy > static_cast<float>(g.ny) ||
          fz < -1.f || fz > static_cast<float>(g.nz))) {
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy), iz = static_cast<int>(fz);
        const int x0 = max(ix - 1, 0), x1 = min(ix + 1, g.nx - 1);
        const int y0 = max(iy - 1, 0), y1 = min(iy + 1, g.ny - 1);
        const int z0 = max(iz - 1, 0), z1 = min(iz + 1, g.nz - 1);

        for (int cz = z0; cz <= z1; ++cz)
            for (int cy = y0; cy <= y1; ++cy) {
                const int row = g.nx * (cy + g.ny * cz);
                const int jb = __ldg(&cellStart[row + x0]);
                const int je = __ldg(&cellStart[row + x1 + 1]);
                for (int j = jb; j < je; ++j) {
                    const float4 c = __ldg(&gpts[j]);
                    const float dx = c.x - q.x;
                    const float dy = c.y - q.y;
                    const float dz = c.z - q.z;
                    const float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < best) { best = d2; best_j = j; }
                }
            }
    }
    match[i] = best_j;
    d2out[i] = (best_j >= 0) ? best : 0.f;
}


__global__ void reduce_kernel(int n, const float4* __restrict__ moved, const int* __restrict__ match,
                              const float* __restrict__ d2, const float4* __restrict__ gpts,
                              double* __restrict__ partial) {
    double v[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) v[k] = 0.0;

    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int j = match[i];
        if (j < 0) continue;
        const float4 p4 = moved[i];
        const float4 q4 = __ldg(&gpts[j]);
        const double px = p4.x, py = p4.y, pz = p4.z;
        const double qx = q4.x, qy = q4.y, qz = q4.z;
        v[0] += 1.0;
        v[1] += px; v[2] += py; v[3] += pz;
        v[4] += qx; v[5] += qy; v[6] += qz;
        v[7] += px * qx;  v[8] += px * qy;  v[9] += px * qz;
        v[10] += py * qx; v[11] += py * qy; v[12] += py * qz;
        v[13] += pz * qx; v[14] += pz * qy; v[15] += pz * qz;
        v[16] += static_cast<double>(d2[i]);
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

#pragma unroll
    for (int k = 0; k < NV; ++k)
        for (int off = 16; off > 0; off >>= 1) v[k] += __shfl_down_sync(0xffffffffu, v[k], off);

    __shared__ double sh[NV][32];
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < NV; ++k) sh[k][warp] = v[k];
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < NV; ++k) {
            double x = (lane < nwarps) ? sh[k][lane] : 0.0;
            for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
            if (lane == 0) partial[static_cast<size_t>(blockIdx.x) * NV + k] = x;
        }
    }
}

void check_block(int b, const char* name) {
    if (b < 32 || b > 1024 || b % 32 != 0)
        throw std::runtime_error(std::string(name) + " must be a multiple of 32 between 32 and 1024");
}

std::vector<float4> to_float4(const PointCloud& pc) {
    std::vector<float4> v(pc.size());
    for (size_t i = 0; i < pc.size(); ++i) v[i] = make_float4(pc.x[i], pc.y[i], pc.z[i], 0.f);
    return v;
}

std::vector<float4> to_float4(const Grid& g) {
    std::vector<float4> v(g.x.size());
    for (size_t i = 0; i < g.x.size(); ++i) v[i] = make_float4(g.x[i], g.y[i], g.z[i], 0.f);
    return v;
}

}

IcpResult icp_cuda(const PointCloud& src, const PointCloud& tgt, const IcpParams& p, const Rigid& init,
                   const CudaOptions& opt, CudaInfo* info) {
    if (src.empty()) {
        throw std::runtime_error("icp_cuda: empty source");
    }

    if (tgt.empty()) {
        throw std::runtime_error("icp_cuda: empty target");
    }

    if (src.size() >= static_cast<size_t>(INT_MAX) ||
        tgt.size() >= static_cast<size_t>(INT_MAX)) {
        throw std::runtime_error("icp_cuda: too many points");
    }

    if (p.iters <= 0) {
        throw std::runtime_error("icp_cuda: the number of iterations must be positive");
    }

    if (!(p.dmax > 0.0f) || !std::isfinite(p.dmax)) {
        throw std::runtime_error("icp_cuda: dmax must be positive and finite");
    }

    check_block(opt.block_nn, "block_nn");
    check_block(opt.block_other, "block_other");

    Timer ti;
    CUDA_CHECK(cudaSetDevice(opt.device));
    CUDA_CHECK(cudaFree(nullptr));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
    if (info) {
        info->t_init = ti.toc();
        info->device_name = prop.name;
        info->sm_count = prop.multiProcessorCount;
    }

    IcpResult res;
    res.T = init;
    Timer total;
    Timer t;

    Grid grid;
    t.tic();
    grid_build(tgt, p.dmax, grid);
    const int n = static_cast<int>(src.size());
    const std::vector<float4> h_src = to_float4(src);
    const std::vector<float4> h_gpts = to_float4(grid);
    const size_t ncells1 = grid.cellStart.size();

    DevBuf<float4> d_src(n), d_moved(n), d_gpts(h_gpts.size());
    DevBuf<int> d_cellStart(ncells1), d_match(n);
    DevBuf<float> d_d2(n);

    const int nb_apply = (n + opt.block_other - 1) / opt.block_other;
    const int nb_nn = (n + opt.block_nn - 1) / opt.block_nn;
    const int nb_red = std::max(1, std::min(nb_apply, 4 * std::max(prop.multiProcessorCount, 1)));
    DevBuf<double> d_partial(static_cast<size_t>(nb_red) * NV);
    std::vector<double> h_partial(static_cast<size_t>(nb_red) * NV);

    Timer th;
    CUDA_CHECK(cudaMemcpy(d_src.p, h_src.data(), h_src.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gpts.p, h_gpts.data(), h_gpts.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellStart.p, grid.cellStart.data(), ncells1 * sizeof(int), cudaMemcpyHostToDevice));
    if (info) info->t_h2d = th.toc();
    res.tm.grid = t.toc();

    GridParams gp;
    gp.minx = grid.minx; gp.miny = grid.miny; gp.minz = grid.minz; gp.inv = grid.inv;
    gp.nx = grid.nx; gp.ny = grid.ny; gp.nz = grid.nz;

    for (int it = 0; it < p.iters; ++it) {
        t.tic();
        apply_kernel<<<nb_apply, opt.block_other>>>(n, d_src.p, d_moved.p, res.T);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        res.tm.apply += t.toc();

        t.tic();
        nn_kernel<<<nb_nn, opt.block_nn>>>(n, d_moved.p, d_cellStart.p, d_gpts.p, gp, p.dmax, d_match.p, d_d2.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        res.tm.nn += t.toc();

        t.tic();
        reduce_kernel<<<nb_red, opt.block_other>>>(n, d_moved.p, d_match.p, d_d2.p, d_gpts.p, d_partial.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        Timer td;
        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial.p, h_partial.size() * sizeof(double), cudaMemcpyDeviceToHost));
        if (info) info->t_d2h += td.toc();
        double acc[NV] = {0};
        for (int b = 0; b < nb_red; ++b)
            for (int k = 0; k < NV; ++k) acc[k] += h_partial[static_cast<size_t>(b) * NV + k];
        CorrSums s;
        s.n = acc[0];
        for (int k = 0; k < 3; ++k) { s.sp[k] = acc[1 + k]; s.sq[k] = acc[4 + k]; }
        for (int k = 0; k < 9; ++k) s.spq[k] = acc[7 + k];
        const double sum_d2 = acc[16];
        const size_t valid = static_cast<size_t>(std::llround(acc[0]));
        res.tm.reduce += t.toc();

        t.tic();
        res.rmse.push_back(valid ? std::sqrt(sum_d2 / static_cast<double>(valid)) : 0.0);
        res.matches.push_back(valid);
        Rigid delta;
        if (!kabsch_from_sums(s, delta)) {
            res.tm.solve += t.toc();
            break;
        }
        res.T = rigid_compose(delta, res.T);
        res.tm.solve += t.toc();
    }

    res.tm.total = total.toc();
    return res;
}