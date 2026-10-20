#include "voxel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

PointCloud voxel_downsample(const PointCloud& in, float voxel) {
    const size_t n = in.size();
    if (n == 0 || voxel <= 0.f) return in;
    if (n >= (1ull << 32)) throw std::runtime_error("voxel_downsample: troppi punti");

    float minx = in.x[0], miny = in.y[0], minz = in.z[0];
    for (size_t i = 1; i < n; ++i) {
        minx = std::min(minx, in.x[i]);
        miny = std::min(miny, in.y[i]);
        minz = std::min(minz, in.z[i]);
    }

    const double inv = 1.0 / static_cast<double>(voxel);
    const uint64_t LIM = 1ull << 21;

    struct Item { uint64_t key; uint32_t idx; };
    std::vector<Item> items(n);
    for (size_t i = 0; i < n; ++i) {
        uint64_t ix = static_cast<uint64_t>(std::floor((in.x[i] - minx) * inv));
        uint64_t iy = static_cast<uint64_t>(std::floor((in.y[i] - miny) * inv));
        uint64_t iz = static_cast<uint64_t>(std::floor((in.z[i] - minz) * inv));
        if (ix >= LIM || iy >= LIM || iz >= LIM)
            throw std::runtime_error("voxel_downsample: voxel troppo piccolo rispetto all'estensione");
        items[i] = {(ix << 42) | (iy << 21) | iz, static_cast<uint32_t>(i)};
    }

    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.key != b.key ? a.key < b.key : a.idx < b.idx;
    });

    PointCloud out;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        double sx = 0, sy = 0, sz = 0;
        while (j < n && items[j].key == items[i].key) {
            sx += in.x[items[j].idx];
            sy += in.y[items[j].idx];
            sz += in.z[items[j].idx];
            ++j;
        }
        const double c = static_cast<double>(j - i);
        out.push_back(static_cast<float>(sx / c), static_cast<float>(sy / c), static_cast<float>(sz / c));
        i = j;
    }
    return out;
}
