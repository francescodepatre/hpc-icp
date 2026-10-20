#pragma once 

#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <vector>
#include <cstddef>

struct PointCloud {
    std::vector<float> x, y, z;

    std::size_t size() const { 
        return x.size(); 
    }

    bool empty() const {
        return x.empty();
    }

    void clear() {
        x.clear();
        y.clear();
        z.clear();
    }

    void reserve(size_t n) {
        x.reserve(n);
        y.reserve(n);
        z.reserve(n);
    }

    void resize(size_t n) {
        x.resize(n);
        y.resize(n);
        z.resize(n);
    }

    void push_back(float a, float b, float c){
        x.push_back(a);
        y.push_back(b);
        z.push_back(c);
    }
};

#endif








