#pragma once
// Timer e metriche di correttezza.

#ifndef METRICS_H
#define METRICS_H

#include <chrono>
#include "kabsch.h"

class Timer {
public:
    Timer() { tic(); }
    void tic() { t0_ = std::chrono::steady_clock::now(); }
    
    double toc() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

double rotation_error_deg(const double Ra[9], const double Rb[9]);

double translation_error(const double ta[3], const double tb[3]);

void pose_error(const Rigid& est, const Rigid& gt, double& rot_deg, double& trans);

#endif