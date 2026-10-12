#include "metrics.h"

#include <algorithm>
#include <cmath>

double rotation_error_deg(const double Ra[9], const double Rb[9]) {
    // Rrel = Ra^T Rb; angolo = atan2(|parte antisimmetrica|, (trace - 1) / 2).
    // Piu' accurato di acos() per angoli molto piccoli.
    double Rr[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Rr[3 * i + j] = Ra[i] * Rb[j] + Ra[3 + i] * Rb[3 + j] + Ra[6 + i] * Rb[6 + j];
    const double tr = Rr[0] + Rr[4] + Rr[8];
    const double vx = 0.5 * (Rr[7] - Rr[5]);
    const double vy = 0.5 * (Rr[2] - Rr[6]);
    const double vz = 0.5 * (Rr[3] - Rr[1]);
    const double s = std::sqrt(vx * vx + vy * vy + vz * vz);
    return std::atan2(s, 0.5 * (tr - 1.0)) * 180.0 / 3.14159265358979323846;
}

double translation_error(const double ta[3], const double tb[3]) {
    const double dx = ta[0] - tb[0], dy = ta[1] - tb[1], dz = ta[2] - tb[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void pose_error(const Rigid& est, const Rigid& gt, double& rot_deg, double& trans) {
    rot_deg = rotation_error_deg(est.R, gt.R);
    trans = translation_error(est.t, gt.t);
}
