#pragma once

#ifndef KABSCH_H
#define KABSCH_H

#include <cmath>

struct Rigid {
    double R[9];
    double t[3];
};

inline Rigid rigid_identity() {
    Rigid T = {{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}};
    return T;
}

inline Rigid rigid_compose(const Rigid& A, const Rigid& B) {
    Rigid C;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            C.R[3 * i + j] = A.R[3 * i] * B.R[j] + A.R[3 * i + 1] * B.R[3 + j] + A.R[3 * i + 2] * B.R[6 + j];
        C.t[i] = A.R[3 * i] * B.t[0] + A.R[3 * i + 1] * B.t[1] + A.R[3 * i + 2] * B.t[2] + A.t[i];
    }
    return C;
}

inline Rigid rigid_inverse(const Rigid& T) {
    Rigid I;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) I.R[3 * i + j] = T.R[3 * j + i];  // trasposta
    for (int i = 0; i < 3; ++i)
        I.t[i] = -(I.R[3 * i] * T.t[0] + I.R[3 * i + 1] * T.t[1] + I.R[3 * i + 2] * T.t[2]);
    return I;
}

inline void rigid_apply(const Rigid& T, double x, double y, double z, double out[3]) {
    out[0] = T.R[0] * x + T.R[1] * y + T.R[2] * z + T.t[0];
    out[1] = T.R[3] * x + T.R[4] * y + T.R[5] * z + T.t[1];
    out[2] = T.R[6] * x + T.R[7] * y + T.R[8] * z + T.t[2];
}

struct CorrSums {
    double n = 0;
    double sp[3] = {0, 0, 0};
    double sq[3] = {0, 0, 0};
    double spq[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  

    void add(double px, double py, double pz, double qx, double qy, double qz) {
        n += 1;
        sp[0] += px; sp[1] += py; sp[2] += pz;
        sq[0] += qx; sq[1] += qy; sq[2] += qz;
        spq[0] += px * qx; spq[1] += px * qy; spq[2] += px * qz;
        spq[3] += py * qx; spq[4] += py * qy; spq[5] += py * qz;
        spq[6] += pz * qx; spq[7] += pz * qy; spq[8] += pz * qz;
    }
};

bool kabsch_from_sums(const CorrSums& s, Rigid& out);

#endif