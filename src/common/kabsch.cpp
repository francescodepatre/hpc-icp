#include "kabsch.h"

#include <algorithm>

namespace {

// Autovalori/autovettori di una matrice simmetrica 4x4 con il metodo di Jacobi ciclico.
// Su ritorno A e' (quasi) diagonale, le colonne di V sono gli autovettori.
void jacobi4(double A[4][4], double V[4][4]) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0, diag = 0;
        for (int i = 0; i < 4; ++i) {
            diag += A[i][i] * A[i][i];
            for (int j = i + 1; j < 4; ++j) off += A[i][j] * A[i][j];
        }
        if (off <= 1e-30 * (diag + 1e-300)) break;

        for (int p = 0; p < 3; ++p)
            for (int q = p + 1; q < 4; ++q) {
                if (std::fabs(A[p][q]) < 1e-300) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < 4; ++k) {  // A <- A J
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 4; ++k) {  // A <- J^T A
                    const double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 4; ++k) {  // V <- V J
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
    }
}

}  // namespace

bool kabsch_from_sums(const CorrSums& s, Rigid& out) {
    if (s.n < 3) return false;

    double cp[3], cq[3];
    for (int a = 0; a < 3; ++a) {
        cp[a] = s.sp[a] / s.n;
        cq[a] = s.sq[a] / s.n;
    }

    // Matrice di covarianza incrociata centrata: S_ab = sum (p_a - cp_a)(q_b - cq_b)
    double S[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) S[a][b] = s.spq[3 * a + b] - s.n * cp[a] * cq[b];

    const double Sxx = S[0][0], Sxy = S[0][1], Sxz = S[0][2];
    const double Syx = S[1][0], Syy = S[1][1], Syz = S[1][2];
    const double Szx = S[2][0], Szy = S[2][1], Szz = S[2][2];

    // Matrice N di Horn: il quaternione ottimo e' l'autovettore del suo autovalore massimo
    double N[4][4] = {
        {Sxx + Syy + Szz, Syz - Szy,        Szx - Sxz,         Sxy - Syx},
        {Syz - Szy,       Sxx - Syy - Szz,  Sxy + Syx,         Szx + Sxz},
        {Szx - Sxz,       Sxy + Syx,       -Sxx + Syy - Szz,   Syz + Szy},
        {Sxy - Syx,       Szx + Sxz,        Syz + Szy,        -Sxx - Syy + Szz}};

    double V[4][4];
    jacobi4(N, V);

    int k = 0;
    for (int i = 1; i < 4; ++i)
        if (N[i][i] > N[k][k]) k = i;

    double w = V[0][k], x = V[1][k], y = V[2][k], z = V[3][k];
    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    w /= norm; x /= norm; y /= norm; z /= norm;

    out.R[0] = 1 - 2 * (y * y + z * z);
    out.R[1] = 2 * (x * y - z * w);
    out.R[2] = 2 * (x * z + y * w);
    out.R[3] = 2 * (x * y + z * w);
    out.R[4] = 1 - 2 * (x * x + z * z);
    out.R[5] = 2 * (y * z - x * w);
    out.R[6] = 2 * (x * z - y * w);
    out.R[7] = 2 * (y * z + x * w);
    out.R[8] = 1 - 2 * (x * x + y * y);

    for (int i = 0; i < 3; ++i)
        out.t[i] = cq[i] - (out.R[3 * i] * cp[0] + out.R[3 * i + 1] * cp[1] + out.R[3 * i + 2] * cp[2]);
    return true;
}
