#include "icp.h"

#include <cmath>

#include "metrics.h"

IcpResult icp_seq(const PointCloud& src, const PointCloud& tgt, const IcpParams& p, const Rigid& init) {
    IcpResult res;
    res.T = init;
    Timer total;
    Timer t;

    // Griglia sul target, con lato della cella = dmax
    Grid grid;
    t.tic();
    grid_build(tgt, p.dmax, grid);
    res.tm.grid = t.toc();

    const size_t n = src.size();
    PointCloud moved;               // sorgente trasformata
    moved.resize(n);
    std::vector<int> match(n);      // indice (nella griglia) del vicino, -1 se nessuno

    for (int it = 0; it < p.iters; ++it) {
        // --- apply: si parte sempre dalla sorgente ORIGINALE (niente deriva numerica)
        t.tic();
        for (size_t i = 0; i < n; ++i) {
            double o[3];
            rigid_apply(res.T, src.x[i], src.y[i], src.z[i], o);
            moved.x[i] = static_cast<float>(o[0]);
            moved.y[i] = static_cast<float>(o[1]);
            moved.z[i] = static_cast<float>(o[2]);
        }
        res.tm.apply += t.toc();

        // --- nn: ricerca dei vicini
        t.tic();
        double sum_d2 = 0;
        size_t valid = 0;
        for (size_t i = 0; i < n; ++i) {
            float d2 = 0;
            match[i] = grid_nearest(grid, moved.x[i], moved.y[i], moved.z[i], p.dmax, &d2);
            if (match[i] >= 0) { sum_d2 += d2; ++valid; }
        }
        res.tm.nn += t.toc();

        // --- reduce: somme sulle coppie valide
        t.tic();
        CorrSums s;
        for (size_t i = 0; i < n; ++i) {
            const int j = match[i];
            if (j < 0) continue;
            s.add(moved.x[i], moved.y[i], moved.z[i], grid.x[j], grid.y[j], grid.z[j]);
        }
        res.tm.reduce += t.toc();

        // --- solve: passo (R,t) e composizione con la T corrente
        t.tic();
        res.rmse.push_back(valid ? std::sqrt(sum_d2 / static_cast<double>(valid)) : 0.0);
        res.matches.push_back(valid);
        Rigid delta;
        if (!kabsch_from_sums(s, delta)) {  // troppe poche coppie: ci si ferma
            res.tm.solve += t.toc();
            break;
        }
        res.T = rigid_compose(delta, res.T);
        res.tm.solve += t.toc();
    }

    res.tm.total = total.toc();
    return res;
}
