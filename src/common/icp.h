#pragma once

#ifndef ICP_H
#define ICP_H
// Pezzo 6: ICP point-to-point sequenziale con numero di iterazioni FISSO.
//
// Trova T tale che T(source) sia allineata a target.
// Ogni iterazione ha 4 fasi (cronometrate separatamente, servono per Amdahl):
//   apply  : trasforma la sorgente con la T corrente
//   nn     : ricerca del vicino piu' prossimo nella griglia del target
//   reduce : somme sulle coppie valide (le 16 quantita' di CorrSums)
//   solve  : stima del passo (R,t) e aggiornamento di T
#include <vector>
#include "grid.h"
#include "kabsch.h"
#include "PointCloud.h"

struct IcpParams {
    float dmax = 0.5f;  // distanza massima di corrispondenza (e lato della cella)
    int iters = 30;     // numero di iterazioni (fisso)
};

struct IcpTimings {
    double grid = 0;    // costruzione griglia del target (una volta)
    double apply = 0, nn = 0, reduce = 0, solve = 0;  // somma su tutte le iterazioni
    double total = 0;
};

struct IcpResult {
    Rigid T;                        // trasformazione stimata sorgente -> target
    IcpTimings tm;
    std::vector<double> rmse;       // RMSE delle coppie valide a ogni iterazione
    std::vector<size_t> matches;    // numero di coppie valide a ogni iterazione
};

IcpResult icp_seq(const PointCloud& src, const PointCloud& tgt, const IcpParams& p,
                  const Rigid& init = rigid_identity());

#endif