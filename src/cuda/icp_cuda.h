#pragma once
// Versione CUDA: stessa interfaccia di icp_seq / icp_omp, cosi' i risultati sono confrontabili 1:1.
//
// Schema (una GPU, un processo):
//   - la griglia del target e' costruita sull'host con grid_build() (costo ~1 ms) e caricata
//     sulla GPU una sola volta insieme alla sorgente
//   - per ogni iterazione, tre kernel + un passo sull'host:
//       apply  : moved = T * src                      (1 thread per punto)
//       nn     : vicino piu' prossimo nella griglia   (1 thread per punto)
//       reduce : le 16 somme di CorrSums + sum d2     (riduzione a blocchi: shuffle di warp
//                + memoria condivisa, somma finale dei parziali sull'host in ordine fisso
//                -> risultato deterministico)
//       solve  : Horn 4x4 sull'host (sono solo 16 numeri)
//
// Convenzioni di tempo (IcpTimings):
//   grid   = costruzione griglia sull'host + upload di griglia e sorgente (H2D)
//   apply, nn  = tempo reale del kernel (con cudaDeviceSynchronize)
//   reduce = kernel + copia D2H dei parziali + somma sull'host
//   total  = tutto quanto sopra; NON include la creazione del contesto CUDA (CudaInfo::t_init),
//            che avviene una volta per processo.
#include <string>

#include "grid.h"
#include "icp.h"
#include "kabsch.h"

struct CudaOptions {
    int device = 0;
    int block_nn = 128;     // thread per blocco nel kernel nn (multiplo di 32, <= 1024)
    int block_other = 256;  // thread per blocco in apply e reduce (multiplo di 32, <= 1024)
};

struct CudaInfo {
    std::string device_name;
    int sm_count = 0;
    double t_init = 0;  // creazione del contesto CUDA (s)
    double t_h2d = 0;   // upload sorgente + griglia (s)
    double t_d2h = 0;   // download dei parziali della riduzione, somma su tutte le iterazioni (s)
};

// Lancia std::runtime_error se un'operazione CUDA fallisce o i parametri sono invalidi.
IcpResult icp_cuda(const PointCloud& src, const PointCloud& tgt, const IcpParams& p,
                   const Rigid& init = rigid_identity(), const CudaOptions& opt = CudaOptions(),
                   CudaInfo* info = nullptr);
