#pragma once
// ICP con MPI (data-parallel) e, se compilato con OpenMP, ibrido MPI+OpenMP.
//
// Schema: sorgente e target sono replicati su tutti i rank (vedi bcast_cloud). Ogni rank
// ricerca i vicini solo per il PROPRIO sottoinsieme di punti sorgente; le 16 somme di
// CorrSums (piu' somma delle distanze e numero di coppie) vengono sommate con UNA sola
// collettiva per iterazione. La griglia del target e' costruita da ogni rank (replicata).
#include "grid.h"
#include "icp.h"
#include "kabsch.h"

// Come si distribuiscono i punti sorgente tra i rank
enum class Partition {
    Block,   // blocchi contigui: buona localita', ma sensibile a densita' non uniforme
    Cyclic   // punti i = rank, rank+P, rank+2P...: carico piu' uniforme, localita' peggiore
};

// Come si combinano le somme a ogni iterazione
enum class Collective {
    ReduceBcast,  // MPI_Reduce su rank 0, SVD su rank 0, MPI_Bcast di T: T identico su tutti i rank
    Allreduce     // MPI_Allreduce e SVD su tutti i rank (una collettiva sola)
};

struct MpiOptions {
    Partition partition = Partition::Block;
    Collective collective = Collective::ReduceBcast;
};

// Statistiche locali al rank (utili per bilanciamento e comunicazione)
struct MpiStats {
    double compute = 0;          // apply + nn + somme locali (senza comunicazione)
    double comm = 0;             // tempo nelle collettive (include l'attesa del rank piu' lento)
    size_t local_points = 0;     // punti sorgente assegnati a questo rank
};

// Tutti i rank devono chiamarla con gli stessi src, tgt, p, init.
// IcpResult::tm: grid, apply, nn, reduce (= solo somme LOCALI), solve (solo rank 0 nella
// modalita' ReduceBcast), total. La comunicazione e' in MpiStats::comm.
IcpResult icp_mpi(const PointCloud& src, const PointCloud& tgt, const IcpParams& p,
                  const Rigid& init = rigid_identity(), const MpiOptions& opt = MpiOptions(),
                  MpiStats* stats = nullptr);

// Distribuisce una nuvola dal rank `root` a tutti (MPI_Bcast). Chiamata da tutti i rank.
void bcast_cloud(PointCloud& pc, int root = 0);
