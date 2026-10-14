#!/usr/bin/env python3
"""
Speedup, parallel efficiency e frazione seriale (Karp-Flatt) dai risultati di bench.py.

  python3 scripts/plot_results.py results/bench.csv [--outdir results] [--metric t_total]

 - speedup di ogni coppia = mediana(t_total seq) / mediana(t_total della configurazione):
   la baseline e' SEMPRE la versione sequenziale
 - 'worker' = thread x processi MPI (omp: T, mpi: P, hybrid: P x T)
 - le serie hybrid sono separate per numero di thread per rank (es. "hybrid T=2")
 - se il CSV contiene dati MPI, stampa anche frazione di comunicazione e sbilanciamento

Richiede numpy e matplotlib.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def serie(variant, threads):
    return f"{variant} T={threads}" if variant.startswith("hybrid") else variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--metric", default="t_total", help="t_total, t_nn, ...")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        raise SystemExit("CSV vuoto")

    def procs(r):  # vecchi CSV senza la colonna mpi_processes
        v = r.get("mpi_processes", "")
        return int(v) if v not in ("", None) else 1

    # mediana sulle ripetizioni per (serie, worker, seq, i, j)
    cell = defaultdict(list)
    extra = defaultdict(lambda: defaultdict(list))   # (serie, worker) -> {comm_frac, imb}
    for r in rows:
        T, P = int(r["threads"]), procs(r)
        s = serie(r["variant"], T)
        W = T * P
        cell[(s, W, r["seq"], int(r["i"]), int(r["j"]))].append(float(r[a.metric]))
        if r.get("t_comm", "") not in ("", None) and float(r["t_total"]) > 0:
            extra[(s, W)]["comm"].append(float(r["t_comm"]) / float(r["t_total"]))
            extra[(s, W)]["imb"].append(float(r["imbalance"]))
    med = {k: float(np.median(v)) for k, v in cell.items()}

    sp = defaultdict(list)   # (serie, worker, seq) -> speedup per coppia
    for (s, W, seq, i, j), t in med.items():
        if s == "seq":
            continue
        base = med.get(("seq", 1, seq, i, j))
        if base is None:
            continue
        sp[(s, W, seq)].append(base / t)

    variants = sorted({k[0] for k in sp})
    seqs = sorted({k[2] for k in sp})
    if not variants:
        raise SystemExit("nessun dato di una variante parallela con baseline seq nello stesso CSV")

    print(f"metrica: {a.metric}  (speedup rispetto alla versione sequenziale)\n")
    print(f"{'serie':<24}{'seq':<6}{'worker':>7}{'speedup':>10}{'efficienza':>12}{'Karp-Flatt e':>14}")
    table = {}
    for var in variants:
        for s in seqs:
            for W in sorted({k[1] for k in sp if k[0] == var and k[2] == s}):
                v = np.array(sp[(var, W, s)])
                S = float(np.median(v))
                E = S / W
                e = ((1 / S - 1 / W) / (1 - 1 / W)) if W > 1 else float("nan")
                table[(var, s, W)] = (S, E, float(v.min()), float(v.max()))
                print(f"{var:<24}{s:<6}{W:>7}{S:>10.2f}{E:>12.2f}{e:>14.4f}")

    if extra:
        print("\nMPI: comunicazione e sbilanciamento (mediana su tutte le misure)")
        print(f"{'serie':<24}{'worker':>7}{'comm/totale':>13}{'max/medio':>11}")
        for (s, W) in sorted(extra):
            print(f"{s:<24}{W:>7}{np.median(extra[(s, W)]['comm']):>13.3f}{np.median(extra[(s, W)]['imb']):>11.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib non installato: salto i grafici (uv add matplotlib / pip install matplotlib)")
        return

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def disegna(ax, Ws_all, ylabel_speed):
        ax[0].plot(Ws_all, Ws_all, "k--", label="ideale")
        ax[0].set_ylabel(ylabel_speed)
        ax[1].axhline(1.0, color="k", ls="--")
        ax[1].set_ylabel("parallel efficiency")
        ax[1].set_ylim(0, 1.15)
        for x in ax:
            x.set_xlabel("worker (thread x processi MPI)")
            x.set_xscale("log", base=2)
            x.set_xticks(Ws_all)
            x.set_xticklabels(Ws_all)
            x.grid(alpha=0.3)

    # una figura per serie (linee = sequenze, banda = min-max tra le coppie)
    for var in variants:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        Ws_all = sorted({k[2] for k in table if k[0] == var})
        for s in seqs:
            Ws = sorted(k[2] for k in table if k[0] == var and k[1] == s)
            if not Ws:
                continue
            ax[0].plot(Ws, [table[(var, s, W)][0] for W in Ws], "o-", label=f"seq. {s}")
            ax[0].fill_between(Ws, [table[(var, s, W)][2] for W in Ws], [table[(var, s, W)][3] for W in Ws], alpha=0.15)
            ax[1].plot(Ws, [table[(var, s, W)][1] for W in Ws], "o-", label=f"seq. {s}")
        disegna(ax, Ws_all, "speedup vs sequenziale")
        ax[0].set_title(f"Speedup ({var}, {a.metric})")
        ax[1].set_title("Efficienza")
        ax[0].legend()
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9]+", "_", var).strip("_")
        path = outdir / f"speedup_{safe}_{a.metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"grafico: {path}")

    # confronto tra serie: mediana su tutte le coppie di tutte le sequenze
    if len(variants) > 1:
        allsp = defaultdict(list)
        for (var, W, seq), v in sp.items():
            allsp[(var, W)].extend(v)
        Ws_all = sorted({k[1] for k in allsp})
        print(f"\nconfronto tra serie (mediana su tutte le coppie, {a.metric}):")
        print(f"{'serie':<24}" + "".join(f"{('W=' + str(W)):>9}" for W in Ws_all))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for var in variants:
            Ws = sorted(W for (v_, W) in allsp if v_ == var)
            S = [float(np.median(allsp[(var, W)])) for W in Ws]
            print(f"{var:<24}" + "".join(
                f"{('%.2f' % float(np.median(allsp[(var, W)]))) if (var, W) in allsp else '-':>9}" for W in Ws_all))
            ax[0].plot(Ws, S, "o-", label=var)
            ax[1].plot(Ws, [x / W for x, W in zip(S, Ws)], "o-", label=var)
        disegna(ax, Ws_all, "speedup vs sequenziale")
        ax[0].legend(fontsize=8)
        ax[0].set_title(f"Confronto ({a.metric})")
        ax[1].set_title("Efficienza")
        fig.tight_layout()
        path = outdir / f"confronto_varianti_{a.metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"grafico: {path}")


if __name__ == "__main__":
    main()
