#!/usr/bin/env python3
"""
Speedup, parallel efficiency e frazione seriale (Karp-Flatt) dai risultati di bench.py.

  python3 scripts/plot_results.py results/bench.csv [--outdir results]

Speedup di ogni coppia = mediana(t_total seq) / mediana(t_total variante,thread)
(la baseline e' SEMPRE la versione sequenziale). Poi si riporta la mediana tra le coppie
di ciascuna sequenza. Richiede numpy e matplotlib.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--metric", default="t_total", help="t_total oppure t_nn, ecc.")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        raise SystemExit("CSV vuoto")

    # mediana sulle ripetizioni per (variante, thread, seq, i, j)
    cell = defaultdict(list)
    for r in rows:
        cell[(r["variant"], int(r["threads"]), r["seq"], int(r["i"]), int(r["j"]))].append(float(r[a.metric]))
    med = {k: float(np.median(v)) for k, v in cell.items()}

    # speedup per coppia rispetto a seq
    sp = defaultdict(list)   # (variante, thread, seq) -> lista di speedup
    for (var, T, seq, i, j), t in med.items():
        if var == "seq":
            continue
        base = med.get(("seq", 1, seq, i, j))
        if base is None:
            continue
        sp[(var, T, seq)].append(base / t)

    variants = sorted({k[0] for k in sp})
    seqs = sorted({k[2] for k in sp})
    if not variants:
        raise SystemExit("nessun dato di una variante parallela con baseline seq nello stesso CSV")

    print(f"metrica: {a.metric}  (speedup rispetto alla versione sequenziale)\n")
    print(f"{'variante':<20}{'seq':<6}{'thread':>7}{'speedup':>10}{'efficienza':>12}{'Karp-Flatt e':>14}")
    table = {}
    for var in variants:
        for s in seqs:
            for T in sorted({k[1] for k in sp if k[0] == var and k[2] == s}):
                v = np.array(sp[(var, T, s)])
                S = float(np.median(v))
                E = S / T
                e = ((1 / S - 1 / T) / (1 - 1 / T)) if T > 1 else float("nan")
                table[(var, s, T)] = (S, E, float(v.min()), float(v.max()))
                print(f"{var:<20}{s:<6}{T:>7}{S:>10.2f}{E:>12.2f}{e:>14.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib non installato: salto i grafici (uv add matplotlib / pip install matplotlib)")
        return

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for var in variants:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        allT = sorted({k[2] for k in table if k[0] == var})
        for s in seqs:
            Ts = sorted(k[2] for k in table if k[0] == var and k[1] == s)
            S = [table[(var, s, T)][0] for T in Ts]
            lo = [table[(var, s, T)][2] for T in Ts]
            hi = [table[(var, s, T)][3] for T in Ts]
            E = [table[(var, s, T)][1] for T in Ts]
            ax[0].plot(Ts, S, "o-", label=f"seq. {s}")
            ax[0].fill_between(Ts, lo, hi, alpha=0.15)
            ax[1].plot(Ts, E, "o-", label=f"seq. {s}")
        ax[0].plot(allT, allT, "k--", label="ideale")
        ax[0].set_xlabel("thread"); ax[0].set_ylabel("speedup vs sequenziale")
        ax[0].set_title(f"Speedup ({var}, {a.metric})"); ax[0].grid(alpha=0.3); ax[0].legend()
        ax[1].axhline(1.0, color="k", ls="--")
        ax[1].set_xlabel("thread"); ax[1].set_ylabel("parallel efficiency")
        ax[1].set_title("Efficienza"); ax[1].set_ylim(0, 1.15); ax[1].grid(alpha=0.3)
        for x in ax:
            x.set_xscale("log", base=2); x.set_xticks(allT); x.set_xticklabels(allT)
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9]+", "_", var).strip("_")
        path = outdir / f"speedup_{safe}_{a.metric}.png"
        fig.savefig(path, dpi=150)
        print(f"grafico: {path}")


    # Confronto tra varianti (es. schedule diverse): una linea per variante, mediana su tutte le coppie
    if len(variants) > 1:
        allsp = defaultdict(list)
        for (var, T, seq), v in sp.items():
            allsp[(var, T)].extend(v)
        Ts_all = sorted({k[1] for k in allsp})
        print(f"\nconfronto tra varianti (mediana su tutte le coppie, {a.metric}):")
        print(f"{'variante':<22}" + "".join(f"{('T=' + str(T)):>9}" for T in Ts_all))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for var in variants:
            Ts = sorted(T for (v_, T) in allsp if v_ == var)
            S = [float(np.median(allsp[(var, T)])) for T in Ts]
            print(f"{var:<22}" + "".join(f"{(('%.2f' % float(np.median(allsp[(var, T)]))) if (var, T) in allsp else '-'):>9}" for T in Ts_all))
            ax[0].plot(Ts, S, "o-", label=var)
            ax[1].plot(Ts, [x / t for x, t in zip(S, Ts)], "o-", label=var)
        ax[0].plot(Ts_all, Ts_all, "k--", label="ideale")
        ax[0].set_ylabel("speedup vs sequenziale"); ax[1].set_ylabel("parallel efficiency")
        ax[1].axhline(1.0, color="k", ls="--"); ax[1].set_ylim(0, 1.15)
        for x in ax:
            x.set_xlabel("thread"); x.set_xscale("log", base=2); x.set_xticks(Ts_all)
            x.set_xticklabels(Ts_all); x.grid(alpha=0.3)
        ax[0].legend(fontsize=8)
        ax[0].set_title(f"Confronto varianti ({a.metric})"); ax[1].set_title("Efficienza")
        fig.tight_layout()
        path = outdir / f"confronto_varianti_{a.metric}.png"
        fig.savefig(path, dpi=150)
        print(f"grafico: {path}")


if __name__ == "__main__":
    main()
