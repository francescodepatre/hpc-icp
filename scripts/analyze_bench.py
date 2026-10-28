#!/usr/bin/env python3
"""
Analisi dei CSV prodotti da bench.py per ICP (seq, OpenMP, MPI, hybrid).

Accetta uno o più CSV. Se i risultati sono separati per variante, passali tutti
insieme, includendo il CSV della baseline sequenziale.

Esempi:
    python3 scripts/analyze_bench.py results/bench.csv
    python3 scripts/analyze_bench.py results/bench_seq.csv results/bench_omp.csv results/bench_mpi.csv
    python3 scripts/analyze_bench.py results/bench_*.csv --outdir results/analysis
    python3 scripts/analyze_bench.py results/bench_*.csv --metric t_total --show

Produce:
    summary.csv             speedup, efficienza, Karp-Flatt e tempi per configurazione
    paired_speedups.csv     speedup di ogni coppia/sequenza
    report.md               riepilogo testuale utilizzabile nella relazione
    speedup_*.png           speedup ed efficienza per variante
    confronto_varianti.png  confronto tra tutte le varianti
    tempi_*.png             tempi mediani per componente
    mpi_overhead.png        comunicazione e sbilanciamento, se presenti

Dipendenze: numpy, matplotlib (matplotlib è necessario solo per i grafici).
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


TIME_METRICS = [
    "t_total", "t_grid", "t_apply", "t_nn", "t_reduce", "t_solve"
]
REQUIRED = {"seq", "i", "j", "t_total"}  # variant e threads sono opzionali

# Nomi alternativi delle colonne (nuovo formato -> nome interno usato dallo script).
COLUMN_ALIASES = {
    "sequence": "seq",
    "src_idx": "i",
    "tgt_idx": "j",
    "mpi_ranks": "mpi_processes",
}
DEFAULT_SEQUENCE = "00"  # usata quando la colonna sequence è vuota (es. KITTI 00)


def normalize_row(row):
    """Rinomina le colonne alias, senza sovrascrivere quelle già presenti."""
    out = dict(row)
    for old, new in COLUMN_ALIASES.items():
        if old in out and (new not in out or out[new] in (None, "")):
            out[new] = out[old]
    return out


def parse_int(value, default=1):
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def parse_float(value):
    if value is None or str(value).strip() == "":
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def variant_series(variant, threads):
    """Mantiene separate le configurazioni hybrid con T diversi."""
    variant = (variant or "").strip()
    if variant.startswith("hybrid"):
        return f"{variant} T={threads}"
    return variant


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def read_csvs(paths):
    rows = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"File CSV non trovato: {path}")
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"Avviso: CSV vuoto o senza intestazione, ignorato: {path}", file=sys.stderr)
                continue
            fields = set(reader.fieldnames) | {
                COLUMN_ALIASES[c] for c in reader.fieldnames if c in COLUMN_ALIASES
            }
            # La colonna "variant" può mancare: viene dedotta dal nome del file.
            missing = REQUIRED - fields
            if missing:
                raise SystemExit(
                    f"{path}: mancano colonne richieste: {', '.join(sorted(missing))}"
                )
            count = 0
            for row in reader:
                row = normalize_row(row)
                row["_source_file"] = path.name
                # Alcuni CSV non hanno threads / mpi_processes (es. baseline seq).
                row["threads"] = str(parse_int(row.get("threads"), 1))
                row["mpi_processes"] = str(parse_int(row.get("mpi_processes"), 1))
                row["variant"] = (row.get("variant") or "").strip()
                
                if not row["variant"]:
                    # Deduce la variante dal nome file, rimuovendo anche il suffisso
                    # della sequenza: bench_seq_00.csv -> seq
                    name = path.stem.replace("bench_", "", 1)
                    row["variant"] = re.sub(r"_\d+$", "", name)

                if not str(row.get("seq") or "").strip():
                    row["seq"] = DEFAULT_SEQUENCE
                try:
                    row["seq"] = str(row["seq"]).strip()
                    row["i"] = str(parse_int(row["i"], 0))
                    row["j"] = str(parse_int(row["j"], 0))
                    row["t_total"] = str(float(row["t_total"]))
                except (TypeError, ValueError):
                    print(f"Avviso: riga non valida ignorata in {path}: {row}", file=sys.stderr)
                    continue
                rows.append(row)
                count += 1
            print(f"Letto {path}: {count} righe")
    if not rows:
        raise SystemExit("Nessun dato valido nei CSV forniti.")
    return rows


def median_or_nan(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="uno o più CSV prodotti da bench.py")
    ap.add_argument("--outdir", default="results/analysis", help="cartella dei risultati")
    ap.add_argument("--metric", default="t_total", help="metrica per lo speedup (default: t_total)")
    ap.add_argument("--show", action="store_true", help="apre i grafici dopo averli salvati")
    ap.add_argument("--no-plots", action="store_true", help="produce solo CSV e report Markdown")
    a = ap.parse_args()

    rows = read_csvs(a.csv)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metric_names = {k for r in rows for k, v in r.items() if not k.startswith("_") and v not in ("", None)}
    if a.metric not in metric_names:
        raise SystemExit(
            f"Metrica '{a.metric}' non trovata nei CSV. Colonne disponibili: "
            + ", ".join(sorted(metric_names))
        )

    # Mediana delle ripetizioni per serie/configurazione/coppia.
    # W = threads * processi MPI; per MPI puro threads dovrebbe essere 1.
    samples = defaultdict(lambda: defaultdict(list))
    mpi_extra = defaultdict(lambda: {"comm_frac": [], "imbalance": []})
    for r in rows:
        T = parse_int(r.get("threads"), 1)
        P = parse_int(r.get("mpi_processes"), 1)
        W = max(1, T * P)
        series = variant_series(r["variant"], T)
        seq, i, j = r["seq"], parse_int(r["i"], 0), parse_int(r["j"], 0)
        key = (series, W, T, P, seq, i, j)
        for metric in TIME_METRICS + [a.metric]:
            if metric in r:
                value = parse_float(r.get(metric))
                if value is not None:
                    samples[key][metric].append(value)
        total = parse_float(r.get("t_total"))
        comm = parse_float(r.get("t_comm"))
        imb = parse_float(r.get("imbalance"))
        if total and total > 0 and comm is not None:
            mpi_extra[(series, W, T, P)]["comm_frac"].append(comm / total)
        if imb is not None:
            mpi_extra[(series, W, T, P)]["imbalance"].append(imb)

    med = {k: {m: median_or_nan(v) for m, v in metrics.items()}
           for k, metrics in samples.items()}

    # Baseline: mediana sequenziale della stessa sequenza e stessa coppia.
    # La chiave seq deve avere la variante esattamente "seq".
    seq_base = {}
    for key, metrics in med.items():
        series, W, T, P, seq, i, j = key
        if series == "seq" and W == 1:
            seq_base[(seq, i, j)] = metrics.get(a.metric, float("nan"))

    paired = []
    speedups = defaultdict(list)
    # (series, W, T, P, sequence) -> valori per coppia
    for key, metrics in med.items():
        series, W, T, P, seq, i, j = key
        if series == "seq":
            continue
        base = seq_base.get((seq, i, j))
        measured = metrics.get(a.metric, float("nan"))
        if base is None or not math.isfinite(base) or base <= 0:
            continue
        if not math.isfinite(measured) or measured <= 0:
            continue
        speedup = base / measured
        speedups[(series, W, T, P, seq)].append(speedup)
        paired.append({
            "variant": series, "threads": T, "mpi_processes": P, "workers": W,
            "seq": seq, "i": i, "j": j, "metric": a.metric,
            "t_seq_median": base, "t_variant_median": measured, "speedup": speedup
        })

    if not speedups:
        seen = sorted({variant_series(r["variant"], parse_int(r.get("threads"), 1)) for r in rows})
        raise SystemExit(
            "Non riesco a calcolare lo speedup: serve una baseline con variant='seq' "
            "nello stesso insieme di CSV, per le stesse coppie (seq, i, j). "
            f"Varianti trovate: {', '.join(seen)}"
        )

    summary = []
    for (series, W, T, P, seq), vals in sorted(speedups.items()):
        S = float(np.median(vals))
        eff = S / W if W > 0 else float("nan")
        if W > 1 and S > 0 and math.isfinite(S):
            kf = (1.0 / S - 1.0 / W) / (1.0 - 1.0 / W)
        else:
            kf = float("nan")
        # tempi mediani aggregati sulle coppie della stessa sequenza/configurazione
        config_keys = [k for k in med if k[0] == series and k[1] == W and k[2] == T
                       and k[3] == P and k[4] == seq]
        timing = {}
        for metric in TIME_METRICS:
            timing[metric] = median_or_nan([med[k].get(metric, float("nan")) for k in config_keys])
        ex = mpi_extra.get((series, W, T, P), {})
        summary.append({
            "variant": series, "threads": T, "mpi_processes": P, "workers": W,
            "seq": seq, "n_pairs": len(vals), "metric": a.metric,
            "speedup_median": S, "speedup_min": min(vals), "speedup_max": max(vals),
            "parallel_efficiency": eff, "karp_flatt": kf,
            **timing,
            "comm_fraction_median": median_or_nan(ex.get("comm_frac", [])),
            "imbalance_median": median_or_nan(ex.get("imbalance", [])),
        })

    write_csv(outdir / "paired_speedups.csv",
              ["variant", "threads", "mpi_processes", "workers", "seq", "i", "j",
               "metric", "t_seq_median", "t_variant_median", "speedup"], paired)
    summary_fields = [
        "variant", "threads", "mpi_processes", "workers", "seq", "n_pairs", "metric",
        "speedup_median", "speedup_min", "speedup_max", "parallel_efficiency", "karp_flatt",
        *TIME_METRICS, "comm_fraction_median", "imbalance_median"
    ]
    write_csv(outdir / "summary.csv", summary_fields, summary)

    # Stampa tabella riassuntiva.
    print(f"\nMetrica dello speedup: {a.metric}")
    print("Speedup calcolato sulle stesse coppie della baseline sequenziale.")
    print(f"{'serie':<24}{'seq':<6}{'W':>5}{'coppie':>8}{'speedup':>10}{'efficienza':>13}{'Karp-Flatt':>13}")
    for r in summary:
        kf = r["karp_flatt"]
        kf_s = f"{kf:.4f}" if math.isfinite(kf) else "n/a"
        print(f"{r['variant']:<24}{r['seq']:<6}{r['workers']:>5}{r['n_pairs']:>8}"
              f"{r['speedup_median']:>10.2f}{r['parallel_efficiency']:>13.3f}{kf_s:>13}")
    print(f"\nCSV riepilogativo: {outdir / 'summary.csv'}")
    print(f"Speedup per coppia: {outdir / 'paired_speedups.csv'}")

    # Report testuale pronto da citare/adattare nella presentazione.
    report = [
        "# Analisi dei benchmark ICP", "",
        f"- Metrica usata per lo speedup: `{a.metric}`.",
        "- Baseline: versione sequenziale, confrontata sulle stesse coppie (sequenza, i, j).",
        "- Le ripetizioni sono aggregate con la mediana.",
        "- `worker` è definito come thread × processi MPI.",
        "- L'efficienza parallela è `speedup / worker`.",
        "- Karp-Flatt è una stima effettiva della frazione seriale; può diventare negativa "
        "quando lo speedup supera il numero di worker, per rumore o effetti di cache.", "",
        "| Variante | Sequenza | Worker | Coppie | Speedup | Efficienza | Karp-Flatt |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        kf = f"{r['karp_flatt']:.4f}" if math.isfinite(r["karp_flatt"]) else "n/a"
        report.append(
            f"| {r['variant']} | {r['seq']} | {r['workers']} | {r['n_pairs']} | "
            f"{r['speedup_median']:.3f} | {r['parallel_efficiency']:.3f} | {kf} |"
        )
    report += ["", "## Note di interpretazione", "",
               "- Uno speedup maggiore di 1 indica un'accelerazione rispetto alla versione sequenziale.",
               "- Un'efficienza vicina a 1 indica che lo speedup cresce quasi linearmente con i worker.",
               "- Per confronti corretti, usare la stessa macchina, gli stessi parametri ICP e le stesse coppie.",
               "- Per MPI, la frazione di comunicazione è la mediana di `t_comm / t_total`; "
               "lo sbilanciamento è la mediana del campo `imbalance` presente nei CSV.", ""]
    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Report: {outdir / 'report.md'}")

    if a.no_plots:
        return

    try:
        import matplotlib
        if not a.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib non installato: CSV e report sono stati prodotti; grafici saltati.")
        return

    variants = sorted({r["variant"] for r in summary})
    seqs = sorted({r["seq"] for r in summary})
    table = {}
    for r in summary:
        table[(r["variant"], r["seq"], r["workers"])] = r

    def setup_x(ax, workers):
        ax.set_xlabel("Worker (thread × processi MPI)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(workers)
        ax.set_xticklabels([str(w) for w in workers])
        ax.grid(True, alpha=0.3)

    # Grafico per variante: speedup ed efficienza, una linea per sequenza.
    for variant in variants:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        workers_all = sorted({r["workers"] for r in summary if r["variant"] == variant})
        for seq in seqs:
            points = sorted(
                (r["workers"], r) for r in summary
                if r["variant"] == variant and r["seq"] == seq
            )
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1]["speedup_median"] for p in points]
            es = [p[1]["parallel_efficiency"] for p in points]
            axes[0].plot(xs, ys, "o-", label=f"sequenza {seq}")
            axes[1].plot(xs, es, "o-", label=f"sequenza {seq}")
        axes[0].plot(workers_all, workers_all, "k--", label="ideale")
        axes[0].set_ylabel("Speedup")
        axes[1].axhline(1.0, color="k", linestyle="--", label="efficienza ideale")
        axes[1].set_ylabel("Efficienza parallela")
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            setup_x(ax, workers_all)
            ax.legend(fontsize=8)
        axes[0].set_title(f"Speedup: {variant} ({a.metric})")
        axes[1].set_title("Efficienza")
        fig.tight_layout()
        path = outdir / f"speedup_{safe_name(variant)}_{safe_name(a.metric)}.png"
        fig.savefig(path, dpi=160)
        print(f"Grafico: {path}")
        if a.show:
            plt.show()
        plt.close(fig)

    # Confronto tra varianti: aggregazione delle coppie di tutte le sequenze.
    all_speedups = defaultdict(list)
    for r in paired:
        all_speedups[(r["variant"], r["workers"])].append(r["speedup"])
    workers_all = sorted({w for _, w in all_speedups})
    if len(variants) > 1 and workers_all:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for variant in variants:
            xs = sorted(w for v, w in all_speedups if v == variant)
            ys = [float(np.median(all_speedups[(variant, w)])) for w in xs]
            axes[0].plot(xs, ys, "o-", label=variant)
            axes[1].plot(xs, [s / w for s, w in zip(ys, xs)], "o-", label=variant)
            print(f"\n{variant}: " + ", ".join(f"W={w}: {s:.2f}x" for w, s in zip(xs, ys)))
        axes[0].plot(workers_all, workers_all, "k--", label="ideale")
        axes[0].set_ylabel("Speedup")
        axes[1].axhline(1.0, color="k", linestyle="--")
        axes[1].set_ylabel("Efficienza parallela")
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            setup_x(ax, workers_all)
            ax.legend(fontsize=8)
        axes[0].set_title(f"Confronto varianti ({a.metric})")
        axes[1].set_title("Efficienza")
        fig.tight_layout()
        path = outdir / f"confronto_varianti_{safe_name(a.metric)}.png"
        fig.savefig(path, dpi=160)
        print(f"Grafico: {path}")
        if a.show:
            plt.show()
        plt.close(fig)

    # Tempi per componente: mediana delle coppie, separata per worker e variante.
    component_metrics = [m for m in TIME_METRICS if m in metric_names]
    if component_metrics:
        for metric in component_metrics:
            # Non produrre un grafico se tutti i dati sono vuoti o non numerici.
            entries = []
            for (series, W, T, P, seq, i, j), vals in med.items():
                value = vals.get(metric, float("nan"))
                if math.isfinite(value):
                    entries.append((series, W, seq, value))
            if not entries:
                continue
            agg = defaultdict(list)
            for series, W, seq, value in entries:
                agg[(series, W)].append(value)
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            for series in sorted({k[0] for k in agg}):
                xs = sorted(w for s, w in agg if s == series)
                ys = [float(np.median(agg[(series, w)])) for w in xs]
                ax.plot(xs, ys, "o-", label=series)
            ax.set_xlabel("Worker (thread × processi MPI)")
            ax.set_ylabel(f"Tempo mediano {metric} (s)")
            ax.set_title(f"Tempi per componente: {metric}")
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted({k[1] for k in agg}))
            ax.set_xticklabels([str(w) for w in sorted({k[1] for k in agg})])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = outdir / f"tempi_{safe_name(metric)}.png"
            fig.savefig(path, dpi=160)
            print(f"Grafico: {path}")
            if a.show:
                plt.show()
            plt.close(fig)

    mpi_rows = []
    for (series, W, T, P), values in sorted(mpi_extra.items()):
        comm = median_or_nan(values["comm_frac"])
        imb = median_or_nan(values["imbalance"])
        if math.isfinite(comm) or math.isfinite(imb):
            mpi_rows.append((series, W, T, P, comm, imb))
    if mpi_rows:
        write_csv(outdir / "mpi_overhead.csv",
                  ["variant", "workers", "threads", "mpi_processes",
                   "comm_fraction_median", "imbalance_median"],
                  [dict(variant=s, workers=w, threads=t, mpi_processes=p,
                        comm_fraction_median=c, imbalance_median=i)
                   for s, w, t, p, c, i in mpi_rows])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for series in sorted({x[0] for x in mpi_rows}):
            sub = [x for x in mpi_rows if x[0] == series]
            sub.sort(key=lambda x: x[1])
            axes[0].plot([x[1] for x in sub], [x[4] for x in sub], "o-", label=series)
            axes[1].plot([x[1] for x in sub], [x[5] for x in sub], "o-", label=series)
        axes[0].set_ylabel("Frazione comunicazione (t_comm / t_total)")
        axes[1].set_ylabel("Sbilanciamento (max / medio)")
        for ax in axes:
            ax.set_xlabel("Worker (thread × processi MPI)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted({x[1] for x in mpi_rows}))
            ax.set_xticklabels([str(w) for w in sorted({x[1] for x in mpi_rows})])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = outdir / "mpi_overhead.png"
        fig.savefig(path, dpi=160)
        print(f"Grafico: {path}")
        if a.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()