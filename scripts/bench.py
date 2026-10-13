#!/usr/bin/env python3
"""
Benchmark multi-sequenza per le versioni di icp (seq, omp, ...).

Esempio (dalla radice del progetto):
  python3 scripts/bench.py --data data --sequences 00 05 --pairs 10 --step 1 \
      --exe seq=build/icp_seq omp=build/icp_omp --threads 1 2 4 8 \
      --voxel 0.3 --dmax 1.5 --iters 30 --reps 5 --out results/bench.csv

Per ogni sequenza sceglie `--pairs` coppie equispaziate (target = scansione i, sorgente = i+step),
calcola il ground truth dalle pose (solo sequenze 00-10) e lancia ogni eseguibile.
 - le varianti il cui nome inizia per "omp" vengono lanciate per ogni valore di --threads
 - le altre (seq) una sola volta
Ogni configurazione fa prima `--warmup` esecuzioni scartate, poi `--reps` misurate.
Le righe 'CSV,...' stampate dagli eseguibili vengono raccolte in un unico file CSV.

Opzioni utili:
 --schedules static dynamic,64 guided   confronta piu' schedule OpenMP nello stesso file: le
                                        varianti si chiamano omp[static], omp[dynamic,64], ...
 Le ripetizioni sono INTERCALATE tra le configurazioni (ordine casuale a ogni giro, --seed):
 cosi' deriva termica, frequenza della CPU e processi di sistema pesano allo stesso modo su
 tutte le configurazioni, invece di favorire quella misurata per ultima.
"""
import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

FIELDS = ["variant", "threads", "seq", "i", "j", "rep", "n_src", "n_tgt", "iters", "dmax",
          "t_grid", "t_apply", "t_nn", "t_reduce", "t_solve", "t_total", "rot_err", "trans_err"]


def to4x4(vals):
    m = np.eye(4)
    m[:3, :4] = np.asarray(vals, dtype=float).reshape(3, 4)
    return m


def load_poses(path):
    return [to4x4(line.split()) for line in open(path) if line.strip()]


def load_tr(path):
    for line in open(path):
        if line.startswith("Tr"):
            return to4x4(line.split(":", 1)[1].split())
    raise ValueError(f"{path}: riga 'Tr' non trovata")


def write_gt(path, Pi, Pj, Tr):
    G = np.linalg.inv(Tr) @ np.linalg.inv(Pi) @ Pj @ Tr   # sorgente j -> target i (Velodyne)
    np.savetxt(path, G, fmt="%.12f")


def parse_csv_line(out):
    for line in out.splitlines():
        if line.startswith("CSV,"):
            return line.strip().split(",")[1:]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="cartella che contiene data_odometry_*")
    ap.add_argument("--sequences", nargs="+", default=["00"])
    ap.add_argument("--pairs", type=int, default=10, help="coppie per sequenza")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1, help="distanza tra le scansioni di una coppia")
    ap.add_argument("--exe", nargs="+", required=True, help="nome=percorso, es. seq=build/icp_seq")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--voxel", type=float, default=0.3)
    ap.add_argument("--dmax", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--schedules", nargs="+", default=None,
                    help="schedule OpenMP da confrontare (valori di OMP_SCHEDULE)")
    ap.add_argument("--seed", type=int, default=1, help="seme per l'ordine casuale delle ripetizioni")
    ap.add_argument("--no-bind", action="store_true", help="non impostare OMP_PROC_BIND/OMP_PLACES")
    ap.add_argument("--out", default="results/bench.csv")
    a = ap.parse_args()

    data = Path(a.data)
    exes = []
    for e in a.exe:
        name, path = e.split("=", 1)
        if not Path(path).exists():
            sys.exit(f"eseguibile non trovato: {path}")
        exes.append((name, path))

    rng = random.Random(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out).parent / "_gt_tmp.txt"
    out_f = open(a.out, "w", newline="")
    w = csv.writer(out_f, lineterminator="\n")
    w.writerow(FIELDS)

    for seq in a.sequences:
        vel = data / "data_odometry_velodyne/dataset/sequences" / seq / "velodyne"
        if not vel.is_dir():
            print(f"[{seq}] cartella velodyne non trovata, salto")
            continue
        n_scans = len(list(vel.glob("*.bin")))
        poses_f = data / "data_odometry_poses/dataset/poses" / f"{seq}.txt"
        calib_f = data / "data_odometry_calib/dataset/sequences" / seq / "calib.txt"
        have_gt = poses_f.exists() and calib_f.exists()
        poses, Tr = (load_poses(poses_f), load_tr(calib_f)) if have_gt else (None, None)
        if not have_gt:
            print(f"[{seq}] nessun ground truth (pose assenti): errori non calcolati")

        span = n_scans - a.step - a.start
        if span <= 0:
            print(f"[{seq}] troppo poche scansioni ({n_scans}), salto")
            continue
        spacing = max(1, span // a.pairs)
        idxs = [a.start + k * spacing for k in range(a.pairs) if a.start + k * spacing + a.step < n_scans]

        for i in idxs:
            j = i + a.step
            tgt = vel / f"{i:06d}.bin"
            src = vel / f"{j:06d}.bin"
            gt_args = []
            if have_gt:
                write_gt(tmp, poses[i], poses[j], Tr)
                gt_args = ["--gt", str(tmp)]

            # elenco delle configurazioni da misurare su questa coppia
            configs = []
            for name, path in exes:
                is_omp = name.startswith("omp")
                tlist = a.threads if is_omp else [1]
                scheds = (a.schedules if (is_omp and a.schedules) else [None])
                for T in tlist:
                    for sch in scheds:
                        env = dict(os.environ)
                        label = name if sch is None else f"{name}[{sch}]"
                        cmd = [path, str(src), str(tgt), "--voxel", str(a.voxel), "--dmax", str(a.dmax),
                               "--iters", str(a.iters)] + gt_args
                        if is_omp:
                            env["OMP_NUM_THREADS"] = str(T)
                            if sch is not None:
                                env["OMP_SCHEDULE"] = sch
                            if not a.no_bind:
                                env["OMP_PROC_BIND"] = "close"
                                env["OMP_PLACES"] = "cores"
                            cmd += ["--threads", str(T)]
                        configs.append((label, T, path, cmd, env))

            def esegui(cfg):
                label, T, path, cmd, env = cfg
                res = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if res.returncode != 0:
                    sys.exit(f"errore in {' '.join(cmd)}\n{res.stderr}")
                f = parse_csv_line(res.stdout)
                if f is None:
                    sys.exit(f"riga CSV non trovata nell'output di {path}")
                return f

            for cfg in configs:                       # warm-up (scartato)
                for _ in range(a.warmup):
                    esegui(cfg)

            for r in range(a.reps):                   # misure intercalate, ordine casuale
                ordine = configs[:]
                rng.shuffle(ordine)
                for cfg in ordine:
                    label, T = cfg[0], cfg[1]
                    f = esegui(cfg)
                    # f = variante,N,M,iters,dmax,grid,apply,nn,reduce,solve,total,rot,trans[,threads]
                    w.writerow([label, T, seq, i, j, r, f[1], f[2], f[3], f[4],
                                f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12]])
                    out_f.flush()
            print(f"[{seq}] coppia {i}-{j}: {len(configs)} configurazioni x {a.reps} ripetizioni ok")

    out_f.close()
    if tmp.exists():
        tmp.unlink()
    print(f"\nrisultati in {a.out}")


if __name__ == "__main__":
    main()
