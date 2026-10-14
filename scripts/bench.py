#!/usr/bin/env python3
"""
Benchmark multi-sequenza per le versioni di icp: seq, omp, mpi (MPI puro), hybrid (MPI+OpenMP).

Esempio (dalla radice del progetto):
  python3 scripts/bench.py --data data --sequences 00 05 --pairs 10 --step 1 \
      --exe seq=build/icp_seq omp=build/icp_omp mpi=build/icp_mpi hybrid=build/icp_mpi \
      --threads 2 4 --mpi-processes 1 2 4 \
      --voxel 0.1 --dmax 1.5 --iters 30 --reps 5 --out results/bench.csv

Il NOME prima di '=' decide come si lancia l'eseguibile:
  seq*     un solo processo, un solo thread
  omp*     OMP_NUM_THREADS e --threads per ogni valore di --threads
  mpi*     mpirun -np P per ogni valore di --mpi-processes, 1 thread per rank (MPI puro)
  hybrid*  mpirun -np P e --threads T: tutte le combinazioni P x T (stesso eseguibile di mpi)

Per ogni sequenza sceglie `--pairs` coppie equispaziate (target = scansione i, sorgente = i+step),
calcola il ground truth dalle pose (solo sequenze 00-10) e lancia ogni configurazione.
Ogni configurazione fa prima `--warmup` esecuzioni scartate, poi `--reps` misurate.
Le ripetizioni sono INTERCALATE tra le configurazioni (ordine casuale a ogni giro, --seed):
deriva termica e processi di sistema pesano allo stesso modo su tutte.

Opzioni utili:
  --schedules static dynamic,64    schedule OpenMP (omp e hybrid); la variante si chiama es. omp[static]
  --partitions block cyclic        distribuzione dei punti tra i rank (mpi e hybrid); es. mpi[cyclic]
  --collectives reduce allreduce   collettiva per iterazione (mpi e hybrid); es. mpi[allreduce]
  --mpi-flags=--oversubscribe      flag extra per mpirun (OpenMPI; usa --mpi-flags= con srun/MPICH).
                                   Scrivi SEMPRE col segno "=": senza, argparse scambia "--..." per un'opzione.
                                   Piu' flag: --mpi-flags="--oversubscribe --bind-to none"
  --mpiexec mpirun                 launcher (es. srun sul cluster SLURM)
"""
import argparse
import csv
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

FIELDS = ["variant", "threads", "mpi_processes", "seq", "i", "j", "rep",
          "n_src", "n_tgt", "iters", "dmax",
          "t_grid", "t_apply", "t_nn", "t_reduce", "t_solve", "t_total",
          "rot_err", "trans_err", "t_comm", "imbalance"]


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


def field(f, k, default=""):
    return f[k] if k < len(f) else default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="cartella che contiene data_odometry_*")
    ap.add_argument("--sequences", nargs="+", default=["00"])
    ap.add_argument("--pairs", type=int, default=10, help="coppie per sequenza")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1, help="distanza tra le scansioni di una coppia")
    ap.add_argument("--exe", nargs="+", required=True, help="nome=percorso, es. seq=build/icp_seq")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--mpi-processes", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--mpiexec", default="mpirun")
    ap.add_argument("--mpi-flags", default="--oversubscribe",
                    help='flag extra per il launcher MPI; "" per nessuno')
    ap.add_argument("--voxel", type=float, default=0.3)
    ap.add_argument("--dmax", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--schedules", nargs="+", default=None)
    ap.add_argument("--partitions", nargs="+", default=None, choices=["block", "cyclic"])
    ap.add_argument("--collectives", nargs="+", default=None, choices=["reduce", "allreduce"])
    ap.add_argument("--seed", type=int, default=1)
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

    def esegui(cfg):
        label, T, P, path, cmd, env = cfg
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"errore nell'esecuzione di:\n  {' '.join(cmd)}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
        f = parse_csv_line(res.stdout)
        if f is None:
            sys.exit(f"riga CSV non trovata nell'output di {path}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
        return f

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

            # ---- elenco delle configurazioni da misurare su questa coppia
            configs = []
            for name, path in exes:
                is_hybrid = name.startswith("hybrid")
                is_mpi = name.startswith("mpi") or is_hybrid
                is_omp = name.startswith("omp") or is_hybrid

                tlist = a.threads if is_omp else [1]
                plist = a.mpi_processes if is_mpi else [1]
                scheds = a.schedules if (is_omp and a.schedules) else [None]
                parts = a.partitions if (is_mpi and a.partitions) else [None]
                colls = a.collectives if (is_mpi and a.collectives) else [None]

                for P in plist:
                    for T in tlist:
                        for sch in scheds:
                            for part in parts:
                                for col in colls:
                                    env = dict(os.environ)
                                    # L'etichetta contiene solo le OPZIONI (schedule, partizione, collettiva);
                                    # np e thread sono nelle colonne 'mpi_processes' e 'threads'.
                                    label = name
                                    for tag in (sch, part, col):
                                        if tag is not None:
                                            label += f"[{tag}]"

                                    app = [path, str(src), str(tgt), "--voxel", str(a.voxel),
                                           "--dmax", str(a.dmax), "--iters", str(a.iters)] + gt_args
                                    if is_omp:
                                        env["OMP_NUM_THREADS"] = str(T)
                                        if sch is not None:
                                            env["OMP_SCHEDULE"] = sch
                                        if not a.no_bind:
                                            env["OMP_PROC_BIND"] = "close"
                                            env["OMP_PLACES"] = "cores"
                                        app += ["--threads", str(T)]
                                    elif is_mpi:
                                        app += ["--threads", "1"]       # MPI puro: 1 thread per rank
                                    if is_mpi:
                                        if part is not None:
                                            app += ["--partition", part]
                                        if col is not None:
                                            app += ["--collective", col]
                                        cmd = [a.mpiexec] + shlex.split(a.mpi_flags) + ["-np", str(P)] + app
                                    else:
                                        cmd = app
                                    configs.append((label, T, P, path, cmd, env))

            # ---- warm-up (scartato) e misure intercalate in ordine casuale
            for cfg in configs:
                for _ in range(a.warmup):
                    esegui(cfg)

            for r in range(a.reps):
                ordine = configs[:]
                rng.shuffle(ordine)
                for cfg in ordine:
                    label, T, P = cfg[0], cfg[1], cfg[2]
                    f = esegui(cfg)
                    # f = variante,N,M,iters,dmax,grid,apply,nn,reduce,solve,total,rot,trans[,...]
                    # MPI: ...,trans,np,threads,comm,sbilanciamento   (indici 13..16)
                    w.writerow([label, T, P, seq, i, j, r,
                                f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12],
                                field(f, 15), field(f, 16)])
                    out_f.flush()
            print(f"[{seq}] coppia {i}-{j}: {len(configs)} configurazioni x {a.reps} ripetizioni ok")

    out_f.close()
    if tmp.exists():
        tmp.unlink()
    print(f"\nrisultati in {a.out}")


if __name__ == "__main__":
    main()
