#!/usr/bin/env python3
"""
Ground truth relativo tra due scansioni KITTI Odometry (Velodyne), nel formato di icp_seq --gt.

Uso:
  python3 kitti_gt.py --poses <.../poses/00.txt> --calib <.../sequences/00/calib.txt> \
                      --i 0 --j 1 --out ../data/kitti_gt.txt

Convenzione: ICP con sorgente = scansione j e target = scansione i. Il file contiene la
matrice 4x4 G che porta i punti della scansione j nel sistema del Velodyne della scansione i:

    G = Tr^-1 * P_i^-1 * P_j * Tr

 - P_k  : posa della camera k (riga k di poses/XX.txt, 3x4), camera k -> camera 0
 - Tr   : calibrazione Velodyne -> camera 0 (riga "Tr" di calib.txt)
"""
import argparse
import numpy as np


def to4x4(v):
    m = np.eye(4)
    m[:3, :4] = np.asarray(v, dtype=float).reshape(3, 4)
    return m


def read_pose(path, k):
    with open(path) as f:
        for n, line in enumerate(f):
            if n == k:
                vals = line.split()
                if len(vals) != 12:
                    raise ValueError(f"riga {k} di {path}: attesi 12 numeri")
                return to4x4(vals)
    raise IndexError(f"{path}: la scansione {k} non esiste")


def read_tr(path):
    with open(path) as f:
        for line in f:
            if line.startswith("Tr"):  # "Tr:" nell'odometry, "Tr_velo_to_cam:" nel raw
                return to4x4(line.split(":", 1)[1].split())
    raise ValueError(f"{path}: riga 'Tr' non trovata")


def relative_gt(Pi, Pj, Tr):
    return np.linalg.inv(Tr) @ np.linalg.inv(Pi) @ Pj @ Tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--i", type=int, required=True, help="indice della scansione target")
    ap.add_argument("--j", type=int, required=True, help="indice della scansione sorgente")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    G = relative_gt(read_pose(a.poses, a.i), read_pose(a.poses, a.j), read_tr(a.calib))
    np.savetxt(a.out, G, fmt="%.12f")

    ang = np.degrees(np.arccos(np.clip((np.trace(G[:3, :3]) - 1) / 2, -1, 1)))
    print(f"scritto {a.out}")
    print(f"movimento tra le due scansioni: traslazione {np.linalg.norm(G[:3, 3]):.3f} m, rotazione {ang:.3f} gradi")


if __name__ == "__main__":
    main()