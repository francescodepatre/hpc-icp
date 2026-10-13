#!/usr/bin/env python3
"""
Riproduce una sequenza di nuvole di punti (cartella di .ply oppure di .bin KITTI).

  python3 visualizza_sequenza.py CARTELLA [--fps 5] [--voxel 0.2] [--start 0] [--step 1]

Esempio con KITTI:
  python3 visualizza_sequenza.py data/data_odometry_velodyne/dataset/sequences/00/velodyne --voxel 0.2

Colore = altezza (z). La camera NON viene resettata a ogni frame, quindi si puo' ruotare/zoomare.
Tasti:  SPAZIO = pausa/riprendi | freccia destra/sinistra = frame successivo/precedente
        + / - = piu' veloce / piu' lento | Q = esci
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def carica_frame(percorso):
    import open3d as o3d

    if percorso.suffix.lower() == ".bin":
        return np.fromfile(percorso, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    return np.asarray(o3d.io.read_point_cloud(str(percorso)).points)


def colori_altezza(punti):
    """Gradiente blu -> verde -> rosso in funzione di z (percentili 2-98 per ignorare gli estremi)."""
    z = punti[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.stack([t, 1 - np.abs(2 * t - 1), 1 - t], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cartella")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--voxel", type=float, default=0.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()

    import open3d as o3d

    cart = Path(a.cartella)
    if not cart.is_dir():
        sys.exit(f"Errore: cartella non trovata: {cart}")
    files = sorted(list(cart.glob("*.ply")) + list(cart.glob("*.bin")))[a.start::a.step]
    if not files:
        sys.exit(f"Nessun file .ply o .bin in {cart}")
    print(f"{len(files)} frame in {cart}")

    stato = {"i": 0, "pausa": False, "fps": a.fps, "ultimo": 0.0, "mostrato": -1}

    def costruisci(i):
        pts = carica_frame(files[i])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if a.voxel > 0:
            pcd = pcd.voxel_down_sample(a.voxel)
        p = np.asarray(pcd.points)
        pcd.colors = o3d.utility.Vector3dVector(colori_altezza(p))
        return pcd

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Sequenza point cloud", width=1280, height=720)
    cloud = costruisci(0)
    vis.add_geometry(cloud)
    vis.get_render_option().point_size = 1.5
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.08])
    stato["mostrato"] = 0

    def vai(delta):
        def cb(v):
            stato["i"] = (stato["i"] + delta) % len(files)
            stato["ultimo"] = 0.0  # forza l'aggiornamento al prossimo giro
            return False
        return cb

    def pausa(v):
        stato["pausa"] = not stato["pausa"]
        return False

    def velocita(f):
        def cb(v):
            stato["fps"] = min(60.0, max(0.5, stato["fps"] * f))
            print(f"fps: {stato['fps']:.1f}")
            return False
        return cb

    chiudi = {"v": False}

    def esci(v):
        chiudi["v"] = True
        return False

    vis.register_key_callback(32, pausa)
    vis.register_key_callback(262, vai(+1))   # freccia destra
    vis.register_key_callback(263, vai(-1))   # freccia sinistra
    vis.register_key_callback(ord("="), velocita(1.5))
    vis.register_key_callback(ord("+"), velocita(1.5))
    vis.register_key_callback(ord("-"), velocita(1 / 1.5))
    vis.register_key_callback(ord("Q"), esci)

    print("Tasti: SPAZIO pausa | frecce = frame | +/- velocita | Q esci")
    try:
        # Loop non bloccante: niente time.sleep lunghi, cosi' la finestra resta reattiva
        while vis.poll_events() and not chiudi["v"]:
            ora = time.time()
            if not stato["pausa"] and ora - stato["ultimo"] >= 1.0 / stato["fps"] \
                    and stato["i"] == stato["mostrato"]:
                stato["i"] = (stato["i"] + 1) % len(files)
            if stato["i"] != stato["mostrato"]:
                nuovo = costruisci(stato["i"])
                cloud.points, cloud.colors = nuovo.points, nuovo.colors
                vis.update_geometry(cloud)
                stato["mostrato"] = stato["i"]
                stato["ultimo"] = ora
                print(f"frame {stato['i'] + 1}/{len(files)}: {files[stato['i']].name}", end="\r")
            vis.update_renderer()
            time.sleep(0.005)
    finally:
        vis.destroy_window()
        print()


if __name__ == "__main__":
    main()