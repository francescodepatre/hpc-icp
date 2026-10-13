#!/usr/bin/env python3
"""
Confronto visivo prima/dopo ICP.

  python3 visualizza_icp.py SORGENTE TARGET [--aligned aligned.ply] [--gt gt.txt] [--voxel 0.2]

 - SORGENTE e TARGET possono essere .ply oppure .bin (KITTI Velodyne)
 - --aligned : sorgente allineata scritta da  icp_seq/icp_omp --out aligned.ply
 - --gt      : matrice 4x4 vera (sorgente -> target), la stessa usata da  icp_* --gt
               (la sorgente trasformata con il ground truth e' il riferimento "corretto")
 - --voxel   : downsampling solo per la visualizzazione (nuvole grandi)

Colori: target = blu, sorgente iniziale = rosso, allineata (ICP) = verde, ground truth = arancione.

Tasti nella finestra:
  N / SPAZIO : stato successivo della sorgente (iniziale -> ICP -> ground truth)
  T          : mostra/nascondi il target
  +  /  -    : ingrandisce/riduce i punti
  Q          : esci
"""
import argparse
import sys
from pathlib import Path

import numpy as np

COL_TARGET = [0.10, 0.40, 0.90]
COL_BEFORE = [0.90, 0.20, 0.20]
COL_ICP = [0.10, 0.70, 0.30]
COL_GT = [1.00, 0.60, 0.00]


def carica_punti(percorso):
    """Ritorna un array (N,3) float64 da .ply o .bin KITTI."""
    import open3d as o3d

    p = Path(percorso)
    if not p.is_file():
        sys.exit(f"Errore: file non trovato: {p}")
    if p.suffix.lower() == ".bin":
        if p.stat().st_size % 16 != 0:
            sys.exit(f"Errore: {p} non sembra un .bin KITTI (dimensione non multipla di 16 byte)")
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    else:
        pts = np.asarray(o3d.io.read_point_cloud(str(p)).points)
    if len(pts) == 0:
        sys.exit(f"Errore: nuvola vuota o file non valido: {p}")
    return pts


def a_pcd(punti, colore, voxel=0.0):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(punti))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.paint_uniform_color(colore)
    return pcd


def leggi_matrice(percorso):
    m = np.loadtxt(percorso)
    if m.shape != (4, 4):
        sys.exit(f"Errore: {percorso} deve contenere una matrice 4x4")
    return m


def applica(m, punti):
    return punti @ m[:3, :3].T + m[:3, 3]


def costruisci_stati(sorgente, aligned=None, gt=None):
    """Lista di (nome, punti, colore) per la sorgente nei vari stati."""
    stati = [("sorgente iniziale (rosso)", sorgente, COL_BEFORE)]
    if aligned is not None:
        stati.append(("allineata con ICP (verde)", aligned, COL_ICP))
    if gt is not None:
        stati.append(("ground truth (arancione)", applica(gt, sorgente), COL_GT))
    return stati


def distanza_media_vicini(a, b, voxel_stat=0.0):
    """Distanza media dal vicino piu' prossimo di a in b (solo per il report a console)."""
    import open3d as o3d

    pa = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a))
    pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b))
    return float(np.mean(np.asarray(pa.compute_point_cloud_distance(pb))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sorgente")
    ap.add_argument("target")
    ap.add_argument("--aligned")
    ap.add_argument("--gt")
    ap.add_argument("--voxel", type=float, default=0.0)
    ap.add_argument("--point-size", type=float, default=2.0)
    a = ap.parse_args()

    import open3d as o3d

    src = carica_punti(a.sorgente)
    tgt = carica_punti(a.target)
    ali = carica_punti(a.aligned) if a.aligned else None
    gt = leggi_matrice(a.gt) if a.gt else None
    if ali is not None and len(ali) != len(src):
        # --voxel di icp_* riduce la sorgente: la sorgente allineata ha meno punti. Va bene lo stesso.
        print(f"nota: sorgente {len(src)} punti, allineata {len(ali)} punti (probabile --voxel in icp_*)")

    stati = costruisci_stati(src, ali, gt)

    # Report numerico a console: distanza media sorgente -> target per ogni stato
    print(f"target: {len(tgt)} punti | sorgente: {len(src)} punti")
    for nome, pts, _ in stati:
        print(f"  distanza media al target, {nome}: {distanza_media_vicini(pts, tgt):.4f}")

    geom_tgt = a_pcd(tgt, COL_TARGET, a.voxel)
    stato = {"i": 0, "tgt": True, "size": a.point_size}
    mobile = a_pcd(stati[0][1], stati[0][2], a.voxel)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="ICP: " + stati[0][0], width=1280, height=720)
    vis.add_geometry(geom_tgt)
    vis.add_geometry(mobile)
    vis.get_render_option().point_size = stato["size"]
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.08])

    def prossimo(v):
        stato["i"] = (stato["i"] + 1) % len(stati)
        nome, pts, col = stati[stato["i"]]
        nuovo = a_pcd(pts, col, a.voxel)
        mobile.points, mobile.colors = nuovo.points, nuovo.colors
        v.update_geometry(mobile)
        print("sorgente mostrata:", nome)
        return False

    def toggle_target(v):
        stato["tgt"] = not stato["tgt"]
        if stato["tgt"]:
            v.add_geometry(geom_tgt, reset_bounding_box=False)
        else:
            v.remove_geometry(geom_tgt, reset_bounding_box=False)
        return False

    def punti(delta):
        def cb(v):
            stato["size"] = max(1.0, stato["size"] + delta)
            v.get_render_option().point_size = stato["size"]
            return False
        return cb

    vis.register_key_callback(ord("N"), prossimo)
    vis.register_key_callback(32, prossimo)  # SPAZIO
    vis.register_key_callback(ord("T"), toggle_target)
    vis.register_key_callback(ord("="), punti(+1))   # '+' senza shift
    vis.register_key_callback(ord("+"), punti(+1))
    vis.register_key_callback(ord("-"), punti(-1))
    vis.register_key_callback(ord("Q"), lambda v: v.close() or False)

    print("\nTasti: N/SPAZIO = cambia stato | T = target on/off | +/- = dimensione punti | Q = esci")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()