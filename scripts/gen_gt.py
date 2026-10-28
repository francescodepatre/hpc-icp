#!/usr/bin/env python3
"""
[LOCALE] Genera ground truth + manifest delle coppie, replicando ESATTAMENTE la selezione di bench.py.
Riusa kitti_gt.py (stessa cartella) per la formula G = Tr^-1 Pi^-1 Pj Tr. Richiede solo numpy.

Uso:
  python3 gen_gt.py --data data --sequences 00 05 --pairs 10 --step 1 --out hpc_in
Produce:
  hpc_in/gt/<seq>/<i>_<j>.txt   matrice 4x4 (stesso formato di kitti_gt.py)
  hpc_in/pairs.tsv              colonne: seq  i  j  has_gt   (letto da run_bench.sh)
  hpc_in/files.txt              path dei .bin necessari, relativi a --data (per rsync --files-from)
Usa gli stessi --pairs/--step/--start che darai al job.
"""
import argparse
from pathlib import Path
import numpy as np
from kitti_gt import read_tr, relative_gt, to4x4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sequences", nargs="+", default=["00"])
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default="hpc_in")
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest, files = [], []
    for seq in a.sequences:
        vel = data / "data_odometry_velodyne/dataset/sequences" / seq / "velodyne"
        if not vel.is_dir():
            print(f"[{seq}] cartella velodyne non trovata, salto"); continue
        n_scans = len(list(vel.glob("*.bin")))
        poses_f = data / "data_odometry_poses/dataset/poses" / f"{seq}.txt"
        calib_f = data / "data_odometry_calib/dataset/sequences" / seq / "calib.txt"
        have_gt = poses_f.exists() and calib_f.exists()
        if have_gt:
            poses = [to4x4(l.split()) for l in open(poses_f) if l.strip()]
            Tr = read_tr(calib_f)
            (out / "gt" / seq).mkdir(parents=True, exist_ok=True)
        span = n_scans - a.step - a.start
        if span <= 0:
            print(f"[{seq}] troppo poche scansioni ({n_scans}), salto"); continue
        spacing = max(1, span // a.pairs)
        idxs = [a.start + k * spacing for k in range(a.pairs) if a.start + k * spacing + a.step < n_scans]
        for i in idxs:
            j = i + a.step
            if have_gt:
                np.savetxt(out / "gt" / seq / f"{i}_{j}.txt", relative_gt(poses[i], poses[j], Tr), fmt="%.12f")
            manifest.append(f"{seq}\t{i}\t{j}\t{int(have_gt)}")
            for k in (i, j):
                files.append(str(vel.relative_to(data) / f"{k:06d}.bin"))
        print(f"[{seq}] {len(idxs)} coppie, gt={'si' if have_gt else 'NO'}")
    (out / "pairs.tsv").write_text("\n".join(manifest) + "\n")
    (out / "files.txt").write_text("\n".join(sorted(set(files))) + "\n")
    print(f"scritti {out}/pairs.tsv ({len(manifest)} coppie), files.txt ({len(set(files))} scansioni)")


if __name__ == "__main__":
    main()
