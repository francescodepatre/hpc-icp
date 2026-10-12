import open3d as o3d
from pathlib import Path
import sys
import time


def visualizza_sequenza(cartella, pausa=0.3):
    cartella = Path(cartella)

    if not cartella.is_dir():
        print(f"Errore: cartella non trovata: {cartella}")
        return

    # Cerca tutti i file PLY e li ordina
    files = sorted(cartella.glob("*.ply"))

    if not files:
        print(f"Nessun file PLY trovato in {cartella}")
        return

    print(f"Trovati {len(files)} file PLY")

    # Crea la finestra di visualizzazione
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Sequenza Point Cloud ICP",
        width=1280,
        height=720
    )

    cloud = None

    try:
        for i, file in enumerate(files):
            print(f"Frame {i + 1}/{len(files)}: {file.name}")

            nuova_cloud = o3d.io.read_point_cloud(str(file))

            if nuova_cloud.is_empty():
                print(f"File vuoto, salto: {file.name}")
                continue

            nuova_cloud.paint_uniform_color([0.2, 0.7, 1.0])

            if cloud is None:
                cloud = nuova_cloud
                vis.add_geometry(cloud)
            else:
                cloud.points = nuova_cloud.points
                cloud.colors = nuova_cloud.colors
                cloud.normals = nuova_cloud.normals
                vis.update_geometry(cloud)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(pausa)

            if not vis.poll_events():
                break

    finally:
        vis.destroy_window()


if __name__ == "__main__":
    cartella = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../../../data/output/icp/00"
    )

    visualizza_sequenza(cartella)