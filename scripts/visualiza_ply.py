import open3d as o3d
import os
import sys


def visualizza_ply(percorso):
    # Verifica che il file esista
    if not os.path.isfile(percorso):
        print(f"Errore: file non trovato: {percorso}")
        sys.exit(1)

    # Carica la nuvola di punti
    cloud = o3d.io.read_point_cloud(percorso)

    if cloud.is_empty():
        print("Errore: la nuvola di punti è vuota o il file non è valido.")
        sys.exit(1)

    print(f"File: {percorso}")
    print(f"Numero di punti: {len(cloud.points)}")

    # Visualizza le coordinate dei primi 5 punti
    print("\nPrimi 5 punti:")
    for punto in list(cloud.points)[:5]:
        print(f"x={punto[0]:.4f}, y={punto[1]:.4f}, z={punto[2]:.4f}")

    # Colora la nuvola
    cloud.paint_uniform_color([0.2, 0.7, 1.0])

    # Visualizza la nuvola in 3D
    o3d.visualization.draw_geometries(
        [cloud],
        window_name="Point Cloud Allineata - ICP",
        width=1280,
        height=720,
        point_show_normal=False
    )


if __name__ == "__main__":
    percorso = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../data/output/aligned.ply"
    )

    visualizza_ply(percorso)