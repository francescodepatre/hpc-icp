#!/usr/bin/env bash

set -u

# Cartella principale del dataset KITTI
DATASET="data/data_odometry_velodyne/dataset/sequences"

# Eseguibile C++ ICP
ICP_EXEC="./build/icp_seq"

# Cartella di output
OUTPUT="./data/output/icp/"

# Parametri ICP
VOXEL=0.3
DMAX=1.5
ITERS=30

# Crea la cartella di output
mkdir -p "$OUTPUT"

# CSV riepilogativo
CSV="$OUTPUT/risultati.csv"

echo "sequenza,frame_source,frame_target,file_output,return_code" > "$CSV"

# Contatori
TOTAL=0
OK=0
ERRORS=0

# Sequenze KITTI da 00 a 20
for seq_num in $(seq -w 0 20); do

    VEL_DIR="$DATASET/$seq_num/velodyne"

    if [[ ! -d "$VEL_DIR" ]]; then
        echo "[SKIP] Sequenza $seq_num: cartella non trovata"
        continue
    fi

    # Recupera i file .bin in ordine
    mapfile -t FILES < <(find "$VEL_DIR" -maxdepth 1 -type f -name "*.bin" | sort)

    N=${#FILES[@]}

    if (( N < 2 )); then
        echo "[SKIP] Sequenza $seq_num: meno di 2 scansioni"
        continue
    fi

    echo
    echo "=========================================="
    echo "Sequenza $seq_num: $N scansioni"
    echo "=========================================="

    # Cartella di output per la sequenza
    SEQ_OUTPUT="$OUTPUT/$seq_num"
    mkdir -p "$SEQ_OUTPUT"

    # Elabora tutte le coppie consecutive
    for ((i = 0; i < N - 1; i++)); do

        SOURCE="${FILES[$i]}"
        TARGET="${FILES[$((i + 1))]}"

        SOURCE_NAME=$(basename "$SOURCE")
        TARGET_NAME=$(basename "$TARGET")

        FRAME_NAME="${SOURCE_NAME%.bin}"
        OUTPUT_PLY="$SEQ_OUTPUT/${FRAME_NAME}_aligned.ply"

        echo "[$seq_num] $SOURCE_NAME -> $TARGET_NAME"

        # Esegui ICP
        if "$ICP_EXEC" \
            "$SOURCE" \
            "$TARGET" \
            --voxel "$VOXEL" \
            --dmax "$DMAX" \
            --iters "$ITERS" \
            --out "$OUTPUT_PLY"; then

            RETURN_CODE=0
            ((OK += 1))
            echo "  OK"

        else
            RETURN_CODE=$?
            ((ERRORS += 1))
            echo "  ERRORE: codice $RETURN_CODE"
        fi

        ((TOTAL += 1))

        # Registra il risultato nel CSV
        echo "$seq_num,$SOURCE_NAME,$TARGET_NAME,$OUTPUT_PLY,$RETURN_CODE" >> "$CSV"

    done
done

echo
echo "=========================================="
echo "ELABORAZIONE TERMINATA"
echo "=========================================="
echo "Coppie elaborate: $TOTAL"
echo "Successi:         $OK"
echo "Errori:           $ERRORS"
echo "Risultati:        $OUTPUT"
echo "CSV:              $CSV"