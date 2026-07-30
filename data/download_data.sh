#!/bin/bash
# Télécharge le dataset COVID-19 Radiography Database depuis Kaggle
# et l'organise en normal/ viral/ covid/ test/ (30 images/classe en test)

set -e

DATASET="tawsifurrahman/covid19-radiography-database"
ROOT_DIR="COVID-19 Radiography Database"

echo "Téléchargement du dataset depuis Kaggle..."
kaggle datasets download -d "$DATASET"

echo "Décompression..."
unzip -o covid19-radiography-database.zip -d "$ROOT_DIR"

echo "Organisation des dossiers (normal / viral / covid / test)..."
python3 - <<'PYEOF'
import os
import random
import shutil

random.seed(0)

root_dir = "COVID-19 Radiography Database"
source_dirs = ["NORMAL", "Viral Pneumonia", "COVID-19"]
class_names = ["normal", "viral", "covid"]

if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.makedirs(os.path.join(root_dir, "test"), exist_ok=True)

    for i, d in enumerate(source_dirs):
        src = os.path.join(root_dir, d)
        dst = os.path.join(root_dir, class_names[i])
        if os.path.isdir(src):
            os.rename(src, dst)

    for c in class_names:
        os.makedirs(os.path.join(root_dir, "test", c), exist_ok=True)

    for c in class_names:
        class_dir = os.path.join(root_dir, c)
        images = [x for x in os.listdir(class_dir) if x.lower().endswith("png")]
        selected = random.sample(images, min(30, len(images)))
        for image in selected:
            shutil.move(
                os.path.join(class_dir, image),
                os.path.join(root_dir, "test", c, image),
            )
    print("Organisation terminée.")
else:
    print("Dossiers déjà organisés (ou structure inattendue) — rien à faire.")
PYEOF

echo "Terminé ! Le dataset est prêt dans '$ROOT_DIR'."
