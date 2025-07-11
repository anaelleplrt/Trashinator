from ultralytics import YOLO
import os

# Charger ton modèle entraîné
model_path = "runs/detect_poubelle/weights/best.pt"  
model = YOLO(model_path)

# Dossier d'images à traiter (images originales NON crop)
image_folder = "Data/train/no_label"  

# Dossier de sortie pour les images annotées
output_folder = "runs/test_detect"

# Détection sur toutes les images du dossier
results = model.predict(
    source=image_folder, 
    save=True,                  # Sauvegarde des images annotées
    save_crop=True,             # Sauvegarde les crops des objets détectés
    project="runs",             # Dossier racine
    name="test_detect",         # Sous-dossier créé automatiquement
    exist_ok=True               # Ne pas écraser si déjà existant
)

print("✅ Détection terminée.")
print(f"Images annotées sauvegardées dans : {output_folder}")
print(f"Images croppées sauvegardées dans : {os.path.join(output_folder, 'crops/poubelle')}")
