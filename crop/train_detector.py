from ultralytics import YOLO

# =====================================
# 📄 Charger un modèle pré-entraîné
# =====================================
model = YOLO('yolov8n.pt')  # modèle nano (léger et rapide), tu peux aussi tester 'yolov8s.pt'

# =====================================
# 📄 Entraîner le modèle
# =====================================
model.train(
    data='dataset.yaml',  # chemin vers ton fichier yaml que tu dois créer (voir ci-dessous)
    epochs=50,            # nombre d'époques
    imgsz=640,            # taille des images
    batch=8,              # batch size (adapter selon ta VRAM)
    workers=4,            # nombre de workers
    device='cpu',  # utiliser CPU si pas de GPU 
    project='runs',      # dossier principal où tout sera sauvegardé
    name='detect_poubelle',  # sous-dossier spécifique
    verbose=True
)
