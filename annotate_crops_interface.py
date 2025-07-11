import cv2
import os
import shutil

INPUT_DIR = "cropped/dirty"
VALID_DIR = "bin_classifier_project/annotated_dataset/validated_bins"
REJECTED_DIR = "bin_classifier_project/annotated_dataset/rejected_non_bins"

os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
index = 0

while index < len(files):
    filename = files[index]
    path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        index += 1
        continue

    cv2.imshow("Annotation : ← rejeter / → valider / q quitter", img)
    key = cv2.waitKey(0)

    if key == ord('q'):
        break
    elif key == 81:  # ←
        shutil.copy(path, os.path.join(REJECTED_DIR, filename))
        print(f"[REJETÉ] {filename}")
        index += 1
    elif key == 83:  # →
        shutil.copy(path, os.path.join(VALID_DIR, filename))
        print(f"[VALIDÉ] {filename}")
        index += 1
    else:
        print("Touche non reconnue")

cv2.destroyAllWindows()
