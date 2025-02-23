import cv2
import mss
import numpy as np
import time
from ultralytics import YOLO

# Charger le modèle YOLOv8
# model = YOLO("last.pt")  # Utilise un modèle léger pour meilleures performances
model = YOLO("./models/best.pt")  # Utilise un modèle léger pour meilleures performances

# Définir une zone de capture (None pour plein écran)
screen_region = {"top": 200, "left": 100, "width": 1000, "height": 800}
# screen_region = None  # Ex: {"top": 100, "left": 100, "width": 800, "height": 600}

# Classes à détecter (laisser None pour tout détecter)
# allowed_classes = ["person", "car", "truck", "dog", "cat"]  # Exemple pour un jeu
allowed_classes = None  # Exemple pour un jeu
# allowed_classes = ["car", "truck"]  # Exemple pour un jeu
# allowed_classes = ["person", "head"]  # Exemple pour un jeu

print(model.names)  # Afficher les classes détectées

with mss.mss() as sct:
    prev_time = 0  # Pour calculer le FPS

    while True:
        # Capture d'écran
        screenshot = sct.grab(screen_region or sct.monitors[1])

        # Convertir en tableau numpy et en format BGR (OpenCV)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Détection avec YOLO
        results = model(img)

        # Dessiner les boîtes
        for result in results:
            if not result.boxes:
                continue  # Évite un crash si rien n'est détecté
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées
                conf = box.conf[0]  # Confiance
                cls = int(box.cls[0])  # Classe de l'objet
                class_name = model.names[cls]  # Nom de l'objet

                # Vérifier si l'objet est autorisé
                if allowed_classes and class_name not in allowed_classes:
                    continue

                # Dessiner un rectangle et le label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calcul du FPS
        new_time = time.time()
        fps = 1 / (new_time - prev_time) if prev_time else 0
        prev_time = new_time

        # Affichage du FPS
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Afficher l'image en direct
        cv2.imshow("Game Analysis", img)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
