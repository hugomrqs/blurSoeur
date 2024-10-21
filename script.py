
import cv2
import os
import dlib

detector = dlib.get_frontal_face_detector()
line_width = 3
color_green = (0, 255, 0)  # Define the color green

# Chemin vers le dossier contenant les images
images_folder = './soeurImg'
# Chemin vers le dossier où sauvegarder les images traitées
processed_folder = './soeurProc'


# Liste de toutes les images dans le dossier
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(image_files)

for image_file in image_files:
    print(image_file + " en cours de traitement")
    # Chemin complet de l'image actuelle
    image_path = os.path.join(images_folder, image_file)
    # Lire l'image
    img = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Détecter les visages dans l'image
    dets = detector(rgb_image)
    print("Détecter les visages dans l'image")
        # Parcourir tous les visages détectés
    for det in dets:
            # Assurer que les coordonnées sont dans les limites de l'image
        top, bottom, left, right = max(0, det.top()), min(img.shape[0], det.bottom()), max(0, det.left()), min(img.shape[1], det.right())
            
            # Extraire la région du visage
        face = img[top:bottom, left:right]
        print("Extraire la région du visage")
            # Appliquer un flou à la région du visage
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            # Remplacer la région du visage par la version floue
        img[top:bottom, left:right] = blurred_face
            
            # Dessiner un cercle autour de la région du visage
    
        processed_image_path = os.path.join(processed_folder, image_file)
        cv2.imwrite(processed_image_path, img)
        print("Enregistrement de l'image traitée")

cv2.destroyAllWindows()
 
