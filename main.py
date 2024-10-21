import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image
import zipfile
import os
from io import BytesIO

# Initialisation du détecteur de visages
detector = dlib.get_frontal_face_detector()

def blur_faces(image):
    # Convertir l'image PIL en tableau numpy en format RGB
    img = np.array(image.convert('RGB'))
    rgb_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Détecter les visages dans l'image
    faces = detector(rgb_image)
    
    # Flouter les visages détectés
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_region = img[y:y+h, x:x+w]
        face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
        img[y:y+h, x:x+w] = face_region
    
    return img

def process_and_zip_images(uploaded_files):
    # Créer un fichier ZIP en mémoire
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for uploaded_file in uploaded_files:
            # Lire l'image téléchargée
            image = Image.open(uploaded_file)
            
            # Flouter les visages dans l'image
            blurred_image = blur_faces(image)
            
            # Convertir l'image floutée de BGR à RGB avant de la convertir en image PIL
            blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
            blurred_image_pil = Image.fromarray(blurred_image_rgb)
            
            # Sauvegarder l'image floutée dans le fichier ZIP
            img_byte_arr = BytesIO()
            blurred_image_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            zip_file.writestr(uploaded_file.name, img_byte_arr)
    
    zip_buffer.seek(0)
    return zip_buffer

st.title("Démo de Floutage de Visages")

uploaded_files = st.file_uploader("Glissez et déposez des images ici", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    zip_buffer = process_and_zip_images(uploaded_files)
    
    # Fournir un lien pour télécharger le fichier ZIP
    st.download_button(
        label="Télécharger toutes les images floutées",
        data=zip_buffer,
        file_name="images_floutees.zip",
        mime="application/zip"
    )