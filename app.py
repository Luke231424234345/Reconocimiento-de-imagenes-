import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from mtcnn import MTCNN
from PIL import Image
import os

# 📌 RUTA DEL MODELO (AJÚSTALA SI ES NECESARIO)
MODEL_PATH = "modelo_Antirobos.keras"

# 📌 ETIQUETAS DEFINIDAS MANUALMENTE (AJÚSTALAS SEGÚN TU MODELO)
LABELS = ["sin casco/tapabocas", "con casco", "con tapabocas", "otros"]

# 📌 FUNCIÓN PARA DETECTAR ROSTROS Y CLASIFICARLOS
def detect_faces(image, model):
    detector = MTCNN()
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)  # Convertir imagen a RGB
    
    # Detectar rostros
    faces = detector.detect_faces(img_rgb)
    
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_crop = img_rgb[y:y+h, x:x+w]
        
        # **Ajustar el tamaño de la imagen del rostro a 128x128, que es lo que espera el modelo**
        face_crop = cv2.resize(face_crop, (128, 128))  # Cambié de 224x224 a 128x128
        face_crop = np.expand_dims(face_crop / 255.0, axis=0)
        
        # Predicción del modelo
        prediction = model.predict(face_crop)
        class_index = np.argmax(prediction)
        class_name = LABELS[class_index]
        confidence = prediction[0][class_index]
        
        # Establecer color y etiqueta
        color = (0, 255, 0) if class_name == "otros" else (255, 0, 0)
        
        # Dibuja el rectángulo alrededor del rostro
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)
        
        # Agregar texto con la clase predicha
        cv2.putText(img_rgb, f"{class_name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convertir la imagen modificada de nuevo a formato PIL para Streamlit
    return Image.fromarray(img_rgb)

# 📌 INTERFAZ DE STREAMLIT
st.title("🔍 Detección de Intrusos con IA")

# 📌 CARGA DEL MODELO
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error(f"❌ No se encontró el modelo en {MODEL_PATH}. Asegúrate de que está en la ubicación correcta.")
    st.stop()

# 📌 OPCIONES DE DETECCIÓN
option = st.sidebar.selectbox("Selecciona una opción", ["Imagen", "Video en Tiempo Real"])

if option == "Imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen Original", use_container_width=True)  # Cambiado a use_container_width
        # Procesar la imagen y mostrar la detección de rostros con predicción
        processed_image = detect_faces(image, model)
        st.image(processed_image, caption="Detección de Rostros", use_container_width=True)  # Cambiado a use_container_width

elif option == "Video en Tiempo Real":
    st.warning("⚠️ Función en desarrollo para transmisión en tiempo real en Streamlit.")
