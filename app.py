import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from mtcnn import MTCNN
from PIL import Image
import os

# 📌 MONTA GOOGLE DRIVE SI ESTÁS EN GOOGLE COLAB
if "google.colab" in str(getattr(__import__("sys"), "modules", {})):
    from google.colab import drive
    drive.mount('/content/drive')
    DATASET_PATH = "/content/drive/My Drive/dataset_faces"
    MODEL_PATH = "/content/drive/My Drive/modelo_Antirobos.keras"
else:
    DATASET_PATH = "dataset_faces"  # Ruta local del dataset
    MODEL_PATH = "modelo_Antirobos.keras"  # Ruta local del modelo

# 📌 FUNCIÓN PARA CARGAR LAS ETIQUETAS (NOMBRES DE CLASES)
def load_labels(dataset_path):
    if not os.path.exists(dataset_path):
        st.error(f"❌ No se encontró el dataset en {dataset_path}. Asegúrate de que está montado correctamente.")
        return []
    
    labels = sorted([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])
    
    if not labels:
        st.warning("⚠️ El dataset está vacío o mal estructurado. Debe contener carpetas con nombres de clases.")
    
    return labels

# 📌 FUNCIÓN PARA DETECTAR ROSTROS Y CLASIFICARLOS
def detect_faces(image, model, labels):
    detector = MTCNN()
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_crop = img_rgb[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop = np.expand_dims(face_crop / 255.0, axis=0)
        
        prediction = model.predict(face_crop)
        class_index = np.argmax(prediction)
        class_name = labels[class_index]
        confidence = prediction[0][class_index]
        
        color = (0, 255, 0) if class_name == "otros" else (255, 0, 0)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_rgb, f"{class_name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return Image.fromarray(img_rgb)

# 📌 INTERFAZ DE STREAMLIT
st.title("🔍 Detección de Intrusos con IA")

# 📌 CARGA DEL MODELO Y EL DATASET
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error(f"❌ No se encontró el modelo en {MODEL_PATH}. Asegúrate de que está en la ubicación correcta.")
    st.stop()

labels = load_labels(DATASET_PATH)
if not labels:
    st.stop()  # Detiene la app si no hay etiquetas

# 📌 OPCIONES DE DETECCIÓN
option = st.sidebar.selectbox("Selecciona una opción", ["Imagen", "Video en Tiempo Real"])

if option == "Imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen Original", use_column_width=True)
        processed_image = detect_faces(image, model, labels)
        st.image(processed_image, caption="Detección de Rostros", use_column_width=True)

elif option == "Video en Tiempo Real":
    st.warning("⚠️ Función en desarrollo para transmisión en tiempo real en Streamlit.")
