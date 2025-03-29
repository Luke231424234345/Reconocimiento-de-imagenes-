import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from mtcnn import MTCNN
from PIL import Image, ImageDraw, ImageFont

# 📌 RUTA DEL MODELO (AJÚSTALA SI ES NECESARIO)
MODEL_PATH = "modelo_Antirobos.keras"

# 📌 ETIQUETAS DEL MODELO (AJÚSTALAS SI ES NECESARIO)
LABELS = ["sin casco/tapabocas", "con casco", "con tapabocas", "otros"]

# 📌 CARGA DEL MODELO
if not hasattr(st.session_state, 'model'):
    try:
        st.session_state.model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo: {e}")
        st.stop()

# 📌 FUNCIÓN PARA DETECTAR ROSTROS Y CLASIFICARLOS
def detect_faces(image):
    detector = MTCNN()
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    
    if not faces:
        st.warning("⚠ No se detectaron rostros en la imagen.")
        return image

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        # Recortar y redimensionar la cara
        face_crop = img_rgb[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (128, 128))  # Ajustar al tamaño de entrada del modelo
        face_crop = np.expand_dims(face_crop / 255.0, axis=0)

        # Hacer la predicción
        model = st.session_state.model
        prediction = model.predict(face_crop)
        class_index = np.argmax(prediction)
        class_name = LABELS[class_index]
        confidence = prediction[0][class_index]

        # Dibujar rectángulo y etiqueta
        color = (0, 255, 0) if class_name == "otros" else (255, 0, 0)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)
        text = f"{class_name} - {confidence:.2f}"
        cv2.putText(img_rgb, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return Image.fromarray(img_rgb)

# 📌 INTERFAZ DE STREAMLIT
st.title("🔍 Detección de Intrusos con IA")

# 📌 OPCIONES DE DETECCIÓN
option = st.sidebar.selectbox("Selecciona una opción", ["Imagen", "Video en Tiempo Real"])

if option == "Imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen Original", use_container_width=True)
        processed_image = detect_faces(image)
        st.image(processed_image, caption="Detección de Rostros", use_container_width=True)

elif option == "Video en Tiempo Real":
    st.warning("⚠️ Función en desarrollo para transmisión en tiempo real en Streamlit.")
