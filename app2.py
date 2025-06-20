import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Configuración de la página
st.set_page_config(page_title="Clasificador Gato vs Perro", layout="wide")

# Título y descripción
st.title("🐶🐱 Clasificador Gato vs Perro")
st.markdown("""
Sube una imagen de tu mascota y el modelo predecirá si es un **gato** o un **perro**.
""")

# Función para cargar y preprocesar imagen desde un archivo subido
def cargar_preprocesar_imagen_desde_bytes(file, tamaño=(128, 128)):
    img = Image.open(file).convert("RGB")
    img = img.resize(tamaño)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Cargar el modelo fine-tuned
@st.cache_resource
def cargar_modelo():
    return load_model("mobile_fine_tuning.keras")

modelo = cargar_modelo()

# Widget para subir la imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Crear columnas para imagen y resultado
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(uploaded_file, caption="Imagen subida", use_container_width=True)

    with col_result:
        if st.button("Clasificar"):
            st.write("🔍 Clasificando con modelo fine-tuned...")

            # Preprocesar la imagen
            img_array = cargar_preprocesar_imagen_desde_bytes(uploaded_file)

            # Hacer predicción
            prediction = modelo.predict(img_array)

            # Interpretar resultado
            clase = "🐶 Es un **perro**" if prediction[0][0] > 0.5 else "🐱 Es un **gato**"
            probabilidad = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100

            # Mostrar resultado
            st.success(clase)
            st.write(f"🔢 Confianza del modelo: **{probabilidad:.2f}%**")

# Sidebar con info adicional
with st.sidebar:
    st.markdown("## ℹ️ Acerca de")
    st.markdown("""
    Esta app usa:
    - **Streamlit** para la interfaz.
    - **MobileNetV2 Fine-tuned** .
    - **TensorFlow/Keras** para la predicción y Python 3.10.
    """)
    st.markdown("---")
    st.markdown("Código en [GitHub](https://github.com/LuisEduardoRomeroOlmos/Proyectos/tree/main)")
