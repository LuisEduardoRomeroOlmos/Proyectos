import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image # Usamos el alias existente

# Configuración de la página
st.set_page_config(page_title="Clasificador Gato vs Perro", layout="wide")

# Título y descripción
st.title("🐶🐱 Clasificador Gato vs Perro")
st.markdown("""
Sube una imagen de tu mascota y el modelo predecirá si es un **gato** o un **perro**.
""")

# Función para cargar y preprocesar imagen desde un archivo subido
def cargar_preprocesar_imagen_desde_bytes(file, tamaño=(128, 128)):
    # Abre la imagen y la convierte a RGB (por si es PNG con transparencia)
    img = Image.open(file).convert("RGB")
    # Redimensiona la imagen al tamaño esperado por el modelo (128x128)
    img = img.resize(tamaño)
    # Convierte la imagen a un array de NumPy
    img_array = image.img_to_array(img)
    # Añade una dimensión para representar el batch size (esperado por Keras)
    img_array = np.expand_dims(img_array, axis=0)
    # Normaliza los píxeles al rango [0, 1]
    img_array = img_array / 255.0
    return img_array

# Cargar el modelo fine-tuned con corrección para el ValueError
@st.cache_resource
def cargar_modelo():
    # Se añade compile=False para evitar que Keras intente reconstruir
    # las funciones de pérdida/métricas, lo que causa el ValueError
    # si hay objetos personalizados faltantes.
    return load_model("mobile_fine_tuning.keras", compile=False)

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo. Asegúrate de que 'mobile_fine_tuning.keras' existe y no requiere objetos personalizados.")
    st.exception(e)
    # Detenemos la ejecución si el modelo no se carga
    st.stop()


# Widget para subir la imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Crear columnas para imagen y resultado
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(uploaded_file, caption="Imagen subida", use_container_width=True)

    with col_result:
        # Usamos un contenedor para mostrar el resultado después de la predicción
        result_container = st.container()

        if st.button("Clasificar"):
            result_container.write("🔍 Analizando ...")

            # Preprocesar la imagen
            img_array = cargar_preprocesar_imagen_desde_bytes(uploaded_file)

            # Hacer predicción
            # La predicción devuelve un array, e.g., [[0.98]]
            prediction = modelo.predict(img_array, verbose=0) 

            # Interpretar resultado: la salida es una probabilidad de ser perro (clase 1)
            # asumimos que Gato es 0 y Perro es 1.
            score = prediction[0][0]
            
            if score > 0.5:
                clase = "🐶 Es un **perro**"
                probabilidad = score * 100
            else:
                clase = "🐱 Es un **gato**"
                probabilidad = (1 - score) * 100

            # Mostrar resultado
            result_container.empty() # Limpiamos el mensaje "Analizando..."
            result_container.success(clase)
            result_container.write(f"🔢 Confianza del modelo: **{probabilidad:.2f}%**")

# Sidebar con info adicional
with st.sidebar:
    st.markdown("## ℹ️ Acerca de")
    st.markdown("""
    Esta app usa:
    - **Streamlit** para la interfaz.
    - **MobileNetV2 Fine-tuned** (asumiendo que este es el modelo).
    - **TensorFlow/Keras** para la predicción y Python 3.10.
    """)
    st.markdown("---")
    st.markdown("Código en [GitHub](https://github.com/LuisEduardoRomeroOlmos/Proyectos/tree/main)")
