import streamlit as st
import numpy as np
from PIL import Image
import keras as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Configuración de la página
st.set_page_config(page_title="Clasificador de Imágenes", layout="wide")

# Título y descripción
st.title("🖼️ Clasificador de Imágenes con MobileNetV2")
st.markdown("""
Sube una imagen y el modelo preentrenado **MobileNetV2** te dirá qué contiene.
*Ejemplo: perros, gatos, coches, etc.*
""")

# Cargar el modelo (se cachea para no cargarlo en cada ejecución)
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# Widget para subir la imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", width=300)

    # Preprocesar y predecir
    if st.button("Clasificar"):
        st.write("🔍 Analizando...")
        
        # Redimensionar y preprocesar
        image = image.resize((224, 224))  # Tamaño esperado por MobileNetV2
        image_array = np.array(image)
        image_array = preprocess_input(image_array)  # Normalización
        image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión batch

        # Hacer la predicción
        predictions = model.predict(image_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]  # Top 3 resultados

        # Mostrar resultados
        st.success("✅ Predicciones:")
        for i, (imagenet_id, label, prob) in enumerate(decoded_predictions):
            st.write(f"{i+1}. **{label}** (probabilidad: {prob*100:.2f}%)")

# Sidebar con información adicional
with st.sidebar:
    st.markdown("## ℹ️ Acerca de")
    st.markdown("""
    Esta app usa:
    - **Streamlit** para la interfaz web.
    - **MobileNetV2** (modelo preentrenado en ImageNet).
    - **TensorFlow/Keras** para el procesamiento.
    """)
    st.markdown("---")
    st.markdown("Código en [GitHub](https://github.com/tu-usuario/mi-app-streamlit)")
