import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions

# Configuración de la página
st.set_page_config(page_title="Clasificador de Imágenes", layout="wide")

# Título y descripción
st.title("🖼️ Clasificador de Imágenes con EfficientNetB0")
st.markdown("""
Sube una imagen y el modelo preentrenado **EfficientNetB0** te dirá qué contiene.
*Ejemplo: perros, gatos, coches, flores, etc.*
""")

# Cargar el modelo
@st.cache_resource
def load_model():
    return EfficientNetB0(weights="imagenet")

model = load_model()

# Widget para subir la imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Clasificar cuando se presione el botón
    if st.button("Clasificar"):
        st.write("🔍 Analizando...")

        # Preprocesar imagen
        image = image.resize((224, 224))  # Tamaño para EfficientNetB0
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        # Predicción
        predictions = model.predict(image_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

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
    - **EfficientNetB0** (modelo preentrenado).
    - **TensorFlow/Keras** para el procesamiento.
    """)
    st.markdown("---")
    st.markdown("👨‍💻 [Código en GitHub](https://github.com/LuisEduardoRomeroOlmos/Proyectos/tree/main)")
