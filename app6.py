import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# --- Cargar el modelo completo ---
model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')
base_model = model.layers[0]

# Forzar ejecución para construir el modelo (muy importante en Streamlit/Cloud)
dummy_input = np.zeros((1, 300, 300, 3), dtype=np.float32)
_ = model(dummy_input)  # Construye todos los outputs

# Ahora podemos acceder a las capas y sus outputs
last_conv_layer = base_model.get_layer("top_conv")

# Crear modelo Grad-CAM con entrada del modelo base y salidas deseadas
gradcam_model = Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, model.output]
)

# --- Interfaz Streamlit ---
st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded_file = st.file_uploader("Sube una imagen de la piel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesar imagen
    img_array = preprocess_image(image)
    input_tensor = np.expand_dims(img_array, axis=0)

    # Predicción
    pred = model.predict(input_tensor)[0][0]
    label = "Melanoma" if pred >= 0.5 else "No Melanoma"
    conf = pred if pred >= 0.5 else 1 - pred
    st.markdown(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Generar heatmap Grad-CAM
    heatmap = make_gradcam_heatmap(input_tensor, gradcam_model)

    # Mostrar imagen con superposición de Grad-CAM
    gradcam_image = save_and_display_gradcam(image, heatmap)
    st.image(gradcam_image, caption="Grad-CAM", use_container_width=True)
