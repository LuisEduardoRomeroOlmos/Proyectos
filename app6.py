import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# Carga el modelo completo (Sequential que incluye base_model)
model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')

# Extrae la base_model (EfficientNetB3)
base_model = model.layers[0]

# Obtén la última capa convolucional para Grad-CAM
last_conv_layer = base_model.get_layer("top_conv")

# Define un modelo funcional para Grad-CAM con dos salidas:
# salida última capa conv y salida final del modelo completo
inputs = base_model.input
last_conv_output = last_conv_layer.output
outputs = model(inputs)  # conecta entrada con salida del modelo completo

gradcam_model = Model(
    inputs=inputs,
    outputs=[last_conv_output, outputs]
)

st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded_file = st.file_uploader("Sube una imagen de la piel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesar imagen para el modelo
    img_array = preprocess_image(image)  # debe devolver un array con shape (H,W,3)

    # Predicción con modelo completo
    pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
    label = "Melanoma" if pred >= 0.5 else "No Melanoma"
    conf = pred if pred >= 0.5 else 1 - pred
    st.write(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Generar heatmap con gradcam_model
    heatmap = make_gradcam_heatmap(
        np.expand_dims(img_array, axis=0),
        gradcam_model,
        last_conv_layer_name=None  # ya definido en outputs
    )

    # Superponer heatmap sobre imagen original
    gradcam_image = save_and_display_gradcam(image, heatmap)
    st.image(gradcam_image, caption="Grad-CAM", use_container_width=True)


