import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image
from tensorflow.keras.models import Model

# Carga del modelo
model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')

st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded_file = st.file_uploader("Sube una imagen de la piel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)  # usa use_container_width

    # Preprocesar imagen
    img_array = preprocess_image(image)

    # Predicción normal con modelo completo
    pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
    label = "Melanoma" if pred >= 0.5 else "No Melanoma"
    conf = pred if pred >= 0.5 else 1 - pred
    st.write(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Preparar modelo para Grad-CAM
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer("top_conv")


    gradcam_model = Model(
    inputs=base_model.input,        # entrada del base_model (EfficientNetB3)
    outputs=[last_conv_layer.output, model.output]
    )


    # Generar heatmap con gradcam_model
    heatmap = make_gradcam_heatmap(
        np.expand_dims(img_array, axis=0),
        gradcam_model,
        last_conv_layer_name=None  # porque ya se define en las salidas
    )
    gradcam_image = save_and_display_gradcam(image, heatmap)
    st.image(gradcam_image, caption="Grad-CAM", use_container_width=True)
