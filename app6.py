import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# Cargar el modelo completo (Sequential que incluye EfficientNetB3)
model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')

# Obtener el modelo base (EfficientNetB3)
base_model = model.layers[0]

# Ejecutar un paso dummy para asegurar que todas las capas estén construidas
_ = model(np.zeros((1, 300, 300, 3), dtype=np.float32))

# Obtener la última capa convolucional para Grad-CAM
last_conv_layer = base_model.get_layer("top_conv")

# Crear modelo funcional para Grad-CAM con input del modelo base
gradcam_model = Model(
    inputs=model.layers[0].input,
    outputs=[last_conv_layer.output, model.layers[-1].output]
)

# Configuración de la app
st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded_file = st.file_uploader("Sube una imagen de la piel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesamiento de la imagen
    img_array = preprocess_image(image)  # Debe devolver un array (300, 300, 3) normalizado
    img_batch = np.expand_dims(img_array, axis=0)

    # Predicción
    pred = model.predict(img_batch)[0][0]
    label = "Melanoma" if pred >= 0.5 else "No Melanoma"
    conf = pred if pred >= 0.5 else 1 - pred
    st.write(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(
    img_batch,
    gradcam_model
    )
    
    # Visualización Grad-CAM
    gradcam_image = save_and_display_gradcam(image, heatmap)
    st.image(gradcam_image, caption="Grad-CAM", use_container_width=True)
