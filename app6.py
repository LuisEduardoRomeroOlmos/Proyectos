import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# --- Cargar el modelo completo (Sequential con EfficientNet anidado) ---
model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')
base_model = model.layers[0]  # EfficientNetB3 anidado

# Obtener última capa convolucional de EfficientNet para Grad-CAM
last_conv_layer = base_model.get_layer("top_conv")

# Crear modelo funcional para Grad-CAM (input de base_model y output doble: última conv + predicción final)
gradcam_model = Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, model.output]
)

# ⚠️ Forzar inicialización del modelo para evitar errores de ejecución en Streamlit
dummy_input = np.zeros((1, 300, 300, 3), dtype=np.float32)
_ = gradcam_model(dummy_input)

# --- Interfaz Streamlit ---
st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded_file = st.file_uploader("Sube una imagen de la piel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesar imagen
    img_array = preprocess_image(image)  # output esperado: (300, 300, 3)
    input_tensor = np.expand_dims(img_array, axis=0)  # shape: (1, 300, 300, 3)

    # Predicción
    pred = model.predict(input_tensor)[0][0]
    label = "Melanoma" if pred >= 0.5 else "No Melanoma"
    conf = pred if pred >= 0.5 else 1 - pred
    st.markdown(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Heatmap con Grad-CAM
    heatmap = make_gradcam_heatmap(
        input_tensor,
        gradcam_model
        # No se necesita last_conv_layer_name si ya está en el output del modelo
    )

    # Superponer heatmap sobre la imagen original
    gradcam_image = save_and_display_gradcam(image, heatmap)
    st.image(gradcam_image, caption="Grad-CAM", use_container_width=True)
