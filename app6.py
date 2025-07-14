import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# 1) Carga tu modelo completo (Sequential con EfficientNet anidado)
pred_model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')

# 2) Extrae el base_model y localiza la última capa convolucional
base_model = pred_model.layers[0]
last_conv   = base_model.get_layer("top_conv")

# 3) Crea dos submodelos: uno para las activaciones conv, otro para la predicción final
conv_model = Model(inputs=base_model.input, outputs=last_conv.output)
# pred_model ya apunta al modelo completo sequencial

# 4) Forzamos un forward pass para inicializar shapes internos
dummy = np.zeros((1, 300, 300, 3), dtype=np.float32)
_ = conv_model(dummy)
_ = pred_model(dummy)

# Streamlit UI
st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded_file = st.file_uploader("Sube una imagen de la piel", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Mostrar upload
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesamiento
    img_arr = preprocess_image(image)            # → (300,300,3)
    batch   = np.expand_dims(img_arr, axis=0)    # → (1,300,300,3)

    # Predicción de clase
    pred = pred_model.predict(batch)[0][0]
    label = "Melanoma" if pred >= 0.5 else "No Melanoma"
    conf  = pred if pred >= 0.5 else 1 - pred
    st.write(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Generar Grad-CAM
    heatmap = make_gradcam_heatmap(batch, conv_model, pred_model)
    gradcam_img = save_and_display_gradcam(image, heatmap)
    st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)
