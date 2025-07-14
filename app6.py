import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# ——— 1) Carga tu Sequential completo ———
full_model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')

# ——— 2) Extrae la EfficientNet base y la última capa conv ———
base_model     = full_model.layers[0]
last_conv_layer = base_model.get_layer("top_conv")

# ——— 3) Fuerza un forward pass para definir gráfo: ———
dummy = np.zeros((1, 300, 300, 3), dtype=np.float32)
_ = full_model(dummy)

# ——— 4) Crea el modelo funcional para Grad‑CAM ———
#    entrada = misma que espera EfficientNet
#    salidas = [activaciones de top_conv, salida final de full_model]
gradcam_model = Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        full_model(base_model.input)
    ]
)

# ——— 5) Interfaz Streamlit ———
st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded = st.file_uploader("Sube una imagen de la piel", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    # Preprocesar
    arr = preprocess_image(img)           # → (300,300,3)
    batch = np.expand_dims(arr, axis=0)   # → (1,300,300,3)

    # Predicción normal
    pred = full_model.predict(batch)[0][0]
    label = "Melanoma" if pred>=0.5 else "No Melanoma"
    conf  = pred if pred>=0.5 else 1-pred
    st.write(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Grad‑CAM 🔥
    heatmap = make_gradcam_heatmap(batch, gradcam_model)
    gcam_img = save_and_display_gradcam(img, heatmap)
    st.image(gcam_img, caption="Grad-CAM", use_container_width=True)

