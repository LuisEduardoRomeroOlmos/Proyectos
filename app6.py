import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# 1) Carga tu modelo completo
full_model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')

# 2) Separa la base EfficientNet y localiza 'top_conv'
base_model      = full_model.layers[0]
last_conv_layer = base_model.get_layer("top_conv")

# 3) Forzamos un forward pass para definir shapes
dummy = np.zeros((1,300,300,3), dtype=np.float32)
_ = full_model(dummy)

# 4) Reconstruye manualmente el head (las capas 1 en adelante) sobre base_model.input
inputs = base_model.input                # Tensor de entrada (None,300,300,3)
x      = base_model(inputs)              # Sale de EfficientNet
conv_outputs = last_conv_layer.output    # Activaciones de top_conv

# Aplica las capas restantes del Sequential original
for layer in full_model.layers[1:]:
    x = layer(x)

# 'x' es ahora la salida final del modelo completo, pero dentro del mismo grafo
final_output = x

# 5) Modelo funcional que expone CONcurrentemente:
#    [activaciones de top_conv, salida final]
gradcam_model = Model(inputs=inputs, outputs=[conv_outputs, final_output])

# ——— Streamlit UI ———
st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded = st.file_uploader("Sube una imagen de la piel", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    # Preprocesa
    arr   = preprocess_image(img)          # → (300,300,3)
    batch = np.expand_dims(arr, axis=0)    # → (1,300,300,3)

    # Predicción normal
    pred  = full_model.predict(batch)[0][0]
    label = "Melanoma" if pred>=0.5 else "No Melanoma"
    conf  = pred if pred>=0.5 else 1-pred
    st.write(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Grad-CAM
    heatmap = make_gradcam_heatmap(batch, gradcam_model)
    gcam    = save_and_display_gradcam(img, heatmap)
    st.image(gcam, caption="Grad-CAM", use_container_width=True)


