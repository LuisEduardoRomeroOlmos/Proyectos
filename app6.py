import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# 1) Carga tu modelo Sequential (que tiene EfficientNet como primera capa)
model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')

# 2) Extrae el base_model (EfficientNet)
base_model = model.layers[0]

# 3) Forzamos un forward pass para que Keras defina internamente inputs/outputs
dummy = np.zeros((1, 300, 300, 3), dtype=np.float32)
_ = model(dummy)

# 4) Localiza la última capa convolucional dentro de base_model
last_conv = base_model.get_layer("top_conv")

# 5) Construye el modelo funcional para Grad‑CAM **usando model(base_model.input)**
#    - inputs: mismo tensor de entrada que espera base_model
#    - outputs: [salida de last_conv, salida final de model(base_model.input)]
gradcam_model = Model(
    inputs=base_model.input,
    outputs=[
        last_conv.output,
        model(base_model.input)   # aquí conectamos la entrada al modelo completo
    ]
)

# --- Streamlit UI ---
st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded_file = st.file_uploader("Sube una imagen de la piel", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesamiento igual al de entrenamiento
    img_arr = preprocess_image(image)  # → (300, 300, 3)
    batch = np.expand_dims(img_arr, axis=0)  # → (1, 300, 300, 3)

    # Predicción normal
    pred = model.predict(batch)[0][0]
    label = "Melanoma" if pred >= 0.5 else "No Melanoma"
    conf  = pred if pred >= 0.5 else 1 - pred
    st.write(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # 🔥 Grad-CAM
    heatmap = make_gradcam_heatmap(batch, gradcam_model)
    gradcam_img = save_and_display_gradcam(image, heatmap)
    st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)
