import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.preprocessing import preprocess_image

# Carga el modelo completo
model = tf.keras.models.load_model('Modelos/Clasificacion_melanoma_V1_P2.keras')

# Extrae la base_model (EfficientNet)
base_model = model.layers[0]

# Forzar llamada para definir input/output
dummy_input = np.zeros((1, 300, 300, 3), dtype=np.float32)
_ = model(dummy_input)

# Obtener última capa convolucional de base_model
last_conv_layer = base_model.get_layer("top_conv")

# Crear modelo para GradCAM con dos salidas:
# salida última capa conv y salida final del modelo completo
gradcam_model = Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, model.layers[-1].output]
)

st.set_page_config(page_title="Detección de Melanoma", layout="centered")
st.title("🩺 Clasificación de Melanoma con Grad-CAM")

uploaded_file = st.file_uploader("Sube una imagen de la piel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesar imagen para el modelo
    img_array = preprocess_image(image)  # debe devolver array con shape (300, 300, 3)

    # Predecir con el modelo completo
    pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
    label = "Melanoma" if pred >= 0.5 else "No Melanoma"
    conf = pred if pred >= 0.5 else 1 - pred
    st.write(f"**Predicción:** {label} ({conf*100:.2f}%)")

    # Generar heatmap con el modelo gradcam_model
    heatmap = make_gradcam_heatmap(np.expand_dims(img_array, axis=0), gradcam_model)

    # Superponer heatmap sobre la imagen original
    gradcam_image = save_and_display_gradcam(image, heatmap)
    st.image(gradcam_image, caption="Grad-CAM", use_container_width=True)
