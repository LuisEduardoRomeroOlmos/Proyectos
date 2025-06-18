import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Configuración de la página
st.set_page_config(page_title="Clasificador de Flores ", layout="wide")

# Título y descripción
st.title("🌸 Clasificador de Flores ")
st.markdown("""
Sube una imagen de una flor y el modelo predecirá a cuál de las 102 especies del dataset Oxford Flowers pertenece.
""")

# Función para cargar y preprocesar imagen desde un archivo subido
def cargar_preprocesar_imagen_desde_bytes(file, tamaño=(224, 224)):
    img = Image.open(file).convert("RGB")
    img = img.resize(tamaño)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return load_model("Modelos/Clasificacion_flores.keras")  # Asegúrate de tener el modelo correcto

modelo = cargar_modelo()

# Cargar los nombres de las clases (debes tener un archivo con los nombres)
# Esto es un ejemplo, necesitarás adaptarlo a tu estructura real
@st.cache_resource
def cargar_nombres_clases():
    return {
        0: "pink primrose",
        1: "hard-leaved pocket orchid",
        2: "canterbury bells",
        3: "sweet pea",
        4: "english marigold",
        5: "tiger lily",
        6: "moon orchid",
        7: "bird of paradise",
        8: "monkshood",
        9: "globe thistle",
        10: "snapdragon",
        11: "colt's foot",
        12: "king protea",
        13: "spear thistle",
        14: "yellow iris",
        15: "globe-flower",
        16: "purple coneflower",
        17: "peruvian lily",
        18: "ball moss",
        19: "giant white arum lily",
        20: "fire lily",
        21: "pincushion flower",
        22: "fritillary",
        23: "red ginger",
        24: "grape hyacinth",
        25: "corn poppy",
        26: "prince of wales feathers",
        27: "stemless gentian",
        28: "artichoke",
        29: "sweet william",
        30: "carnation",
        31: "garden phlox",
        32: "love in the mist",
        33: "mexican aster",
        34: "alpine sea holly",
        35: "ruby-lipped cattleya",
        36: "cape flower",
        37: "great masterwort",
        38: "siam tulip",
        39: "lenten rose",
        40: "barbeton daisy",
        41: "daffodil",
        42: "sword lily",
        43: "poinsettia",
        44: "bolero deep blue",
        45: "wallflower",
        46: "marigold",
        47: "buttercup",
        48: "oxeye daisy",
        49: "common dandelion",
        50: "petunia",
        51: "wild pansy",
        52: "primula",
        53: "sunflower",
        54: "pelargonium",
        55: "bishop of llandaff",
        56: "gaura",
        57: "geranium",
        58: "orange dahlia",
        59: "pink-yellow dahlia",
        60: "cautleya spicata",
        61: "japanese anemone",
        62: "black-eyed susan",
        63: "silverbush",
        64: "californian poppy",
        65: "osteospermum",
        66: "spring crocus",
        67: "iris",
        68: "windflower",
        69: "tree poppy",
        70: "gazania",
        71: "azalea",
        72: "water lily",
        73: "rose",
        74: "thorn apple",
        75: "morning glory",
        76: "passion flower",
        77: "lotus",
        78: "toad lily",
        79: "anthurium",
        80: "frangipani",
        81: "clematis",
        82: "hibiscus",
        83: "columbine",
        84: "desert-rose",
        85: "tree mallow",
        86: "magnolia",
        87: "cyclamen",
        88: "watercress",
        89: "canna lily",
        90: "hippeastrum",
        91: "bee balm",
        92: "balloon flower",
        93: "barbeton daisy",
        94: "foxglove",
        95: "bougainvillea",
        96: "camellia",
        97: "mallow",
        98: "mexican petunia",
        99: "bromelia",
        100: "blanket flower",
        101: "trumpet creeper"
    }

nombres_clases = cargar_nombres_clases()

# Widget para subir la imagen
uploaded_file = st.file_uploader("Elige una imagen de flor...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, caption="Imagen subida", width=300)

    if st.button("Clasificar"):
        st.write("🔍 Clasificando...")

        # Preprocesar
        img_array = cargar_preprocesar_imagen_desde_bytes(uploaded_file)

        # Hacer predicción
        predictions = modelo.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        # Obtener nombre de la clase
        class_name = nombres_clases.get(predicted_class, f"Clase {predicted_class}")

        # Mostrar resultado
        st.success(f"🌺 Predicción: **{class_name}**")
        st.write(f"🔢 Confianza del modelo: **{confidence:.2f}%**")

        # Mostrar las 5 mejores predicciones
        st.markdown("### Top 5 predicciones:")
        top5_indices = np.argsort(predictions[0])[::-1][:5]
        
        for i, idx in enumerate(top5_indices):
            st.write(f"{i+1}. {nombres_clases.get(idx, f'Clase {idx}')}: {predictions[0][idx]*100:.2f}%")

# Sidebar con info adicional
with st.sidebar:
    st.markdown("## ℹ️ Acerca de")
    st.markdown("""
    Esta app usa:
    - Modelo entrenado en el dataset **Oxford Flowers 102**
    - **TensorFlow/Keras** para la predicción
    - **Creador** El Lalo
    """)
    st.markdown("---")
    st.markdown("Código en [GitHub](https://github.com/LuisEduardoRomeroOlmos/Proyectos/tree/main)")