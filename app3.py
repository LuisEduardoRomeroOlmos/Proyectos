import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import wikipedia
import time

# Configuración de la página
st.set_page_config(page_title="Clasificador de Flores (Oxford 102)", layout="wide")

# Título y descripción
st.title("🌸 Clasificador de Flores (Oxford 102) con Wikipedia")
st.markdown("""
Sube una imagen de una flor y el modelo predecirá a cuál de las 102 especies del dataset Oxford Flowers pertenece,
mostrando además información relevante de Wikipedia.
""")

# Configurar Wikipedia en español
wikipedia.set_lang("es")

# Función para cargar y preprocesar imagen
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
    return load_model("Modelos/Clasificacion_flores.keras")  # Asegúrate de tener este archivo

modelo = cargar_modelo()

# Nombres completos de las 102 clases (Oxford Flowers 102)
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

# Función mejorada para obtener información de Wikipedia
def obtener_info_wikipedia(flor_nombre):
    try:
        # Primera búsqueda directa
        page = wikipedia.page(flor_nombre, auto_suggest=True)
        return {
            "titulo": page.title,
            "resumen": page.summary[:1000] + "...",
            "url": page.url,
            "imagenes": [img for img in page.images if img.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
        }
    except wikipedia.exceptions.DisambiguationError as e:
        # Manejo de páginas de desambiguación
        try:
            page = wikipedia.page(e.options[0])
            return {
                "titulo": page.title,
                "resumen": f"(Resultado de búsqueda similar)\n\n{page.summary[:1000]}...",
                "url": page.url,
                "imagenes": [img for img in page.images if img.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
            }
        except:
            return None
    except wikipedia.exceptions.PageError:
        # Búsqueda con nombres alternativos
        nombres_alternativos = {
            "pink primrose": "Primula rosea",
            "rose": "Rosa",
            "tulip": "Tulipa",
            "sunflower": "Helianthus",
            "daffodil": "Narcissus",
            "water lily": "Nymphaeaceae",
            "lotus": "Nelumbo nucifera"
        }
        try:
            nombre_alternativo = nombres_alternativos.get(flor_nombre.lower(), flor_nombre)
            page = wikipedia.page(nombre_alternativo, auto_suggest=True)
            return {
                "titulo": page.title,
                "resumen": f"(Resultado para {nombre_alternativo})\n\n{page.summary[:1000]}...",
                "url": page.url,
                "imagenes": [img for img in page.images if img.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
            }
        except:
            return None
    except Exception as e:
        st.error(f"Error al consultar Wikipedia: {str(e)}")
        return None

# Interfaz principal
uploaded_file = st.file_uploader("Elige una imagen de flor...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption="Imagen subida", width=300)
    
    if st.button("Clasificar"):
        with st.spinner("🔍 Analizando la imagen y buscando información..."):
            # Procesamiento de imagen
            img_array = cargar_preprocesar_imagen_desde_bytes(uploaded_file)
            
            # Predicción
            predictions = modelo.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            class_name = nombres_clases.get(predicted_class, f"Clase {predicted_class}")
            
            # Obtener info de Wikipedia
            wiki_info = obtener_info_wikipedia(class_name)
        
        # Mostrar resultados
        with col2:
            st.success(f"🌺 **Predicción:** {class_name}")
            st.write(f"📊 **Confianza:** {confidence:.2f}%")
            
            # Top 5 predicciones
            st.markdown("### 🔝 Top 5 predicciones:")
            top5 = np.argsort(predictions[0])[::-1][:5]
            for i, idx in enumerate(top5):
                st.write(f"{i+1}. {nombres_clases.get(idx, f'Clase {idx}')}: {predictions[0][idx]*100:.2f}%")
            
            # Información de Wikipedia
            if wiki_info:
                st.markdown(f"### 📚 {wiki_info['titulo']}")
                st.info(wiki_info["resumen"])
                
                if wiki_info["imagenes"]:
                    try:
                        st.image(wiki_info["imagenes"][0], width=300)
                    except:
                        pass
                
                st.markdown(f"[📖 Artículo completo]({wiki_info['url']})")
            else:
                st.warning("No se encontró información en Wikipedia para esta flor.")

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ Acerca de")
    st.markdown("""
    **Tecnologías:**
    - TensorFlow/Keras
    - Streamlit
    - Wikipedia API
    
    **Dataset:** Oxford Flowers 102
    """)
    st.markdown("---")
    st.markdown("👨‍💻 [Código en GitHub](https://github.com/LuisEduardoRomeroOlmos/Proyectos)")
    st.markdown("🌐 [Dataset original](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)")