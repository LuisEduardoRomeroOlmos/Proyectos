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
        0: "prímula rosa",
        1: "orquídea de bolsillo de hojas duras",
        2: "campanas de Canterbury",
        3: "guisante de olor",
        4: "caléndula inglesa",
        5: "lirio tigre",
        6: "orquídea luna",
        7: "ave del paraíso",
        8: "acónito",
        9: "cardo globo",
        10: "boca de dragón",
        11: "fárfara",
        12: "protea rey",
        13: "cardo lanceolado",
        14: "lirio amarillo",
        15: "trollius",
        16: "equinácea púrpura",
        17: "lirio peruano",
        18: "musgo bola",
        19: "lirio de arum gigante blanco",
        20: "lirio de fuego",
        21: "flor de alfiler",
        22: "fritilaria",
        23: "jengibre rojo",
        24: "jacinto de uva",
        25: "amapola silvestre",
        26: "plumas del príncipe de Gales",
        27: "genciana sin tallo",
        28: "alcachofa",
        29: "clavel del poeta",
        30: "clavel",
        31: "flox de jardín",
        32: "amor en la niebla",
        33: "aster mexicano",
        34: "erizo de mar alpino",
        35: "orquídea cattleya de labios rubí",
        36: "flor del cabo",
        37: "gran maestra",
        38: "tulipán siamés",
        39: "rosa de Cuaresma",
        40: "margarita barbeton",
        41: "narciso",
        42: "gladiolo",
        43: "flor de pascua",
        44: "bolero azul profundo",
        45: "alhelí",
        46: "caléndula",
        47: "ranúnculo",
        48: "margarita común",
        49: "diente de león común",
        50: "petunia",
        51: "pensamiento silvestre",
        52: "prímula",
        53: "girasol",
        54: "geranio",
        55: "obispo de Llandaff",
        56: "gaura",
        57: "geranio",
        58: "dalia naranja",
        59: "dalia rosa-amarilla",
        60: "cautleya spicata",
        61: "anémona japonesa",
        62: "susana de ojos negros",
        63: "arbusto plateado",
        64: "amapola californiana",
        65: "osteospermum",
        66: "crocus de primavera",
        67: "iris",
        68: "anémona",
        69: "amapola arbórea",
        70: "gazania",
        71: "azalea",
        72: "nenúfar",
        73: "rosa",
        74: "manzana espinosa",
        75: "gloria de la mañana",
        76: "flor de la pasión",
        77: "loto",
        78: "lirio sapo",
        79: "anturio",
        80: "frangipani",
        81: "clemátide",
        82: "hibisco",
        83: "aguileña",
        84: "rosa del desierto",
        85: "malva arbórea",
        86: "magnolia",
        87: "ciclamen",
        88: "berro de agua",
        89: "caña de lirio",
        90: "amarilis",
        91: "bálsamo de abeja",
        92: "flor globo",
        93: "margarita barbeton",
        94: "digital",
        95: "buganvilla",
        96: "camelia",
        97: "malva",
        98: "petunia mexicana",
        99: "bromelia",
        100: "flor manta",
        101: "enredadera trompeta"
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
    # Dividir en dos columnas principales con espacio entre ellas
    col_upload, col_results = st.columns([1, 2], gap="large")
    
    with col_upload:
        # Contenedor fijo para la imagen subida
        upload_container = st.container(border=True)
        with upload_container:
            st.image(uploaded_file, 
                   caption="Imagen subida", 
                   width=300,
                   use_container_width=True)  # ¡Parámetro actualizado aquí!
            st.markdown("---")
            classify_btn = st.button("🪄 Clasificar", type="primary", use_container_width=True)
    
    # Columna de resultados
    with col_results:
        if classify_btn:  # Solo se ejecuta al presionar el botón
            with st.spinner("Analizando la imagen..."):
                # Procesamiento de la imagen
                img_array = cargar_preprocesar_imagen_desde_bytes(uploaded_file)
                predictions = modelo.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                class_name = nombres_clases.get(predicted_class, f"Clase {predicted_class}")
                wiki_info = obtener_info_wikipedia(class_name)
            
            # Contenedor de resultados con pestañas
            result_tabs = st.tabs(["🔮 Predicción", "📊 Detalles", "🌍 Wikipedia"])
            
            with result_tabs[0]:  # Pestaña de predicción principal
                st.success(f"**{class_name}**")
                st.metric("Confianza", f"{confidence:.2f}%")
                st.image(uploaded_file, use_container_width=True)
            
            with result_tabs[1]:  # Pestaña de detalles técnicos
                st.subheader("Top 5 predicciones")
                top5 = np.argsort(predictions[0])[::-1][:5]
                for i, idx in enumerate(top5):
                    st.progress(
                        float(predictions[0][idx]),
                        text=f"{i+1}. {nombres_clases.get(idx)}: {predictions[0][idx]*100:.2f}%"
                    )
            
            with result_tabs[2]:  # Pestaña de Wikipedia
                if wiki_info:
                    st.subheader(wiki_info['titulo'])
                    st.write(wiki_info["resumen"])
                    if wiki_info["imagenes"]:
                        st.image(wiki_info["imagenes"][0], use_container_width=True)
                    st.page_link(wiki_info['url'], label="📖 Ver artículo completo")
                else:
                    st.warning("Información no encontrada en Wikipedia")
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