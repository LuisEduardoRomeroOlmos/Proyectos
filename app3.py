import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import wikipedia

# Configuración de la página
st.set_page_config(page_title="Clasificador de Flores (Oxford 102)", layout="wide")

st.title("🌸 Clasificador de Flores 🌻")
st.markdown("""
Sube una imagen de una flor y el modelo predecirá a cuál de las 102 especies del dataset Oxford Flowers pertenece,
mostrando además información relevante de Wikipedia.
""")

wikipedia.set_lang("es")

# --- Funciones ---
def cargar_preprocesar_imagen_desde_bytes(file, tamaño=(224, 224)):
    img = Image.open(file).convert("RGB")
    img = img.resize(tamaño)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@st.cache_resource
def cargar_modelo():
    return load_model("Modelos/Clasificacion_flores.keras")

modelo = cargar_modelo()

@st.cache_data
def cargar_nombres_clases_oficial():
    return {
        0: "prímula rosada",               # carpeta 0
        1: "orquídea de hoja dura",        # carpeta 1
        2: "boca de dragón",               # carpeta 10
        3: "enredadera trompeta",          # carpeta 100
        4: "lirio mora",                   # carpeta 101
        5: "pata de caballo",              # carpeta 11
        6: "protea real",                  # carpeta 12
        7: "cardo lanudo",                 # carpeta 13
        8: "iris amarillo",                # carpeta 14
        9: "flor globo",                  # carpeta 15
        10: "cono púrpura",                # carpeta 16
        11: "lirio peruano",               # carpeta 17
        12: "flor globo",                  # carpeta 18
        13: "lirio arum blanco gigante",  # carpeta 19
        14: "campanillas de Canterbury",  # carpeta 2
        15: "lirio de fuego",              # carpeta 20
        16: "flor de cojín",               # carpeta 21
        17: "fritilaria",                  # carpeta 22
        18: "jengibre rojo",               # carpeta 23
        19: "jacinto de uva",              # carpeta 24
        20: "amapola silvestre",           # carpeta 25
        21: "pluma del príncipe de Gales",# carpeta 26
        22: "genciana sin tallo",          # carpeta 27
        23: "alcachofa",                   # carpeta 28
        24: "clavel del poeta",            # carpeta 29
        25: "guisante de olor",            # carpeta 3
        26: "clavel",                     # carpeta 30
        27: "flores de jardín",            # carpeta 31
        28: "amor en la niebla",           # carpeta 32
        29: "áster mexicano",              # carpeta 33
        30: "cardo azul alpino",           # carpeta 34
        31: "cattleya labios rubí",        # carpeta 35
        32: "flor del Cabo",               # carpeta 36
        33: "gran astrancia",              # carpeta 37
        34: "tulipán siamés",              # carpeta 38
        35: "rosa de cuaresma",            # carpeta 39
        36: "caléndula inglesa",           # carpeta 4
        37: "barbetón margarita",          # carpeta 40
        38: "narciso",                    # carpeta 41
        39: "lirio espada",                # carpeta 42
        40: "flor de pascua",              # carpeta 43
        41: "bolero azul profundo",        # carpeta 44
        42: "flor de pared",               # carpeta 45
        43: "caléndula",                   # carpeta 46
        44: "botón de oro",                # carpeta 47
        45: "margarita común",             # carpeta 48
        46: "diente de león",              # carpeta 49
        47: "lirio tigre",                 # carpeta 5
        48: "petunia",                    # carpeta 50
        49: "pensamiento silvestre",      # carpeta 51
        50: "prímula",                    # carpeta 52
        51: "girasol",                    # carpeta 53
        52: "pelargonio",                 # carpeta 54
        53: "dalia obispo de llandaff",   # carpeta 55
        54: "gaura",                     # carpeta 56
        55: "geranio",                   # carpeta 57
        56: "dalia naranja",              # carpeta 58
        57: "dalia rosa y amarilla",      # carpeta 59
        58: "orquídea lunar",             # carpeta 6
        59: "cautleya espinosa",          # carpeta 60
        60: "anémona japonesa",            # carpeta 61
        61: "susana de ojos negros",      # carpeta 62
        62: "arbusto plateado",           # carpeta 63
        63: "amapola californiana",       # carpeta 64
        64: "osteospermo",                # carpeta 65
        65: "azafrán de primavera",       # carpeta 66
        66: "iris barbado",               # carpeta 67
        67: "anémona",                   # carpeta 68
        68: "amapola arbórea",            # carpeta 69
        69: "ave del paraíso",            # carpeta 7
        70: "gazanias",                  # carpeta 70
        71: "azalea",                    # carpeta 71
        72: "nenúfar",                   # carpeta 72
        73: "rosa",                      # carpeta 73
        74: "manzana espinosa",           # carpeta 74
        75: "gloria de la mañana",        # carpeta 75
        76: "pasiflora",                  # carpeta 76
        77: "loto",                      # carpeta 77
        78: "lirio sapo",                # carpeta 78
        79: "anturio",                   # carpeta 79
        80: "casco de monje",             # carpeta 8
        81: "frangipani",                # carpeta 80
        82: "clemátide",                 # carpeta 81
        83: "hibisco",                   # carpeta 82
        84: "aguileña",                  # carpeta 83
        85: "rosa del desierto",          # carpeta 84
        86: "malva arbórea",              # carpeta 85
        87: "magnolia",                 # carpeta 86
        88: "ciclamen",                 # carpeta 87
        89: "berro",                    # carpeta 88
        90: "canna índica",             # carpeta 89
        91: "cardo globo",               # carpeta 9
        92: "hippeastrum",              # carpeta 90
        93: "bergamota",                # carpeta 91
        94: "musgo de bola",            # carpeta 92
        95: "dedalera",                 # carpeta 93
        96: "bugambilia",               # carpeta 94
        97: "camelia",                 # carpeta 95
        98: "malva",                   # carpeta 96
        99: "petunia mexicana",         # carpeta 97
        100: "bromelia",               # carpeta 98
        101: "gallardía"               # carpeta 99
    }

nombres_clases = cargar_nombres_clases_oficial()

def obtener_info_wikipedia(flor_nombre):
    try:
        page = wikipedia.page(flor_nombre, auto_suggest=True)
        return {
            "titulo": page.title,
            "resumen": page.summary[:1000] + "...",
            "url": page.url,
            "imagenes": [img for img in page.images if img.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
        }
    except wikipedia.exceptions.DisambiguationError as e:
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
        return None
    except Exception as e:
        st.error(f"Error al consultar Wikipedia: {str(e)}")
        return None

# --- Interfaz principal ---
uploaded_file = st.file_uploader("Elige una imagen de flor...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col_upload, col_results = st.columns([1, 2], gap="large")
    
    with col_upload:
        with st.container(border=True):
            st.image(uploaded_file, caption="Imagen subida", width=300, use_container_width=True)
            st.markdown("---")
            classify_btn = st.button("🔎 Analizar", type="primary", use_container_width=True)
    
    with col_results:
        if classify_btn:
            with st.spinner("Analizando la imagen..."):
                try:
                    img_array = cargar_preprocesar_imagen_desde_bytes(uploaded_file)
                    predictions = modelo.predict(img_array)
                    predicted_class = int(np.argmax(predictions[0]))
                    confidence = float(np.max(predictions[0])) * 100
                    class_name = nombres_clases.get(predicted_class, f"Clase {predicted_class}")
                    wiki_info = obtener_info_wikipedia(class_name)
                except Exception as e:
                    st.error(f"Ocurrió un error en la predicción: {e}")
                    st.stop()

            result_tabs = st.tabs(["📚 Análisis", "📊 Top 5", "🌍 Wikipedia"])
            
            with result_tabs[0]:
                st.markdown(f"### 🌸 Flor detectada: `{class_name}`")
                st.metric("Confianza", f"{confidence:.2f}%")
            
            with result_tabs[1]:
                st.subheader("Mejores 5 Coincidencias: ")
                top5 = np.argsort(predictions[0])[::-1][:5]
                for i, idx in enumerate(top5):
                    nombre = nombres_clases.get(idx, f"Clase {idx}")
                    porcentaje = predictions[0][idx] * 100
                    st.write(f"**{i+1}. {nombre}** — `{porcentaje:.2f}%`")
                    st.progress(float(predictions[0][idx]))

            with result_tabs[2]:
                if wiki_info:
                    st.subheader(wiki_info["titulo"])
                    st.write(wiki_info["resumen"])
                    if wiki_info["imagenes"]:
                        st.image(wiki_info["imagenes"][0], use_container_width=True)
                    st.page_link(wiki_info["url"], label="📖 Ver artículo completo")
                else:
                    st.warning("Información no encontrada en Wikipedia")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ℹ️ Acerca de")
    st.markdown("""
**Tecnologías:**
- TensorFlow/Keras
- Wikipedia API  
**Dataset:** Oxford Flowers 102  
**Autor:** Romero Luis E.
""")
    st.markdown("---")
    st.markdown("👨‍💻 [Código en GitHub](https://github.com/LuisEduardoRomeroOlmos/Proyectos)")