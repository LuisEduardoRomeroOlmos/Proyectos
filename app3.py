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
def cargar_nombres_clases_oficial_es():
    return {
        0: "Prímula rosa",
        1: "Orquídea de hojas duras",
        2: "Campanas de Canterbury",
        3: "Guisante dulce",
        4: "Caléndula inglesa",
        5: "Lirio tigre",
        6: "Orquídea luna",
        7: "Ave del paraíso",
        8: "Acónito",
        9: "Cardo globo",
        10: "Boca de dragón",
        11: "Farfara",
        12: "Protea rey",
        13: "Cardo lanzador",
        14: "Iris amarillo",
        15: "Flor globo",
        16: "Echinácea púrpura",
        17: "Lirio peruano",
        18: "Flor globo (balloon flower)",
        19: "Lirio arum gigante blanco",
        20: "Lirio de fuego",
        21: "Flor alfiler",
        22: "Fritilaria",
        23: "Jengibre rojo",
        24: "Jacinto de uva",
        25: "Amapola del campo",
        26: "Plumas del príncipe de Gales",
        27: "Genciana sin tallo",
        28: "Alcachofa",
        29: "Dianthus barbatus (Clavel del poeta)",
        30: "Clavel",
        31: "Flores de jardín (Phlox)",
        32: "Amor en la niebla",
        33: "Aster mexicano",
        34: "Eryngium alpino",
        35: "Orquídea cattleya",
        36: "Flor del cabo",
        37: "Astrantia mayor",
        38: "Tulipán siamés",
        39: "Rosa de Cuaresma",
        40: "Margarita barbeton",
        41: "Narciso",
        42: "Gladiolo",
        43: "Flor de Pascua",
        44: "Bolero azul profundo",
        45: "Alhelí",
        46: "Caléndula",
        47: "Ranúnculo",
        48: "Margarita común",
        49: "Diente de león común",
        50: "Petunia",
        51: "Pensamiento silvestre",
        52: "Prímula",
        53: "Girasol",
        54: "Geranio",
        55: "Obispo de Llandaff",
        56: "Gaura",
        57: "Geranio",
        58: "Dalia naranja",
        59: "Dalia rosa-amarilla",
        60: "Cautleya spicata",
        61: "Anémona japonesa",
        62: "Susana de ojos negros",
        63: "Arbusto plateado",
        64: "Amapola californiana",
        65: "Osteospermum",
        66: "Crocus de primavera",
        67: "Iris",
        68: "Anémona",
        69: "Amapola arbórea",
        70: "Gazania",
        71: "Azalea",
        72: "Nenúfar",
        73: "Rosa",
        74: "Manzana espinosa",
        75: "Gloria de la mañana",
        76: "Flor de la pasión",
        77: "Loto",
        78: "Lirio sapo",
        79: "Anturio",
        80: "Frangipani",
        81: "Clemátide",
        82: "Hibisco",
        83: "Aguileña",
        84: "Rosa del desierto",
        85: "Malva arbórea",
        86: "Magnolia",
        87: "Ciclamen",
        88: "Berro de agua",
        89: "Caña de lirio",
        90: "Amarilis",
        91: "Bálsamo de abeja",
        92: "Flor globo",
        93: "Gallardia",
        94: "Dedalera",
        95: "Buganvilla",
        96: "Camelia",
        97: "Malva",
        98: "Petunia mexicana",
        99: "Bromelia",
        100: "Gallardia",
        101: "Enredadera trompeta"
    }


nombres_clases = cargar_nombres_clases()

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
            classify_btn = st.button("🪄 Clasificar", type="primary", use_container_width=True)
    
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

            result_tabs = st.tabs(["🔮 Predicción", "📊 Detalles", "🌍 Wikipedia"])
            
            with result_tabs[0]:
                st.markdown(f"### 🌸 Flor detectada: `{class_name}`")
                st.metric("Confianza", f"{confidence:.2f}%")
            
            with result_tabs[1]:
                st.subheader("Top 5 predicciones")
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