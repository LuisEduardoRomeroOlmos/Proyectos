import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import wikipedia

# Configuración de la página
st.set_page_config(page_title="🦴 Clasificador de Razas de Perros", layout="wide")

st.title("🐶 Clasificador de Perros 🐾")
st.markdown("""
Sube una imagen de tu lomito y el modelo predecirá a cuál de las 120 razas del dataset Stanford Dogs pertenece,
mostrando además información relevante de Wikipedia.
""")

wikipedia.set_lang("es")

# --- Funciones ---
def cargar_preprocesar_imagen_desde_bytes(file, tamanio=(224, 224)):
    img = Image.open(file).convert("RGB")
    img = img.resize(tamanio)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@st.cache_resource
def cargar_modelo():
    return load_model("Modelos/Clasificacion_perros.keras")

modelo = cargar_modelo()

@st.cache_data
def nombres_perros():
    return {
        0: "Chihuahua", 1: "Japanese spaniel", 2: "Maltese dog", 3: "Pekinese", 4: "Shih-Tzu",
        5: "Blenheim spaniel", 6: "Papillon", 7: "Toy terrier", 8: "Rhodesian ridgeback",
        9: "Afghan hound", 10: "Basset", 11: "Beagle", 12: "Bloodhound", 13: "Bluetick",
        14: "Black-and-tan coonhound", 15: "Walker hound", 16: "English foxhound",
        17: "Redbone", 18: "Borzoi", 19: "Irish wolfhound", 20: "Italian greyhound",
        21: "Whippet", 22: "Ibizan hound", 23: "Norwegian elkhound", 24: "Otterhound",
        25: "Saluki", 26: "Scottish deerhound", 27: "Weimaraner", 28: "Staffordshire bullterrier",
        29: "American Staffordshire terrier", 30: "Bedlington terrier", 31: "Border terrier",
        32: "Kerry blue terrier", 33: "Irish terrier", 34: "Norfolk terrier", 35: "Norwich terrier",
        36: "Yorkshire terrier", 37: "Wire-haired fox terrier", 38: "Lakeland terrier",
        39: "Sealyham terrier", 40: "Airedale", 41: "Cairn", 42: "Australian terrier",
        43: "Dandie Dinmont", 44: "Boston bull", 45: "Miniature schnauzer", 46: "Giant schnauzer",
        47: "Standard schnauzer", 48: "Scotch terrier", 49: "Tibetan terrier",
        50: "Silky terrier", 51: "Soft-coated wheaten terrier", 52: "West Highland white terrier",
        53: "Lhasa", 54: "Flat-coated retriever", 55: "Curly-coated retriever",
        56: "Golden retriever", 57: "Labrador retriever", 58: "Chesapeake Bay retriever",
        59: "German short-haired pointer", 60: "Vizsla", 61: "English setter",
        62: "Irish setter", 63: "Gordon setter", 64: "Brittany spaniel", 65: "Clumber",
        66: "English springer", 67: "Welsh springer spaniel", 68: "Cocker spaniel",
        69: "Sussex spaniel", 70: "Irish water spaniel", 71: "Kuvasz", 72: "Schipperke",
        73: "Groenendael", 74: "Malinois", 75: "Briard", 76: "Kelpie", 77: "Komondor",
        78: "Old English sheepdog", 79: "Shetland sheepdog", 80: "Collie",
        81: "Border collie", 82: "Bouvier des Flandres", 83: "Rottweiler",
        84: "German shepherd", 85: "Doberman", 86: "Miniature pinscher",
        87: "Greater Swiss Mountain dog", 88: "Bernese mountain dog", 89: "Appenzeller",
        90: "EntleBucher", 91: "Boxer", 92: "Bull mastiff", 93: "Tibetan mastiff",
        94: "French bulldog", 95: "Great Dane", 96: "Saint Bernard", 97: "Eskimo dog",
        98: "Malamute", 99: "Siberian husky", 100: "Affenpinscher", 101: "Basenji",
        102: "Pug", 103: "Leonberg", 104: "Newfoundland", 105: "Great Pyrenees",
        106: "Samoyed", 107: "Pomeranian", 108: "Chow", 109: "Keeshond",
        110: "Brabancon griffon", 111: "Pembroke", 112: "Cardigan", 113: "Toy poodle",
        114: "Miniature poodle", 115: "Standard poodle", 116: "Mexican hairless",
        117: "Dingo", 118: "Dhole", 119: "African hunting dog"
    }

nombres_clases = nombres_perros()

def obtener_info_wikipedia(nombre):
    try:
        page = wikipedia.page(nombre, auto_suggest=True)
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
                "resumen": f"(Resultado alternativo)\n\n{page.summary[:1000]}...",
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
uploaded_file = st.file_uploader("Elige una imagen de perro...", type=["jpg", "jpeg", "png"])

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
                st.markdown(f"### 🐺 Raza detectada: {class_name}")
                st.metric("Confianza", f"{confidence:.2f}%")

            with result_tabs[1]:
                st.subheader("Mejores 5 Coincidencias: ")
                top5 = np.argsort(predictions[0])[::-1][:5]
                for i, idx in enumerate(top5):
                    nombre = nombres_clases.get(idx, f"Clase {idx}")
                    porcentaje = predictions[0][idx] * 100
                    st.write(f"*{i+1}. {nombre}* — {porcentaje:.2f}%")
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
*Tecnologías:*
- TensorFlow/Keras
- Wikipedia API  
*Dataset:* Stanford Dogs Dataset  
*Autor:* Romero Luis E.
""")
    st.markdown("---")
    st.markdown("👨‍💻 [Código en GitHub](https://github.com/LuisEduardoRomeroOlmos/Proyectos)")