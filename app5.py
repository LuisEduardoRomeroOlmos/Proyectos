import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import wikipedia
import io
from fpdf import FPDF
import matplotlib.cm as cm


# Configuración
st.set_page_config(page_title="Clasificador de Razas de Perros", layout="wide")
st.title("🐶 Clasificador de Perros 🐾")
st.markdown("""
Sube una imagen de un perro y el modelo predecirá a cuál de las 120 razas del dataset Stanford Dogs pertenece,
mostrando además información relevante de Wikipedia.
""")
st.caption("📂 También puedes arrastrar y soltar la imagen aquí.")

# Idioma Wikipedia
idioma = st.selectbox("🌐 Idioma de Wikipedia", ["es", "en"])
wikipedia.set_lang(idioma)

# Carga y preprocesamiento
def cargar_preprocesar_imagen_desde_bytes(file, tamanio=(224, 224)):
    img = Image.open(file).convert("RGB")
    img = img.resize(tamanio)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

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

# Grad-CAM (estructura inicial, requiere implementación real)


def obtener_heatmap_gradcam(img_array, modelo, clase_idx):
    # Buscar recursivamente la última capa Conv2D
    def buscar_ultima_conv2d(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                name = buscar_ultima_conv2d(sublayer)
                if name:
                    return name
        return None

    ultima_capa_conv = buscar_ultima_conv2d(modelo)
    if ultima_capa_conv is None:
        st.error("No se encontró una capa Conv2D en el modelo.")
        return None

    grad_model = tf.keras.models.Model(
        inputs=modelo.input,
        outputs=[modelo.get_layer(ultima_capa_conv).output, modelo.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, clase_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    heatmap = Image.fromarray(np.uint8(cm.jet(cam.numpy()) * 255)).resize((224, 224))
    return heatmap

# Historial
if "historial" not in st.session_state:
    st.session_state.historial = []

# Carga imagen
uploaded_file = st.file_uploader("Elige una imagen de perro...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.image(uploaded_file, caption="Imagen subida", width=300, use_container_width=True)
        st.markdown("---")
        classify_btn = st.button("🪄 Clasificar", type="primary", use_container_width=True)

    with col2:
        if classify_btn:
            with st.spinner("Analizando la imagen..."):
                try:
                    img_array = cargar_preprocesar_imagen_desde_bytes(uploaded_file)
                    predictions = modelo.predict(img_array)
                    predicted_class = int(np.argmax(predictions[0]))
                    confidence = float(np.max(predictions[0])) * 100
                    class_name = nombres_clases.get(predicted_class, f"Clase {predicted_class}")
                    wiki_info = obtener_info_wikipedia(class_name)
                    st.session_state.historial.append((class_name, confidence))
                except Exception as e:
                    st.error(f"Ocurrió un error en la predicción: {e}")
                    st.stop()

            result_tabs = st.tabs(["🔮 Predicción", "📊 Top 5", "🌍 Wikipedia", "🖼️ Grad-CAM", "📥 Descargar"])

            with result_tabs[0]:
                st.markdown(f"### 🐶 Raza detectada: {class_name}")
                st.metric("Confianza", f"{confidence:.2f}%")

            with result_tabs[1]:
                top5 = np.argsort(predictions[0])[::-1][:5]
                top5_dict = {nombres_clases[i]: predictions[0][i] for i in top5}
                df = pd.DataFrame.from_dict(top5_dict, orient="index", columns=["Probabilidad"])
                st.bar_chart(df)

            with result_tabs[2]:
                if wiki_info:
                    st.subheader(wiki_info["titulo"])
                    st.write(wiki_info["resumen"])
                    for img_url in wiki_info["imagenes"]:
                        st.image(img_url, use_container_width=True)
                    st.page_link(wiki_info["url"], label="📖 Ver artículo completo")
                else:
                    st.warning("Información no encontrada en Wikipedia.")

            with result_tabs[3]:
                heatmap = obtener_heatmap_gradcam(img_array, modelo, predicted_class)
                if heatmap is not None:
                    st.image(heatmap, caption="Grad-CAM", use_container_width=True)
                else:
                    st.info("🧠 Visualización Grad-CAM aún no implementada.")

            with result_tabs[4]:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Resultado de Clasificación", ln=True)
                pdf.cell(200, 10, txt=f"Raza: {class_name}", ln=True)
                pdf.cell(200, 10, txt=f"Confianza: {confidence:.2f}%", ln=True)
                if wiki_info:
                    pdf.multi_cell(0, 10, txt=f"Wikipedia: {wiki_info['url']}")
                pdf_output = io.BytesIO()
                pdf.output(pdf_output)
                st.download_button("📄 Descargar resultado (PDF)", data=pdf_output.getvalue(),
                                   file_name="resultado_perro.pdf", mime="application/pdf")

# Sidebar
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

    if st.session_state.historial:
        st.markdown("### 🐾 Historial de Predicciones")
        for raza, conf in reversed(st.session_state.historial[-5:]):
            st.write(f"{raza}: {conf:.2f}%")
