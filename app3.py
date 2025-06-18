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
    return load_model("flowers_102_model.keras")  # Asegúrate de tener el modelo correcto

modelo = cargar_modelo()

# Cargar los nombres de las clases
@st.cache_resource
def cargar_nombres_clases():
    return {
        0: "pink primrose",
        1: "hard-leaved pocket orchid",
        # ... (todo el diccionario de 102 flores que te proporcioné antes)
        101: "trumpet creeper"
    }

nombres_clases = cargar_nombres_clases()

# Configurar Wikipedia
@st.cache_resource
def configurar_wikipedia():
    return wikipediaapi.Wikipedia('es')  # Puedes cambiar 'es' por otro idioma

wiki_wiki = configurar_wikipedia()

# Función para obtener información de Wikipedia
def obtener_info_wikipedia(flor_nombre):
    try:
        page = wiki_wiki.page(flor_nombre)
        
        if not page.exists():
            # Intentar con nombre científico si el común no funciona
            nombres_cientificos = {
                "pink primrose": "Primula rosea",
                "rose": "Rosa",
                # Añade más mapeos según necesites
            }
            nombre_cientifico = nombres_cientificos.get(flor_nombre, flor_nombre)
            page = wiki_wiki.page(nombre_cientifico)
            
        if page.exists():
            return {
                "titulo": page.title,
                "resumen": page.summary[:1000] + "...",  # Limitar tamaño
                "url": page.fullurl,
                "imagen": None  # Wikipedia API no da imágenes directamente
            }
        return None
    except Exception as e:
        st.error(f"Error al consultar Wikipedia: {str(e)}")
        return None

# Widget para subir la imagen
uploaded_file = st.file_uploader("Elige una imagen de flor...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Imagen subida", width=300)

    if st.button("Clasificar"):
        with st.spinner("🔍 Analizando la imagen y buscando información..."):
            # Preprocesar imagen
            img_array = cargar_preprocesar_imagen_desde_bytes(uploaded_file)

            # Hacer predicción
            predictions = modelo.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            # Obtener nombre de la clase
            class_name = nombres_clases.get(predicted_class, f"Clase {predicted_class}")

            # Obtener información de Wikipedia
            wiki_info = obtener_info_wikipedia(class_name)

        # Mostrar resultados
        with col2:
            st.success(f"🌺 **Predicción:** {class_name}")
            st.write(f"📊 **Confianza del modelo:** {confidence:.2f}%")

            # Mostrar las 5 mejores predicciones
            st.markdown("### 🔝 Top 5 predicciones:")
            top5_indices = np.argsort(predictions[0])[::-1][:5]
            
            for i, idx in enumerate(top5_indices):
                st.write(f"{i+1}. {nombres_clases.get(idx, f'Clase {idx}')}: {predictions[0][idx]*100:.2f}%")

            # Mostrar información de Wikipedia si está disponible
            if wiki_info:
                st.markdown(f"### 📚 Información sobre *{wiki_info['titulo']}*")
                st.info(wiki_info["resumen"])
                st.markdown(f"[📖 Leer más en Wikipedia]({wiki_info['url']})")
            else:
                st.warning("No se encontró información adicional en Wikipedia para esta flor.")

# Sidebar con info adicional
with st.sidebar:
    st.markdown("## ℹ️ Acerca de esta app")
    st.markdown("""
    **Características:**
    - Clasifica flores entre 102 especies
    - Muestra información relevante de Wikipedia
    - Desarrollado con:
      - TensorFlow/Keras
      - Streamlit
      - Wikipedia API
    """)
    st.markdown("---")
    st.markdown("👨‍💻 [Código en GitHub](https://github.com/LuisEduardoRomeroOlmos/Proyectos/tree/main)")
    st.markdown("🌐 [Dataset Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)")