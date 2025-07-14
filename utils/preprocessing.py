import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(image, target_size=(300, 300)):
    # Redimensionar y convertir a array
    image = image.resize(target_size)
    img_array = np.array(image)

    # Asegurar que tiene 3 canales
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # eliminar canal alpha si existe

    # Preprocesamiento específico para EfficientNetB3
    img_array = preprocess_input(img_array)

    return img_array
