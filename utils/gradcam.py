import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def make_gradcam_heatmap(img_batch, model):
    """
    img_batch: array shape (1, H, W, 3)
    model: modelo funcional con dos salidas:
        [conv_outputs, predictions]
    """
    with tf.GradientTape() as tape:
        # Forward pass único
        conv_outputs, preds = model(img_batch, training=False)
        # Tomamos neurona 0 (sigmoide)
        class_channel = preds[:, 0]

    # Gradiente de la neurona elegida con respecto a conv_outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # Promedio espacial de gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    # Construir heatmap 2D
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU + normalización
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """
    Superpone heatmap sobre imagen PIL.
    """
    # Redimensionar heatmap al tamaño original
    heatmap_uint = np.uint8(255 * heatmap)
    heatmap_img  = Image.fromarray(heatmap_uint).resize(img.size, Image.BILINEAR)
    heatmap_arr  = np.array(heatmap_img)

    # Mapear a colores
    cmap = cm.get_cmap('jet')
    colored = cmap(heatmap_arr/255.0)[:, :, :3]
    colored = np.uint8(colored * 255)

    # Superponer
    img_arr = np.array(img)
    superimposed = img_arr * (1-alpha) + colored * alpha
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)


