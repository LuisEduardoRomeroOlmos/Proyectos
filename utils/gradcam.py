import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def make_gradcam_heatmap(img_batch, model):
    """
    img_batch: array (1, H, W, 3)
    model: modelo funcional con outputs [conv_outputs, final_pred]
    """
    with tf.GradientTape() as tape:
        conv_outputs, preds = model(img_batch, training=False)
        tape.watch(conv_outputs)
        # si tu salida es una sigmoide de 1 neurona:
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    # Si aún grads es None, revisa que last_conv_layer realmente influya en preds

    # Promediado espacial de gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    # Sumar ponderada de mapas
    conv_outputs = conv_outputs[0]  # quitar batch
    heatmap      = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU + normalización
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    # Escala heatmap a uint8 y redimensiona
    heatmap_uint = np.uint8(255 * heatmap)
    heatmap_img  = Image.fromarray(heatmap_uint).resize(img.size, Image.BILINEAR)
    heatmap_arr  = np.array(heatmap_img)

    # Colormap
    cmap = cm.get_cmap('jet')
    colored = cmap(heatmap_arr/255.0)[:, :, :3]
    colored = np.uint8(colored * 255)

    # Superposición
    img_arr = np.array(img)
    superimposed = img_arr * (1-alpha) + colored * alpha
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)



