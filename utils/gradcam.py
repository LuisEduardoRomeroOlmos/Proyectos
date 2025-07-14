import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def make_gradcam_heatmap(img_batch, conv_model, pred_model):
    """
    img_batch: tensor shape (1, H, W, 3)
    conv_model: modelo que devuelve la última activación convolucional
    pred_model: modelo que devuelve la predicción final
    """
    with tf.GradientTape() as tape:
        # 1) Sacar activaciones conv y hacer que tape las vigile
        conv_outputs = conv_model(img_batch, training=False)
        tape.watch(conv_outputs)

        # 2) Sacar la predicción final
        preds = pred_model(img_batch, training=False)
        # Clasificación binaria: tomar neuron 0
        class_channel = preds[:, 0]

    # 3) Gradiente de la salida con respecto a conv_outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # 4) Promedio espacial de gradientes por canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5) Construir el heatmap: suma ponderada de los mapas de características
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # 6) ReLU + normalización
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-10)

    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """
    Superpone el heatmap coloreado sobre la imagen original.
    """
    # Redimensionar heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(img.size, resample=Image.BILINEAR)
    heatmap = np.array(heatmap)

    # Aplicar colormap 'jet'
    colormap = cm.get_cmap('jet')
    heatmap_color = colormap(heatmap / 255.0)[:, :, :3]
    heatmap_color = np.uint8(heatmap_color * 255)

    # Superposición
    img_array = np.array(img)
    superimposed = img_array * (1 - alpha) + heatmap_color * alpha
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)

