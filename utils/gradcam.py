import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

import tensorflow as tf
import numpy as np

def make_gradcam_heatmap(img_batch, model):
    with tf.GradientTape() as tape:
        conv_outputs, preds = model(img_batch, training=False)
        tape.watch(conv_outputs)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    
    # ——— Añadimos prints de depuración ———
    tf.print("conv_outputs shape:", tf.shape(conv_outputs))
    tf.print("grads shape:", tf.shape(grads))

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    tf.print("pooled_grads shape:", tf.shape(pooled_grads))
    # ————————————————————————————————

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
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


