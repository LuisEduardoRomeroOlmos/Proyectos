import numpy as np
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf

def make_gradcam_heatmap(img_batch, model):
    """
    img_batch: batch de 1 imagen preprocesada con shape (1, H, W, 3)
    model: modelo funcional con dos salidas [conv_outputs, preds]
    """
    with tf.GradientTape() as tape:
        # Ejecutamos el modelo para obtener activaciones y predicciones
        conv_outputs, preds = model(img_batch, training=False)

        # Nos aseguramos de que tape observe conv_outputs
        tape.watch(conv_outputs)

        # Como es binaria, tomamos la neurona 0
        class_channel = preds[:, 0]

    # Gradiente de la clase con respecto a conv_outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # Ahora sí gradientes no serán None
    # Promediamos sobre el batch y la altura/anchura
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Quitamos la dim de batch de conv_outputs
    conv_outputs = conv_outputs[0]

    # Weighted sum: conv_outputs * pooled_grads, sumando por canal
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU + normalización
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-10)

    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = img.resize((300, 300))
    heatmap = np.uint8(255 * heatmap)

    colormap = cm.get_cmap('jet')
    heatmap_color = colormap(heatmap / 255.0)[:, :, :3]
    heatmap_color = np.uint8(heatmap_color * 255)

    superimposed = np.array(img) * (1 - alpha) + heatmap_color * alpha
    superimposed = np.uint8(superimposed)

    return Image.fromarray(superimposed)

