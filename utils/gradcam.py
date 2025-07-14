import numpy as np
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf

def make_gradcam_heatmap(img_batch, model):
    # img_batch: shape (1, H, W, 3)
    # model: devuelve [conv_outputs, predictions]
    with tf.GradientTape() as tape:
        conv_outputs, preds = model(img_batch, training=False)
        # Si tu salida es sigmoide (1 neurona), siempre tomas índice 0:
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    # Suma ponderada de los mapas:
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = img.resize((300, 300))
    heatmap = np.uint8(255 * heatmap)

    colormap = cm.get_cmap('jet')
    heatmap_color = colormap(heatmap / 255.0)[:, :, :3]  # solo RGB, sin canal alpha
    heatmap_color = np.uint8(heatmap_color * 255)

    # Superponer heatmap sobre imagen original
    superimposed_img = np.array(img) * (1 - alpha) + heatmap_color * alpha
    superimposed_img = np.uint8(superimposed_img)

    return Image.fromarray(superimposed_img)
