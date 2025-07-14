import numpy as np
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf

def make_gradcam_heatmap(img_array, model):
    # model devuelve [conv_output, preds]
    conv_output, predictions = model(img_array, training=False)
    pred_index = tf.argmax(predictions[0])

    with tf.GradientTape() as tape:
        conv_output, predictions = model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
    
def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = img.resize((300, 300))
    heatmap = np.uint8(255 * heatmap)

    colormap = cm.get_cmap('jet')
    heatmap_color = colormap(heatmap / 255.0)[:, :, :3]  # solo RGB, sin alpha
    heatmap_color = np.uint8(heatmap_color * 255)

    superimposed_img = np.array(img) * (1 - alpha) + heatmap_color * alpha
    superimposed_img = np.uint8(superimposed_img)

    return Image.fromarray(superimposed_img)
