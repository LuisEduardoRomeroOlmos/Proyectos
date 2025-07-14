import numpy as np
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf

def make_gradcam_heatmap(img_array, model):
    """
    img_array: imagen preprocesada con shape (1, H, W, 3)
    model: modelo con 2 salidas: [conv_outputs, predictions]
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = model(img_array, training=False)
        
        # Si tu salida es una sola neurona (sigmoide), toma índice 0
        pred_index = 0
        
        # Para multi-clase sería: pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)  # Gradientes de la salida con respecto a la última conv
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Promedio de gradientes por canal

    conv_outputs = conv_outputs[0]  # Quitar batch dimension
    
    # Ponderar cada canal por su gradiente promedio y sumar
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    
    # Normalizar y aplicar ReLU para quitar negativos
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-10)
    
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
