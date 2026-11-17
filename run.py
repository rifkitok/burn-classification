import os
import sys

# Matikan log berlebih dari TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def find_model_path():
    """Mencari file model .h5 baik di direktori aktif maupun subdirektori flask web."""
    candidates = [
        'burn_classification_model_fix.h5',
        os.path.join(os.path.dirname(__file__), 'burn_classification_model_fix.h5'),
        os.path.join(os.path.dirname(__file__), 'flask web', 'burn_classification_model_fix.h5'),
        os.path.join('flask web', 'burn_classification_model_fix.h5')
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def load_and_prepare_image(img_path, target_size=(150, 150)):
    """Memuat dan menormalkan citra input ke skala [0, 1] dengan dimensi (1, 150, 150, 3)."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi
    return img_array

def predict_image(model, img_path, class_labels, show_plot=True):
    """Melakukan prediksi kelas luka bakar dan menampilkan probabilitas untuk setiap kelas."""
    if not os.path.exists(img_path):
        print(f"Error: Gambar '{img_path}' tidak ditemukan.")
        return None

    img_array = load_and_prepare_image(img_path)
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_label = class_labels[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    print("=" * 45)
    print(f"Gambar: {img_path}")
    print(f"Hasil Prediksi : {predicted_label} ({confidence:.2f}%)")
    print("-" * 45)
    print("Probabilitas Tiap Kelas:")
    for label, prob in zip(class_labels, predictions):
        print(f"  - {label:<10}: {prob * 100:6.2f}%")
    print("=" * 45)

    if show_plot:
        try:
            plt.figure(figsize=(6, 6))
            plt.imshow(image.load_img(img_path))
            plt.title(f'Prediksi: {predicted_label} ({confidence:.1f}%)')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            output_plot = 'prediction_plot.png'
            plt.savefig(output_plot, bbox_inches='tight')
            print(f"Plot GUI tidak dapat ditampilkan ({e}). Disimpan ke {output_plot}")
        finally:
            plt.close()

    return predicted_label

if __name__ == '__main__':
    model_path = find_model_path()
    if not model_path:
        print("Error: File model 'burn_classification_model_fix.h5' tidak ditemukan!")
        sys.exit(1)

    print(f"Memuat model dari: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Label kelas klasifikasi luka bakar
    class_labels = ['Class1', 'Class2', 'Class3', 'Class4']

    # Ambil path gambar dari argumen CLI atau gunakan sampel default
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        candidates = [
            'img4.jpg',
            os.path.join(os.path.dirname(__file__), 'flask web', 'uploads', '3.jpg'),
            os.path.join('flask web', 'uploads', '3.jpg'),
            os.path.join('uploads', '3.jpg'),
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), 'img4.jpg')

    predict_image(model, img_path, class_labels, show_plot=False)

