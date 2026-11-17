import os
import sys

# Matikan log berlebih dari TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(150, 150, 3), num_classes=4):
    """
    Membangun arsitektur Convolutional Neural Network (CNN) untuk klasifikasi citra luka bakar.
    - 3 Blok Convolution (32, 64, 128 filter) dengan ReLU & MaxPooling (2, 2)
    - Fully Connected Layer: Dense 512, Dropout 0.5
    - Output: Dense 4 dengan Softmax
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Custom data generator jika ada task regresi berbasis file CSV
class RegressionDataGenerator(Sequence):
    def __init__(self, image_dir, labels, batch_size=32, target_size=(150, 150), shuffle=True):
        self.image_dir = image_dir
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.image_paths = list(self.labels.keys())
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = self.__data_generation(batch_paths)
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)

    def __data_generation(self, batch_paths):
        images = []
        labels = []
        for img_path in batch_paths:
            full_path = os.path.join(self.image_dir, img_path)
            img = tf.keras.preprocessing.image.load_img(full_path, target_size=self.target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0
            images.append(img)
            labels.append(self.labels[img_path])
        return np.array(images), np.array(labels)

def load_labels_from_csv(csv_path):
    """Membaca label dari file CSV."""
    df = pd.read_csv(csv_path)
    labels = dict(zip(df['filename'], df['label']))
    return labels

def train_classification_model(train_dir='dataset/train_data', valid_dir='dataset/valid_data', epochs=25, batch_size=32):
    """Melakukan proses training model klasifikasi 4-kelas luka bakar."""
    if not os.path.exists(train_dir):
        print(f"Error: Direktori dataset latihan '{train_dir}' tidak ditemukan!")
        print("Pastikan struktur dataset berisi subfolder per-kelas:")
        print("  dataset/train_data/Class1/")
        print("  dataset/train_data/Class2/")
        print("  dataset/train_data/Class3/")
        print("  dataset/train_data/Class4/")
        return None

    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Cek apakah direktori validasi tersedia
    val_generator = None
    if os.path.exists(valid_dir):
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)
        val_generator = val_datagen.flow_from_directory(
            valid_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical'
        )
    else:
        print(f"Catatan: Direktori validasi '{valid_dir}' tidak ditemukan. Menggunakan train_generator untuk evaluasi.")
        val_generator = train_generator

    model = build_model(input_shape=(150, 150, 3), num_classes=4)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print(f"\nMemulai training selama {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=1
    )

    save_path = 'burn_classification_model_fix.h5'
    model.save(save_path)
    print(f"\nModel berhasil disimpan ke: {save_path}")

    # Salin juga ke folder flask web jika ada
    flask_model_path = os.path.join('flask web', 'burn_classification_model_fix.h5')
    if os.path.exists('flask web'):
        model.save(flask_model_path)
        print(f"Model juga disalin ke: {flask_model_path}")

    return model

def evaluate_regression_experiment(model, valid_dir='dataset/valid_data', csv_path='label.csv'):
    """Eksperimen evaluasi regresi (opsional jika label.csv tersedia)."""
    if not (os.path.exists(valid_dir) and os.path.exists(csv_path)):
        print(f"\n[Info] Eksperimen regresi dilewati karena '{valid_dir}' atau '{csv_path}' tidak ditemukan.")
        return

    print("\nMenjalankan eksperimen regresi...")
    valid_labels = load_labels_from_csv(csv_path)
    valid_generator = RegressionDataGenerator(valid_dir, valid_labels, batch_size=32)

    # Modifikasi model: ganti layer klasifikasi dengan layer regresi (1 unit)
    model.pop()
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'accuracy']
    )

    loss, mae, accuracy = model.evaluate(valid_generator)
    print(f'Validation MAE: {mae:.2f}')
    print(f'Validation Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    model = train_classification_model()
    if model is not None:
        evaluate_regression_experiment(model)

