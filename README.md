# 🔥 Burn Classification & Detection System

Sistem klasifikasi dan deteksi tingkat keparahan luka bakar (*burn injury*) berbasis citra menggunakan **Convolutional Neural Network (CNN)** dengan antarmuka web interaktif menggunakan **Flask**.

---

## 📌 Daftar Isi
- [Tentang Proyek](#-tentang-proyek)
- [Struktur Direktori](#-struktur-direktori)
- [Arsitektur Model CNN](#-arsitektur-model-cnn)
- [Penjelasan Modul Kode](#-penjelasan-modul-kode)
- [Pembersihan & Refaktorisasi Kode](#-pembersihan--refaktorisasi-kode)
- [Instalasi & Persiapan](#-instalasi--persiapan)
- [Cara Penggunaan](#-cara-penggunaan)
  - [1. Menjalankan Aplikasi Web Flask](#1-menjalankan-aplikasi-web-flask)
  - [2. Inferensi Cepat via Terminal (CLI)](#2-inferensi-cepat-via-terminal-cli)
  - [3. Pra-pemrosesan & Segmentasi Citra](#3-pra-pemrosesan--segmentasi-citra)
  - [4. Melatih Model Ulang (Training)](#4-melatih-model-ulang-training)

---

## 📖 Tentang Proyek

Proyek ini bertujuan untuk membantu identifikasi tingkat luka bakar melalui analisis citra medis atau foto luka. Pipeline sistem terdiri dari:
1. **Pra-pemrosesan & Segmentasi Citra**: Mengisolasi area luka bakar dari warna kulit menggunakan deteksi warna kulit ruang HSV (*skin masking*) dan teknik thresholding Otsu.
2. **Klasifikasi Deep Learning**: Model Convolutional Neural Network (CNN) multi-layer yang dilatih untuk mengklasifikasikan derajat luka bakar ke dalam 4 kelas (`Class1`, `Class2`, `Class3`, `Class4`).
3. **Web Interface (Flask)**: Aplikasi web sederhana yang ramah pengguna untuk mengunggah gambar dan memperoleh hasil prediksi instan beserta tingkat keyakinan (*confidence level*).

---

## 📂 Struktur Direktori

```text
burn classification/
├── README.md                           # Dokumentasi komprehensif proyek
├── requirements.txt                    # Daftar dependensi Python
├── .gitignore                          # Daftar file yang diabaikan oleh Git
├── preprocess.py                       # Skrip segmentasi kulit & pra-pemrosesan citra (OpenCV)
├── run.py                              # Skrip CLI inferensi cepat untuk menguji sebuah gambar
├── train.py                            # Skrip pelatihan model CNN (TensorFlow/Keras)
└── flask web/                          # Direktori aplikasi web berbasis Flask
    ├── app.py                          # Entrypoint & backend server Flask
    ├── burn_classification_model_fix.h5# Model terlatih Keras (bobot & arsitektur)
    ├── static/                         # Aset statis web
    │   ├── dokter.png                  # Ilustrasi grafis halaman web
    │   ├── style.css                   # Gaya CSS halaman beranda
    │   └── upload_style.css            # Gaya CSS halaman unggah & hasil deteksi
    ├── templates/                      # Template HTML Jinja2
    │   ├── home.html                   # Halaman awal / landing page
    │   └── upload.html                 # Halaman upload citra & tampilan hasil prediksi
    └── uploads/                        # Direktori penyimpanan citra yang diunggah pengguna
```

---

## 🧠 Arsitektur Model CNN

Model deep learning dibangun menggunakan TensorFlow/Keras Sequential dengan spesifikasi berikut:

- **Dimensi Input**: Citra berukuran `150 x 150` piksel dengan 3 kanal warna (RGB), dinormalisasi ke rentang `[0, 1]`.
- **Ekstraksi Fitur**:
  - **Blok 1**: `Conv2D(32 filter, kernel 3x3, ReLU)` + `MaxPooling2D(2x2)`
  - **Blok 2**: `Conv2D(64 filter, kernel 3x3, ReLU)` + `MaxPooling2D(2x2)`
  - **Blok 3**: `Conv2D(128 filter, kernel 3x3, ReLU)` + `MaxPooling2D(2x2)`
- **Klasifikasi (Fully Connected)**:
  - `Flatten()` meratakan feature map menjadi vektor 1D
  - `Dense(512 neuron, ReLU)`
  - `Dropout(0.5)` untuk mencegah overfitting
  - `Dense(4 neuron, Softmax)` untuk menghasilkan probabilitas 4 kelas
- **Optimizer & Loss**:
  - Optimizer: `Adam(learning_rate=0.001)`
  - Loss Function: `categorical_crossentropy`
- **Total Parameter**: 19,035,718 parameter (~72.62 MB trainable parameters).

---

## 🛠️ Penjelasan Modul Kode

### 1. `train.py`
Skrip untuk melatih model klasifikasi luka bakar dari dataset gambar.
- Membaca citra latihan dari folder `dataset/train_data` menggunakan `ImageDataGenerator` Keras dengan pembagian per kelas otomatis.
- Melatih model selama 25 epoch dengan metrik akurasi.
- Menyimpan bobot model ke file `burn_classification_model_fix.h5`.
- Memiliki modul eksperimen regresi berbasis `RegressionDataGenerator` dan `label.csv` jika data skor regresi tersedia.

### 2. `preprocess.py`
Skrip pra-pemrosesan citra luka menggunakan pustaka OpenCV:
- Mengubah ukuran citra input menjadi `256 x 256`.
- Menerapkan segmentasi warna kulit berbasis ruang warna **HSV** (`lower_skin = [0, 48, 80]`, `upper_skin = [20, 255, 255]`).
- Menerapkan **Otsu Thresholding** untuk memisahkan luka bakar dari area kulit normal.
- Mengekstrak kontur luka bakar dan membuat masker biner.

### 3. `run.py`
Skrip mandiri berbasis Command Line Interface (CLI) untuk menguji satu file gambar luka bakar:
- Otomatis mendeteksi model `burn_classification_model_fix.h5` di direktori saat ini maupun di dalam folder `flask web/`.
- Menerima argumen path gambar melalui terminal.
- Menampilkan hasil prediksi kelas utama beserta rincian persentase keyakinan (*probability score*) untuk setiap kelas.

### 4. `flask web/app.py`
Server aplikasi web Flask yang menyediakan antarmuka pengguna:
- Route `/`: Menampilkan landing page dengan tombol mulai.
- Route `/upload` (GET): Menampilkan antarmuka unggah foto dengan fitur live image preview.
- Route `/upload` (POST): Menyimpan foto ke direktori `uploads/`, menjalankan inferensi model, dan menampilkan hasil diagnosa langsung di antarmuka web.
- Endpoint API JSON: Jika klien mengirimkan header `Accept: application/json`, server akan merespons data prediksi dalam format JSON.

---

## 🧹 Pembersihan & Refaktorisasi Kode

Selama audit kode, beberapa perbaikan dan pembersihan telah dilakukan:

1. **Penghapusan File Sampah (*Dump File*)**:
   - Menghapus file `flask web/tempCodeRunnerFile.py` (file sementara sisa ekstensi VS Code).
2. **Perbaikan Bug & Dead Code pada `preporc.py`**:
   - Menghapus variabel `rgb`, `lower_blue`, dan `upper_blue` yang sebelumnya belum didefinisikan (*undefined variables*) dan menyebabkan program crash dengan `NameError`.
   - Mengintegrasikan fungsi resize gambar dengan variabel dimensi `resize_dim=(256, 256)`.
   - Menambahkan penanganan headless display (menyimpan hasil ke gambar jika GUI X11 tidak aktif).
3. **Pemberantasan Blocking GUI pada Backend Server (`app.py`)**:
   - Menghapus `import matplotlib.pyplot as plt` dan pemanggilan `plt.show()` di dalam route Flask. Pemanggilan `plt.show()` sebelumnya menyebabkan thread web server macet (*blocking*) setiap kali pengguna mengunggah gambar.
4. **Penyesuaian Path Dinamis (*Dynamic Path Resolution*)**:
   - Memperbaiki path relatif model dan folder unggahan pada `app.py` dan `run.py` menggunakan `os.path.abspath(__file__)` sehingga skrip dapat dieksekusi dari direktori mana pun tanpa `FileNotFoundError`.
5. **Peningkatan Antarmuka Hasil Deteksi (`upload.html`)**:
   - Menampilkan hasil diagnosis, persentase keyakinan, dan foto yang diunggah langsung di dalam kartu UI, menggantikan tampilan string teks polos sebelumnya.

---

## 🚀 Instalasi & Persiapan

### Prasyarat
- Python versi 3.10 atau 3.11
- Virtual environment disarankan

### Langkah Instalasi
```bash
# Masuk ke direktori proyek
cd "/home/rifkitok/Public/burn classification"

# Buat virtual environment (jika belum ada)
python3 -m venv .venv

# Aktifkan virtual environment
source .venv/bin/activate

# Install semua dependensi
pip install -r requirements.txt
```

---

## 🖥️ Antarmuka Web Interaktif (Awwwards-Grade UI)

Antarmuka web telah ditingkatkan menggunakan standar desain **Awwwards Site of the Day** (kategori *MedTech / Clinical Precision AI*):
- **Desain Imersif MedTech**: Palet warna *Dark Slate & Bio-Cyan* dengan efek glassmorphism, glowing grid, dan kontras tinggi standar WCAG.
- **Tipografi Fluid**: Menggunakan kombinasi font Google `Plus Jakarta Sans` (Display), `Inter` (Body), dan `JetBrains Mono` (Data metrics & probabilities).
- **Multi-View Visual Inspector**:
  - Tampilan interaktif beralih seketika antara: **Foto Asli**, **Masker Segmentasi OpenCV (Otsu)**, dan **Overlay Kontur Neon Luka**.
  - Animasi laser scanline berkecepatan tinggi saat proses analisis saraf sedang berlangsung.
- **Deep Neural Assessment Console**:
  - Diagnosis derajat luka bakar utama dengan badge status dan severity level.
  - Gauge persentase keyakinan model (*Primary Confidence*) dan rasio estimasi luas luka bakar (*Estimated Burn Area %*).
  - Distribusi probabilitas seluruh 4 kelas (*Softmax Distribution*) dengan animasi bar progress yang halus.
  - Panduan protokol medis pertolongan pertama (*First Aid Action Checklist*) yang disesuaikan dengan derajat luka bakar terdeteksi.
- **Pengujian 1-Klik Kasus Klinis**: Tersedia 4 tombol sampel tervalidasi (Grade 1 hingga Grade 4) untuk pengujian langsung tanpa perlu mencari file foto di komputer.
- **Cetak & Ekspor Diagnosa**: Fitur cetak ringkasan klinis langsung dari browser.
- **Aksesibilitas & Performa**: Mendukung penuh `@media (prefers-reduced-motion)` dan transisi 60fps yang ringan.

---

## 🖥️ Cara Penggunaan

### 1. Menjalankan Aplikasi Web Flask
```bash
# Masuk ke direktori web
cd "flask web"

# Jalankan server Flask
python app.py
```
Akses aplikasi melalui peramban web di alamat:  
👉 **`http://localhost:5000`**

### 2. Inferensi Cepat via Terminal (CLI)
Anda dapat memprediksi citra luka bakar langsung dari terminal menggunakan `run.py`:
```bash
# Menguji gambar tertentu:
python run.py "flask web/uploads/3.jpg"

# Output contoh:
# =============================================
# Gambar: flask web/uploads/3.jpg
# Hasil Prediksi : Class3 (74.24%)
# ---------------------------------------------
# Probabilitas Tiap Kelas:
#   - Class1    :  12.55%
#   - Class2    :   7.62%
#   - Class3    :  74.24%
#   - Class4    :   5.59%
# =============================================
```

### 3. Pra-pemrosesan & Segmentasi Citra
Untuk melihat visualisasi segmentasi area luka bakar menggunakan OpenCV:
```bash
python preprocess.py "flask web/uploads/img25.jpg"
```

### 4. Melatih Model Ulang (Training)
Jika memiliki dataset baru, susun direktori data seperti berikut:
```text
dataset/
├── train_data/
│   ├── Class1/
│   ├── Class2/
│   ├── Class3/
│   └── Class4/
└── valid_data/ (opsional)
    ├── Class1/
    ├── Class2/
    ├── Class3/
    └── Class4/
```
Lalu jalankan proses pelatihan:
```bash
python train.py
```
Model terlatih otomatis disimpan ke file `burn_classification_model_fix.h5`.
