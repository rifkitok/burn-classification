import os
import sys
import cv2
import numpy as np

def preprocess_and_segment(image_path, resize_dim=(256, 256), show_gui=True):
    """
    Melakukan pra-pemrosesan dan segmentasi area luka bakar pada citra kulit.
    1. Membaca citra dan mengubah ukuran (resize).
    2. Deteksi warna kulit manusia menggunakan ruang warna HSV.
    3. Thresholding Otsu pada area kulit untuk mengekstrak kontur luka bakar.
    4. Masking hasil segmentasi.
    """
    if not os.path.exists(image_path):
        print(f"Error: File citra '{image_path}' tidak ditemukan.")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Gagal memuat citra dari '{image_path}'.")
        return None

    # Resize image
    image = cv2.resize(image, resize_dim)

    # Rentang warna kulit manusia dalam HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Konversi ke HSV dan buat skin mask
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin_area = cv2.bitwise_and(image, image, mask=skin_mask)

    # Konversi skin area ke grayscale untuk thresholding
    gray_image = cv2.cvtColor(skin_area, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Segmentasi kontur area luka bakar
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    smooth_image = cv2.bitwise_and(image, mask)

    if show_gui:
        try:
            cv2.imshow('Original Image', image)
            cv2.imshow('Skin Area', skin_area)
            cv2.imshow('Segmented Result', smooth_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            # Fallback jika environment headless (tanpa display X11)
            output_file = 'segmented_result.jpg'
            cv2.imwrite(output_file, smooth_image)
            print(f"Display GUI tidak tersedia. Hasil segmentasi disimpan ke: {output_file}")
    
    return smooth_image

if __name__ == '__main__':
    # Tentukan path gambar input dari argumen CLI atau default
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Coba cari sampel gambar yang tersedia di uploads jika default tidak ada
        default_candidates = [
            'img25.jpg',
            os.path.join('flask web', 'uploads', 'img25.jpg'),
            os.path.join('flask web', 'uploads', '3.jpg'),
        ]
        img_path = next((p for p in default_candidates if os.path.exists(p)), 'img25.jpg')

    print(f"Memproses gambar: {img_path}")
    preprocess_and_segment(img_path)

