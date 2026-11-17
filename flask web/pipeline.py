import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Informasi medis klinis komprehensif untuk setiap derajat luka bakar
CLINICAL_INFO = {
    'Class1': {
        'grade': 'GRADE 1',
        'title': '1st Degree Burn (Superficial)',
        'severity': 'Mild',
        'badge_color': '#10b981',
        'depth': 'Lapisan Epidermis (Hanya kulit terluar)',
        'description': 'Luka bakar superfisial yang hanya mengenai lapisan epidermis. Ditandai dengan kulit memerah, kering, terasa perih atau panas saat disentuh, tanpa lepuhan (blister).',
        'healing_time': '3 - 6 hari tanpa meninggalkan jaringan parut (scar).',
        'protocols': [
            'Alirkan air sejuk bersih (bukan air es) selama 10-15 menit pada area luka.',
            'Oleskan gel lidah buaya murni (aloe vera) atau pelembap berbahan dasar air.',
            'Jangan gunakan pasta gigi, mentega, atau minyak goreng pada luka bakar.',
            'Lindungi luka dari paparan langsung sinar matahari hingga sembuh.',
            'Konsumsi analgesik ringan (misal paracetamol) jika terasa nyeri mengganggu.'
        ],
        'emergency_alert': False
    },
    'Class2': {
        'grade': 'GRADE 2',
        'title': '2nd Degree Burn (Partial Thickness)',
        'severity': 'Moderate',
        'badge_color': '#f59e0b',
        'depth': 'Lapisan Epidermis & Sebagian Dermis',
        'description': 'Luka bakar yang menembus lapisan dermis. Ditandai dengan kulit sangat merah, bengkak, lepuhan berisi cairan (bula/blister), dan rasa nyeri yang intens.',
        'healing_time': '2 - 3 minggu dengan perawatan luka steril yang konsisten.',
        'protocols': [
            'Dinginkan segera dengan air mengalir suhu ruang selama minimal 15-20 menit.',
            'JANGAN PERNAH memecahkan lenting atau lepuhan kulit untuk mencegah infeksi sekunder.',
            'Tutup luka secara longgar menggunakan perban atau kassa steril non-stick.',
            'Hindari penggunaan kapas berserat yang dapat menempel pada luka basah.',
            'Segera konsultasikan ke fasilitas medis / dokter untuk pemberian salep antibiotik topikal.'
        ],
        'emergency_alert': False
    },
    'Class3': {
        'grade': 'GRADE 3',
        'title': '3rd Degree Burn (Full Thickness)',
        'severity': 'Severe Emergency',
        'badge_color': '#f43f5e',
        'depth': 'Seluruh Lapisan Dermis hingga Jaringan Subkutan',
        'description': 'Luka bakar dalam yang menghancurkan seluruh ketebalan kulit dan folikel rambut. Kulit tampak putih lilin, kasar, kecokelatan, atau hangus kehitaman. Mungkin tidak terasa sakit karena ujung saraf telah rusak.',
        'healing_time': 'Membutuhkan penanganan bedah (debridement & skin graft) dan rehabilitasi panjang.',
        'protocols': [
            'HUBUNGI NOMOR GAWAT DARURAT (118/119) ATAU KE IGD SEGERA!',
            'Jangan melepaskan pakaian yang melekat atau menempel erat pada luka bakar.',
            'Jangan merendam area luka dalam air es, karena berisiko memicu hipotermia dan syok.',
            'Tutup area luka secara longgar dengan kain atau perban steril yang kering dan bersih.',
            'Tinggikan posisi ekstremitas yang terluka di atas posisi jantung jika memungkinkan.'
        ],
        'emergency_alert': True
    },
    'Class4': {
        'grade': 'GRADE 4',
        'title': '4th Degree Burn (Subdermal / Deep)',
        'severity': 'Critical Trauma',
        'badge_color': '#9333ea',
        'depth': 'Menembus Subkutan, Lemak, Tendon, Otot & Tulang',
        'description': 'Kerusakan jaringan tingkat paling parah dan mengancam jiwa. Jaringan tampak hangus, kaku, dan mengekspos struktur anatomi di bawahnya seperti otot atau tulang.',
        'healing_time': 'Penanganan intensif di Unit Perawatan Luka Bakar (Burn Center), bedah rekonstruksi berulang.',
        'protocols': [
            'KONDISI KRITIS: SEGERA BAWA KE UNIT GAWAT DARURAT / RUMAH SAKIT RUJUKAN LUKA BAKAR!',
            'Pastikan jalan napas (airway), pernapasan, dan sirkulasi pasien tetap stabil.',
            'Lindungi pasien dari hipotermia dengan selimut bersih di area non-luka.',
            'Jangan beri makan atau minum jika pasien menunjukkan tanda-tanda syok atau penurunan kesadaran.',
            'Dokumentasikan waktu dan zat penyebab trauma (api, listrik tegangan tinggi, kimia asam/basa kuat).'
        ],
        'emergency_alert': True
    }
}

CLASS_LABELS = ['Class1', 'Class2', 'Class3', 'Class4']

def run_segmentation(image_path, output_dir=None):
    """
    Menjalankan Computer Vision Preprocessing:
    - Segmentasi warna kulit (HSV color space).
    - Thresholding Otsu untuk mengisolasi jaringan luka bakar.
    - Ekstraksi kontur luka dan pembuatan masker biner serta overlay neon.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]

    # Konversi ke HSV untuk isolasi area kulit
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_area = cv2.bitwise_and(image, image, mask=skin_mask)

    # Deteksi kontur luka menggunakan Otsu thresholding pada area kulit
    gray = cv2.cvtColor(skin_area, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter kontur kecil untuk menghilangkan noise
    min_area = (h * w) * 0.0005
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Masker luka bakar
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, valid_contours, -1, 255, -1)

    # Overlay citra dengan garis kontur neon bio-cyan dan highlight semi-transparan
    overlay = image.copy()
    cv2.drawContours(overlay, valid_contours, -1, (255, 230, 0), 2)  # Garis tepi kontur cyan/amber
    
    # Hitung estimasi persentase area luka
    burn_pixels = int(np.count_nonzero(mask))
    total_pixels = h * w
    skin_pixels = int(np.count_nonzero(skin_mask))
    
    burn_percentage_total = round((burn_pixels / total_pixels) * 100, 2)
    burn_percentage_skin = round((burn_pixels / max(skin_pixels, 1)) * 100, 2)

    mask_filename = None
    overlay_filename = None

    if output_dir:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_filename = f"{base_name}_mask.png"
        overlay_filename = f"{base_name}_overlay.png"

        mask_full_path = os.path.join(output_dir, mask_filename)
        overlay_full_path = os.path.join(output_dir, overlay_filename)

        cv2.imwrite(mask_full_path, mask)
        cv2.imwrite(overlay_full_path, overlay)

    return {
        'mask_filename': mask_filename,
        'overlay_filename': overlay_filename,
        'contours_count': len(valid_contours),
        'burn_percentage': burn_percentage_total,
        'burn_percentage_skin': burn_percentage_skin,
        'image_dimensions': f"{w} x {h} px"
    }

def run_classification(image_path, model, target_size=(150, 150)):
    """
    Menjalankan inferensi Deep Learning CNN untuk memprediksi kelas luka bakar.
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_LABELS[predicted_idx]
    confidence = float(predictions[predicted_idx]) * 100

    probabilities = []
    for idx, label in enumerate(CLASS_LABELS):
        prob_val = float(predictions[idx]) * 100
        info = CLINICAL_INFO[label]
        probabilities.append({
            'class': label,
            'grade': info['grade'],
            'title': info['title'],
            'severity': info['severity'],
            'color': info['badge_color'],
            'probability': round(prob_val, 2),
            'is_predicted': (idx == predicted_idx)
        })

    # Urutkan probabilitas dari yang tertinggi
    probabilities_sorted = sorted(probabilities, key=lambda x: x['probability'], reverse=True)

    clinical = CLINICAL_INFO[predicted_class]

    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'clinical': clinical,
        'probabilities': probabilities,
        'probabilities_sorted': probabilities_sorted
    }

def analyze_burn_image(image_path, model, output_dir):
    """
    Pipeline lengkap: Menggabungkan Computer Vision (segmentasi) & Deep Learning (klasifikasi)
    """
    segmentation_result = run_segmentation(image_path, output_dir)
    classification_result = run_classification(image_path, model)

    return {
        'segmentation': segmentation_result,
        'classification': classification_result
    }
