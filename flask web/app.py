import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

# Matikan log berlebih dari TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'burn_classification_model_fix.h5')

if not os.path.exists(MODEL_PATH):
    parent_model = os.path.join(os.path.dirname(BASE_DIR), 'burn_classification_model_fix.h5')
    if os.path.exists(parent_model):
        MODEL_PATH = parent_model

# Pastikan folder uploads tersedia
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Load pipeline helper
import pipeline

print(f"Loading deep learning model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Sampel kasus klinis tervalidasi untuk pengujian 1-klik
SAMPLE_CASES = [
    {
        'id': 'sample_grade_1',
        'title': 'Grade 1 Case',
        'subtitle': 'Superficial Erythema',
        'filename': 'img549.jpg',
        'expected_grade': 'GRADE 1',
        'color': '#10b981'
    },
    {
        'id': 'sample_grade_2',
        'title': 'Grade 2 Case',
        'subtitle': 'Partial Thickness Blister',
        'filename': 'google_0031.jpg',
        'expected_grade': 'GRADE 2',
        'color': '#f59e0b'
    },
    {
        'id': 'sample_grade_3',
        'title': 'Grade 3 Case',
        'subtitle': 'Full Thickness Eschar',
        'filename': '3.jpg',
        'expected_grade': 'GRADE 3',
        'color': '#f43f5e'
    },
    {
        'id': 'sample_grade_4',
        'title': 'Grade 4 Case',
        'subtitle': 'Subdermal Tissue Trauma',
        'filename': 'google_0111.jpg',
        'expected_grade': 'GRADE 4',
        'color': '#9333ea'
    }
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    """Halaman utama konsol diagnostik."""
    return render_template('index.html', samples=SAMPLE_CASES)

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('index.html', samples=SAMPLE_CASES)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Menyajikan file citra asli, mask, maupun overlay kontur."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Endpoint untuk mengambil daftar sampel gambar tervalidasi."""
    return jsonify({
        'status': 'success',
        'samples': SAMPLE_CASES
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Endpoint utama AJAX pipeline analisis luka bakar.
    Menerima: file upload ATAU json {'sample': 'filename.jpg'}
    Mengembalikan: Hasil segmentasi computer vision + klasifikasi CNN + protokol klinis
    """
    filepath = None
    filename = None

    # Cek jika input berasal dari pemilihan sampel
    if request.is_json:
        data = request.get_json() or {}
        sample_name = data.get('sample')
        if sample_name:
            candidate_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(sample_name))
            if os.path.exists(candidate_path):
                filepath = candidate_path
                filename = secure_filename(sample_name)

    # Cek jika input berasal dari upload multipart file
    if not filepath and 'file' in request.files:
        file = request.files['file']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

    if not filepath or not os.path.exists(filepath):
        return jsonify({
            'status': 'error',
            'message': 'File gambar tidak valid atau tidak ditemukan.'
        }), 400

    try:
        # Jalankan pipeline lengkap (OpenCV segmentation + Keras CNN classification)
        result = pipeline.analyze_burn_image(filepath, model, app.config['UPLOAD_FOLDER'])
        
        seg = result['segmentation']
        cls = result['classification']

        return jsonify({
            'status': 'success',
            'filename': filename,
            'original_url': url_for('uploaded_file', filename=filename),
            'mask_url': url_for('uploaded_file', filename=seg['mask_filename']) if seg and seg['mask_filename'] else None,
            'overlay_url': url_for('uploaded_file', filename=seg['overlay_filename']) if seg and seg['overlay_filename'] else None,
            'segmentation': seg,
            'classification': cls
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Gagal menganalisis citra: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_form_fallback():
    """Fallback SSR untuk submit formulir tradisional non-JS."""
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = pipeline.analyze_burn_image(filepath, model, app.config['UPLOAD_FOLDER'])

    return render_template(
        'index.html',
        samples=SAMPLE_CASES,
        ssr_result={
            'filename': filename,
            'original_url': url_for('uploaded_file', filename=filename),
            'mask_url': url_for('uploaded_file', filename=result['segmentation']['mask_filename']),
            'overlay_url': url_for('uploaded_file', filename=result['segmentation']['overlay_filename']),
            'segmentation': result['segmentation'],
            'classification': result['classification']
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
