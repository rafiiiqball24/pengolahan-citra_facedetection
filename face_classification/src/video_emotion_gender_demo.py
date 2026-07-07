# Fungsi mode digunakan untuk mengambil hasil prediksi yang paling sering muncul.
# Cara ini membuat label emosi dan gender lebih stabil antar-frame.
from statistics import mode

# OpenCV digunakan untuk mengakses webcam dan mengolah gambar.
import cv2
# load_model digunakan untuk memuat model CNN yang sudah dilatih.
from keras.models import load_model
# NumPy digunakan untuk mengubah bentuk array sebelum masuk ke model.
import numpy as np

# Mengimpor fungsi-fungsi bantuan dari folder utils.
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# ================================================================
# 1. KONFIGURASI MODEL DAN LABEL
# ================================================================
# Path model Haar Cascade untuk mendeteksi posisi wajah.
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
# Model mini-XCEPTION untuk mengklasifikasikan emosi wajah.
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
# Model CNN untuk mengklasifikasikan gender.
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
# Mengambil nama kelas emosi dari dataset FER2013.
emotion_labels = get_labels('fer2013')
# Mengambil nama kelas gender dari dataset IMDB.
gender_labels = get_labels('imdb')
# Jenis font yang digunakan untuk menulis hasil pada frame.
font = cv2.FONT_HERSHEY_SIMPLEX

# ================================================================
# 2. PARAMETER DETEKSI
# ================================================================
# Jumlah prediksi terakhir yang digunakan untuk menstabilkan hasil.
frame_window = 10
# Area gender dibuat lebih lebar karena model membutuhkan bagian wajah lebih luas.
gender_offsets = (30, 60)
# Area tambahan di sekitar wajah untuk klasifikasi emosi.
emotion_offsets = (20, 40)

# ================================================================
# 3. MEMUAT MODEL
# ================================================================
# Memuat Haar Cascade, model emosi, dan model gender ke memori.
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# Mengambil ukuran input yang diwajibkan oleh masing-masing model.
# Ukuran ini dipakai saat wajah di-resize sebelum proses prediksi.
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# List untuk menyimpan beberapa hasil prediksi terakhir.
gender_window = []
emotion_window = []

# ================================================================
# 4. MEMULAI WEBCAM
# ================================================================
# Membuat jendela output dan membuka webcam utama pada indeks 0.
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Loop berjalan terus sampai pengguna menekan tombol Q.
while True:

    # Membaca satu frame dari webcam dalam format BGR.
    bgr_image = video_capture.read()[1]
    # Grayscale digunakan untuk deteksi wajah dan klasifikasi emosi.
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # RGB digunakan untuk klasifikasi gender dan menggambar hasil.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # Mendeteksi seluruh wajah yang terdapat pada frame.
    faces = detect_faces(face_detection, gray_image)

    # Setiap wajah yang ditemukan diproses secara terpisah.
    for face_coordinates in faces:

        # Mengambil area wajah berwarna untuk prediksi gender.
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        # Mengambil area wajah grayscale untuk prediksi emosi.
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        # Menyesuaikan ukuran wajah dengan ukuran input masing-masing model.
        # Jika area wajah tidak valid, wajah tersebut dilewati.
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        # ============================================================
        # 5. PREDIKSI EMOSI
        # ============================================================
        # Normalisasi nilai piksel agar sesuai dengan input model.
        gray_face = preprocess_input(gray_face, False)
        # Menambahkan dimensi batch: (tinggi, lebar) menjadi
        # (1, tinggi, lebar, 1).
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        # Model menghasilkan probabilitas setiap emosi.
        # argmax memilih indeks dengan probabilitas tertinggi.
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        # Menyimpan hasil ke dalam riwayat prediksi emosi.
        emotion_window.append(emotion_text)

        # ============================================================
        # 6. PREDIKSI GENDER
        # ============================================================
        # Menambahkan dimensi batch dan melakukan normalisasi.
        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        # Menjalankan prediksi gender dan memilih hasil tertinggi.
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        # Menyimpan hasil ke dalam riwayat prediksi gender.
        gender_window.append(gender_text)

        # Membatasi riwayat agar hanya menyimpan 10 frame terakhir.
        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)

        # Mengambil label yang paling sering muncul dari beberapa frame.
        # Tujuannya agar tulisan tidak berubah terlalu cepat.
        try:
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
        except:
            continue

        # Menentukan warna kotak berdasarkan hasil prediksi gender.
        # Nilai warna OpenCV ditulis dalam urutan BGR/RGB sesuai gambar aktif.
        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        # ============================================================
        # 7. MENAMPILKAN HASIL
        # ============================================================
        # Menggambar kotak wajah, label gender, dan label emosi.
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    # Mengubah kembali RGB ke BGR karena imshow menggunakan format BGR.
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # Menampilkan frame yang sudah diberi kotak dan label.
    cv2.imshow('window_frame', bgr_image)

    # Tekan Q untuk menghentikan program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================================================================
# 8. MEMBERSIHKAN RESOURCE
# ================================================================
# Melepaskan webcam dan menutup seluruh jendela OpenCV.
video_capture.release()
cv2.destroyAllWindows()
