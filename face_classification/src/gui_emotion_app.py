import os
import sys
import time
import threading
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Menentukan lokasi folder project berdasarkan lokasi file ini.
# Path absolut digunakan agar model tetap ditemukan meskipun program
# dijalankan dari folder kerja yang berbeda.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"

# Kelas utama aplikasi. Seluruh tampilan dan proses deteksi emosi
# dikendalikan dari kelas EmotionApp.
class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ============================================================
        # 1. KONFIGURASI JENDELA APLIKASI
        # ============================================================
        self.title("Emotion AI - NextGen")
        self.geometry("1280x820")
        self.minsize(1000, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Variabel yang menyimpan model dan informasi klasifikasi emosi.
        self.models_loaded = False
        self.face_detection = None
        self.emotion_classifier = None
        self.emotion_labels = None
        self.emotion_names = None
        self.emotion_target_size = None
        self.emotion_offsets = (10, 10)

        # Variabel untuk mengelola gambar, video, webcam, dan statistik.
        self.capture = None
        self.current_source = None
        self.running = False
        self.photo_image = None
        self.emotion_counter = Counter()

        # Membuat layout utama agar mengikuti ukuran jendela.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Aplikasi memiliki splash screen dan halaman utama.
        self.splash_frame = self._build_splash_screen()
        self.main_frame = self._build_main_layout()

        # Splash screen ditampilkan lebih dahulu saat model dimuat.
        self.splash_frame.grid(row=0, column=0, sticky="nsew")

        # Menjalankan proses pemuatan model setelah jendela muncul.
        self.after(500, self._start_loading_process)

        # Memastikan webcam dilepas saat jendela ditutup.
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # Membuat halaman pembuka dan progress bar pemuatan model.
    def _build_splash_screen(self):
        frame = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=0)

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(5, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(frame, text="EMOTION DETECTION", font=ctk.CTkFont(family="Segoe UI Black", size=72, weight="bold"), text_color="#38bdf8")
        title.grid(row=1, column=0, pady=(0, 10))

        subtitle = ctk.CTkLabel(frame, text="Next-Generation Facial Expression Analysis", font=ctk.CTkFont(family="Segoe UI", size=20), text_color="#94a3b8")
        subtitle.grid(row=2, column=0, pady=(0, 40))

        self.progress_bar = ctk.CTkProgressBar(frame, width=400, height=10, progress_color="#38bdf8", fg_color="#1e293b")
        self.progress_bar.grid(row=3, column=0, pady=20)
        self.progress_bar.set(0)

        self.loading_status = ctk.CTkLabel(frame, text="Initializing environment...", font=ctk.CTkFont(family="Segoe UI", size=14), text_color="#64748b")
        self.loading_status.grid(row=4, column=0)

        return frame

    # ================================================================
    # 2. MEMBANGUN ANTARMUKA UTAMA
    # ================================================================
    # Membuat tombol input, area preview, status, dan panel probabilitas.
    def _build_main_layout(self):
        frame = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=0)

        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0)

        top_bar = ctk.CTkFrame(frame, height=80, fg_color="#1e293b", corner_radius=0)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        btn_font = ctk.CTkFont(family="Segoe UI", size=14, weight="bold")

        ctk.CTkButton(top_bar, text="🖼️ Open Image", command=self.open_image, font=btn_font, fg_color="#0ea5e9", hover_color="#0284c7").pack(side="left", padx=15, pady=20)
        ctk.CTkButton(top_bar, text="🎬 Open Video", command=self.open_video, font=btn_font, fg_color="#0ea5e9", hover_color="#0284c7").pack(side="left", padx=15, pady=20)
        ctk.CTkButton(top_bar, text="📷 Start Webcam", command=self.start_webcam, font=btn_font, fg_color="#0ea5e9", hover_color="#0284c7").pack(side="left", padx=15, pady=20)
        ctk.CTkButton(top_bar, text="⏹️ Stop", command=self.stop_stream, font=btn_font, fg_color="#ef4444", hover_color="#dc2626").pack(side="left", padx=15, pady=20)

        self.status_var = ctk.StringVar(value="Ready. Load an image, video, or start webcam.")
        status_label = ctk.CTkLabel(top_bar, textvariable=self.status_var, font=ctk.CTkFont(family="Segoe UI", size=14), text_color="#cbd5e1")
        status_label.pack(side="right", padx=30, pady=20)

        preview_frame = ctk.CTkFrame(frame, fg_color="#0f172a", corner_radius=0)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(preview_frame, text="No Media Source", font=ctk.CTkFont(size=24, weight="bold"), text_color="#334155")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        sidebar = ctk.CTkFrame(frame, width=340, fg_color="#1e293b", corner_radius=15)
        sidebar.grid(row=1, column=1, sticky="ns", padx=(0, 20), pady=20)
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="Detection Analysis", font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"), text_color="#f8fafc").pack(anchor="w", padx=20, pady=(20, 15))

        self.source_var = ctk.StringVar(value="Source: -")
        self.dominant_var = ctk.StringVar(value="Dominant: -")
        self.faces_var = ctk.StringVar(value="Faces: 0")
        self.frame_var = ctk.StringVar(value="Frame: -")

        stats_frame = ctk.CTkFrame(sidebar, fg_color="#334155", corner_radius=10)
        stats_frame.pack(fill="x", padx=20, pady=(0, 20))

        for var in (self.source_var, self.dominant_var, self.faces_var, self.frame_var):
            ctk.CTkLabel(stats_frame, textvariable=var, font=ctk.CTkFont(family="Consolas", size=13), text_color="#e2e8f0").pack(anchor="w", padx=15, pady=5)

        ctk.CTkLabel(sidebar, text="Live Probabilities", font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"), text_color="#f8fafc").pack(anchor="w", padx=20, pady=(10, 5))

        self.details_text = ctk.CTkTextbox(sidebar, font=ctk.CTkFont(family="Consolas", size=12), fg_color="#0f172a", text_color="#bae6fd", corner_radius=10, wrap="word")
        self.details_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.details_text.configure(state="disabled")

        return frame

    # Memuat model pada thread terpisah agar GUI tidak membeku.
    def _start_loading_process(self):
        thread = threading.Thread(target=self._load_models_worker)
        thread.daemon = True
        thread.start()
        self._animate_progress(0)

    # Menganimasikan progress bar sampai model selesai dimuat.
    def _animate_progress(self, current_val):
        if not self.models_loaded:
            if current_val < 0.95:
                current_val += 0.02
                self.progress_bar.set(current_val)
            self.after(50, self._animate_progress, current_val)
        else:
            self.progress_bar.set(1.0)
            self.loading_status.configure(text="Ready to go!", text_color="#4ade80")
            self.after(800, self._transition_to_main)

    # ================================================================
    # 3. MEMUAT MODEL DETEKSI DAN KLASIFIKASI
    # ================================================================
    def _load_models_worker(self):
        try:
            # Import TensorFlow/Keras dilakukan di sini agar splash screen
            # dapat tampil sebelum proses import yang cukup berat.
            self._update_status_safe("Importing TensorFlow/Keras...")
            global load_model, get_labels, apply_offsets, detect_faces, load_detection_model, preprocess_input
            from keras.models import load_model
            from utils.datasets import get_labels
            from utils.inference import apply_offsets
            from utils.inference import detect_faces
            from utils.inference import load_detection_model
            from utils.preprocessor import preprocess_input

            self._update_status_safe("Loading Neural Networks...")

            # Haar Cascade mendeteksi lokasi wajah, sedangkan model HDF5
            # mengklasifikasikan ekspresi wajah ke dalam tujuh emosi.
            detection_model_path = TRAINED_MODELS_DIR / "detection_models" / "haarcascade_frontalface_default.xml"
            emotion_model_path = TRAINED_MODELS_DIR / "emotion_models" / "fer2013_mini_XCEPTION.102-0.66.hdf5"

            # Program dihentikan dengan pesan yang jelas jika model hilang.
            if not detection_model_path.exists() or not emotion_model_path.exists():
                raise FileNotFoundError("Missing model files.")

            # Memuat model dan membaca ukuran input yang dibutuhkan CNN.
            self.face_detection = load_detection_model(str(detection_model_path))
            self.emotion_classifier = load_model(str(emotion_model_path), compile=False)
            self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

            # Mengambil mapping label FER2013:
            # angry, disgust, fear, happy, sad, surprise, neutral.
            self.emotion_labels = get_labels("fer2013")
            self.emotion_names = [self.emotion_labels[index] for index in sorted(self.emotion_labels)]

            self._update_status_safe("Finalizing setup...")
            time.sleep(0.5)

            self.models_loaded = True

        except Exception as e:
            self._update_status_safe(f"Error loading models: {e}")

    # Memperbarui teks GUI secara aman dari thread pemuatan model.
    def _update_status_safe(self, msg):
        self.after(0, lambda: self.loading_status.configure(text=msg))

    def _transition_to_main(self):
        self.splash_frame.grid_forget()
        self.main_frame.grid(row=0, column=0, sticky="nsew")

    # ================================================================
    # 4. MEMILIH SUMBER MEDIA
    # ================================================================
    # Membuka satu gambar dan langsung menjalankan deteksi emosi.
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not path:
            return
        self.stop_stream()
        self.current_source = Path(path)
        self.emotion_counter.clear()
        frame = cv2.imread(str(self.current_source))
        if frame is None:
            messagebox.showerror("Error", f"Unable to open image:\n{self.current_source}")
            return
        annotated_frame, detections = self.annotate_frame(frame, smooth=False)
        self.update_summary(detections, 0)
        self.show_frame(annotated_frame)
        self.status_var.set(f"Image loaded: {self.current_source.name}")
        self.source_var.set(f"Source: {self.current_source.name}")

    # Membuka video dan memproses frame secara berurutan.
    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v")],
        )
        if not path:
            return
        self.stop_stream()
        self.current_source = Path(path)
        self.capture = cv2.VideoCapture(str(self.current_source))
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror("Error", f"Unable to open video:\n{self.current_source}")
            return
        self.emotion_counter.clear()
        self.running = True
        self.status_var.set(f"Playing video: {self.current_source.name}")
        self.source_var.set(f"Source: {self.current_source.name}")
        self._next_frame()

    # Membuka webcam default pada indeks kamera 0.
    def start_webcam(self):
        self.stop_stream()
        self.current_source = "Webcam"
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror("Error", "Unable to open default camera (index 0).")
            return
        self.emotion_counter.clear()
        self.running = True
        self.status_var.set("Webcam started. Press Stop to end the stream.")
        self.source_var.set("Source: Webcam")
        self._next_frame()

    # Menghentikan video/webcam dan melepaskan resource kamera.
    def stop_stream(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame_var.set("Frame: -")
        self.status_var.set("Stream stopped.")

    # Membaca satu frame, memprosesnya, kemudian menjadwalkan frame berikutnya.
    def _next_frame(self):
        if not self.running or self.capture is None:
            return

        success, frame = self.capture.read()
        if not success:
            self.stop_stream()
            self.status_var.set("Stream ended.")
            return

        frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        annotated_frame, detections = self.annotate_frame(frame, smooth=True)
        self.update_summary(detections, frame_index)
        self.show_frame(annotated_frame)
        self.after(15, self._next_frame)

    # ================================================================
    # 5. PIPELINE DETEKSI EMOSI
    # ================================================================
    def annotate_frame(self, frame, smooth):
        # RGB digunakan untuk tampilan, sedangkan grayscale digunakan
        # oleh Haar Cascade dan model klasifikasi emosi.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mendeteksi seluruh koordinat wajah pada frame.
        faces = detect_faces(self.face_detection, gray_image)
        detections = []

        # Setiap wajah dipotong dan diprediksi secara terpisah.
        for face_coordinates in faces:
            # Menambahkan offset agar area crop tidak terlalu sempit.
            x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(gray_image.shape[1], x2)
            y2 = min(gray_image.shape[0], y2)
            gray_face = gray_image[y1:y2, x1:x2]

            # Menyesuaikan ukuran wajah dengan input model.
            try:
                gray_face = cv2.resize(gray_face, self.emotion_target_size)
            except cv2.error:
                continue

            # Histogram equalization membantu saat memproses video/webcam.
            if smooth:
                gray_face = cv2.equalizeHist(gray_face)

            # Normalisasi piksel ke rentang -1 sampai 1 dan menambahkan
            # dimensi batch serta channel: (1, tinggi, lebar, 1).
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            # Model menghasilkan probabilitas untuk seluruh kelas emosi.
            prediction = self.emotion_classifier(gray_face, training=False)[0].numpy()

            # argmax memilih emosi dengan probabilitas tertinggi.
            emotion_index = int(np.argmax(prediction))
            emotion = self.emotion_labels[emotion_index]
            score = float(np.max(prediction))
            if smooth:
                self.emotion_counter.update([emotion])

            # Menggambar sudut bounding box dan label confidence.
            x, y, w, h = [int(value) for value in face_coordinates]
            color = self._emotion_color(emotion, score)

            thickness = 3
            length = 25

            cv2.line(rgb_image, (x, y), (x + length, y), color, thickness)
            cv2.line(rgb_image, (x, y), (x, y + length), color, thickness)

            cv2.line(rgb_image, (x + w, y), (x + w - length, y), color, thickness)
            cv2.line(rgb_image, (x + w, y), (x + w, y + length), color, thickness)

            cv2.line(rgb_image, (x, y + h), (x + length, y + h), color, thickness)
            cv2.line(rgb_image, (x, y + h), (x, y + h - length), color, thickness)

            cv2.line(rgb_image, (x + w, y + h), (x + w - length, y + h), color, thickness)
            cv2.line(rgb_image, (x + w, y + h), (x + w, y + h - length), color, thickness)

            overlay = rgb_image.copy()
            cv2.rectangle(overlay, (x, y - 35), (x + 160, y), color, -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, rgb_image, 1 - alpha, 0, rgb_image)

            cv2.putText(
                rgb_image,
                f"{emotion.upper()} {int(score*100)}%",
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            top_scores = sorted(
                zip(self.emotion_names, prediction),
                key=lambda item: float(item[1]),
                reverse=True,
            )

            # Menyimpan hasil terstruktur untuk panel analisis GUI.
            detections.append(
                {
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "emotion": emotion,
                    "score": score,
                    "scores": [(label, float(value)) for label, value in top_scores],
                }
            )

        return rgb_image, detections

    # ================================================================
    # 6. MENAMPILKAN HASIL ANALISIS
    # ================================================================
    # Memperbarui jumlah wajah, frame, emosi dominan, dan probabilitas.
    def update_summary(self, detections, frame_index):
        dominant = self.emotion_counter.most_common(1)[0][0] if self.emotion_counter else "-"
        if not self.emotion_counter and detections:
            dominant = Counter([item["emotion"] for item in detections]).most_common(1)[0][0]

        self.dominant_var.set(f"Dominant: {dominant.upper()}")
        self.faces_var.set(f"Faces: {len(detections)}")
        self.frame_var.set(f"Frame: {frame_index}")

        if not detections:
            self._set_details_text("No face detected in the current frame.")
            return

        lines = []
        for face_index, detection in enumerate(detections, start=1):
            box = detection["box"]
            lines.append(f"FACE {face_index}: {detection['emotion'].upper()} ({detection['score']*100:.1f}%)")
            lines.append(f"Location: x={box['x']}, y={box['y']}, w={box['width']}, h={box['height']}")
            lines.append("-" * 30)
            for label, value in detection["scores"]:
                bar_len = int(value * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"{label.capitalize():<10} |{bar}| {value*100:5.1f}%")
            lines.append("=" * 30)
            lines.append("")
        self._set_details_text("\n".join(lines).strip())

    # Mengubah array NumPy menjadi gambar yang dapat ditampilkan CustomTkinter.
    def show_frame(self, rgb_image):
        display_width = 850
        display_height = 600
        image = Image.fromarray(rgb_image)
        image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)

        self.photo_image = ctk.CTkImage(light_image=image, dark_image=image, size=(image.width, image.height))
        self.preview_label.configure(image=self.photo_image, text="")

    # Mengisi panel teks probabilitas dalam mode read-only.
    def _set_details_text(self, text):
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", ctk.END)
        self.details_text.insert(ctk.END, text)
        self.details_text.configure(state="disabled")

    # Menentukan warna bounding box berdasarkan jenis emosi dan confidence.
    def _emotion_color(self, emotion, score):
        if emotion == "angry":
            base = np.asarray((239, 68, 68))
        elif emotion == "sad":
            base = np.asarray((59, 130, 246))
        elif emotion == "happy":
            base = np.asarray((234, 179, 8))
        elif emotion == "surprise":
            base = np.asarray((168, 85, 247))
        else:
            base = np.asarray((34, 197, 94))
        return tuple(int(value) for value in np.clip(base * max(score, 0.4), 0, 255))

    # Menutup aplikasi dengan aman.
    def on_close(self):
        self.stop_stream()
        self.destroy()

# ================================================================
# 7. TITIK MASUK PROGRAM
# ================================================================
def main():
    # Konfigurasi TCL/TK diperlukan oleh Tkinter pada beberapa instalasi Windows.
    python_home = Path(sys.base_prefix)
    tcl_root = python_home / "tcl"
    tcl_library = tcl_root / "tcl8.6"
    tk_library = tcl_root / "tk8.6"

    if tcl_library.exists():
        os.environ.setdefault("TCL_LIBRARY", str(tcl_library))
    if tk_library.exists():
        os.environ.setdefault("TK_LIBRARY", str(tk_library))

    app = EmotionApp()
    app.mainloop()

# Blok ini dijalankan hanya ketika file dibuka sebagai program utama.
if __name__ == "__main__":
    main()
