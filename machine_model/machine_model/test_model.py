import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype('float32') / 255.0

# --- FUNGSI predict_image SUDAH DIHAPUS ---

if __name__ == "__main__":
    
    print("Mencoba memuat model kustom...")
    project_root = Path(__file__).resolve().parents[2]
    model = load_model(project_root / 'model_terbaik.h5')
    print("Model berhasil dimuat.")
    
    # URUTAN INI SUDAH DISESUAIKAN DENGAN OUTPUT ANDA
    custom_labels = ['Beer', 'Glass cup', 'HDPE', 'Mug', 'Other', 'PET', 'Plastic cup', 'Soda', 'Tin', 'Tumbler', 'Wine']

    while True:
        root = tk.Tk()
        root.withdraw()

        print("\n----------------------------------------------------")
        print("Silakan pilih file gambar melalui jendela yang muncul...")
        image_path = filedialog.askopenfilename(
            title="Pilih sebuah file gambar",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.avif")]
        )

        if image_path:
            try:
                processed_image = preprocess_image(image_path)
                
                # --- LOGIKA PREDIKSI YANG BENAR UNTUK MODEL KUSTOM ---
                predictions = model.predict(processed_image)[0]
                predicted_index = np.argmax(predictions)
                predicted_label = custom_labels[predicted_index]
                predicted_score = predictions[predicted_index]

                print(f"\n--- Hasil Deteksi untuk Gambar: {image_path} ---")
                print(f"===> Dideteksi sebagai: '{predicted_label}' (skor: {predicted_score:.2%}) <===")
                    
            except Exception as e:
                print(f"\nERROR: Terjadi masalah saat memproses file: {e}")
        else:
            print("\nTidak ada file yang dipilih.")

        lanjut = input("\nProses gambar lain? (y/n): ")
        if lanjut.lower() == 'n':
            break 
            
    print("\nTerima kasih telah menggunakan program ini!")