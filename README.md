# Aplikasi Clustering Citra dengan K-Means dan DBSCAN

Aplikasi ini merupakan proyek Streamlit yang melakukan segmentasi citra menggunakan dua metode clustering: K-Means dan DBSCAN. Pengguna dapat mengunggah gambar, memilih metode clustering, dan melihat hasil clustering dalam bentuk citra yang sudah tersegmentasi.

## Fitur Utama

- **K-Means Clustering**: Segmentasi citra dengan menentukan jumlah cluster (K).
- **DBSCAN Clustering**: Segmentasi citra berbasis kepadatan dengan parameter `eps` dan `min_samples`.
- **Silhouette Score**: Digunakan untuk mengevaluasi kualitas clustering secara keseluruhan (implementasi manual).
- **User Interface Sederhana**: Aplikasi dilengkapi dengan tampilan yang ramah pengguna menggunakan Streamlit.

## Teknologi yang Digunakan

- **Streamlit**: Framework untuk membangun aplikasi web interaktif berbasis Python.
- **OpenCV**: Digunakan untuk memproses gambar.
- **NumPy**: Untuk komputasi numerik.
- **Matplotlib**: Untuk menyimpan hasil citra clustering.
- **Pillow (PIL)**: Untuk membaca dan menampilkan gambar.

## Cara Menggunakan

1. **Instalasi Dependensi**: 
   - Pastikan Anda sudah menginstal `streamlit`, `opencv-python`, `numpy`, `matplotlib`, dan `Pillow` di lingkungan Python Anda. 
   - Anda bisa menggunakan `requirements.txt` atau perintah di bawah ini:
     ```bash
     pip install streamlit opencv-python numpy matplotlib Pillow
     ```

2. **Menjalankan Aplikasi**: 
   - Setelah dependensi terinstal, jalankan aplikasi Streamlit dengan perintah:
     ```bash
     streamlit run app.py
     ```

3. **Mengunggah Gambar**: 
   - Setelah aplikasi berjalan, unggah gambar dalam format `jpg`, `jpeg`, atau `png` melalui antarmuka aplikasi.
   
4. **Pilih Metode Clustering**: 
   - Pilih apakah Anda ingin menggunakan metode **K-Means** atau **DBSCAN** untuk melakukan clustering.
   - Jika memilih K-Means, atur jumlah cluster (K).
   - Jika memilih DBSCAN, atur nilai `eps` dan `min_samples`.

5. **Lihat Hasil Clustering**: 
   - Aplikasi akan menampilkan gambar asli dan gambar hasil clustering.

## Struktur Proyek

