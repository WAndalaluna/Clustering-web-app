import streamlit as st
from PIL import Image
import numpy as np
import cv2
from models.clustering import process_image_clustering, process_dbscan_clustering

# Judul aplikasi
st.title('Aplikasi Clustering Citra dengan K-Means dan DBSCAN')

# Opsi metode clustering
method = st.sidebar.selectbox(
    'Pilih Metode Clustering',
    ('K-Means', 'DBSCAN')
)

# Jumlah cluster (untuk K-Means) atau epsilon/min_samples (untuk DBSCAN)
if method == 'K-Means':
    k = st.sidebar.slider('Jumlah Cluster (K)', 2, 10, 5)
else:
    eps = st.sidebar.slider('Nilai Epsilon (eps)', 0.01, 0.2, 0.05)
    min_samples = st.sidebar.slider('Min Samples', 1, 10, 5)

# File uploader untuk upload gambar
uploaded_file = st.file_uploader("Upload gambar untuk clustering", type=["jpg", "jpeg", "png"])

# Jika file diupload, proses clustering
if uploaded_file is not None:
    # Membaca gambar menggunakan PIL dan konversi ke array numpy
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Menampilkan gambar asli
    st.image(image_array, caption='Gambar Asli', use_column_width=True)

    # Pilih metode clustering
    if method == 'K-Means':
        st.write(f"Clustering menggunakan K-Means dengan K={k}")
        clustered_image_path = process_image_clustering(image_array, k=k)
    else:
        st.write(f"Clustering menggunakan DBSCAN dengan eps={eps} dan min_samples={min_samples}")
        clustered_image_path = process_dbscan_clustering(image_array, eps=eps, min_samples=min_samples)

    # Menampilkan hasil clustering
    if clustered_image_path:
        clustered_image = Image.open(clustered_image_path)
        st.image(clustered_image, caption='Hasil Clustering', use_column_width=True)

# Informasi tambahan
st.sidebar.markdown("## Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini melakukan segmentasi citra menggunakan dua metode clustering:
- **K-Means**
- **DBSCAN**
""")
