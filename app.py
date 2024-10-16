import streamlit as st
import os
import numpy as np
import cv2
from models.clustering import process_image_clustering

# Setup upload directory
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("Aplikasi Clustering Citra Udara")
st.write("Upload citra udara untuk melakukan segmentasi dan clustering.")

# Upload file
uploaded_file = st.file_uploader("Pilih citra untuk di-cluster", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Simpan gambar yang di-upload
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(filepath, caption='Citra yang di-upload', use_column_width=True)
    st.write("Clustering dalam proses...")

    # Proses clustering
    clustered_image = process_image_clustering(filepath)

    # Tampilkan hasil clustering
    st.image(clustered_image, caption='Hasil Clustering', use_column_width=True)
