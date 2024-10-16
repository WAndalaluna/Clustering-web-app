import numpy as np
import cv2
import matplotlib.pyplot as plt
import tempfile

def load_and_preprocess_image(image):
    # Mengubah dari BGR (OpenCV format) ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resizing citra agar memiliki ukuran yang standar
    image_resized = cv2.resize(image_rgb, (256, 256))
    # Normalisasi nilai piksel (range 0-1)
    image_normalized = image_resized / 255.0
    return image_normalized

def extract_pixel_features(image):
    h, w, c = image.shape
    # Membuat array koordinat X dan Y
    X_coords = np.repeat(np.arange(w), h).reshape(-1, 1)
    Y_coords = np.tile(np.arange(h), w).reshape(-1, 1)
    # Mendapatkan nilai RGB
    RGB_values = image.reshape(-1, 3)
    # Menggabungkan fitur warna dan spasial
    features = np.hstack((RGB_values, X_coords / w, Y_coords / h))  # Normalisasi koordinat spasial
    return features

def k_means(features, k, max_iters=100, tol=1e-4):
    # Inisialisasi centroid secara acak
    centroids = initialize_centroids(features, k)
    for i in range(max_iters):
        # Menyimpan centroid sebelumnya
        centroids_old = centroids.copy()
        # Assign clusters
        cluster_labels = assign_clusters(features, centroids)
        # Update centroid
        centroids = update_centroids(features, cluster_labels, k)
        # Mengecek konvergensi
        if np.linalg.norm(centroids - centroids_old) < tol:
            print(f"K-Means konvergen pada iterasi ke-{i}")
            break
    return cluster_labels, centroids

def initialize_centroids(features, k):
    # Inisialisasi centroid secara acak dari data
    indices = np.random.choice(features.shape[0], k, replace=False)
    centroids = features[indices]
    return centroids

def assign_clusters(features, centroids):
    # Menghitung jarak Euclidean antara fitur dan centroid
    distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
    # Menentukan cluster terdekat
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels

def update_centroids(features, cluster_labels, k):
    # Mengupdate posisi centroid berdasarkan rata-rata fitur dalam cluster
    new_centroids = np.array([features[cluster_labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def process_image_clustering(image, k=5):
    # Preprocess image
    image_resized = load_and_preprocess_image(image)
    # Ekstrak fitur piksel
    pixel_features = extract_pixel_features(image_resized)
    # Lakukan clustering K-Means
    cluster_labels, centroids = k_means(pixel_features, k)
    # Reshape cluster labels kembali ke bentuk citra
    clustered_image = cluster_labels.reshape(image_resized.shape[0], image_resized.shape[1])
    
    # Menyimpan citra hasil clustering sementara
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.imsave(tmp_file.name, clustered_image, cmap='nipy_spectral')
        return tmp_file.name

def dbscan(features, eps, min_samples):
    labels = np.full(features.shape[0], -1)  # Inisialisasi semua label sebagai noise (-1)
    cluster_id = 0
    for i in range(features.shape[0]):
        if labels[i] != -1:
            continue  # Sudah di-label
        neighbors = region_query(features, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1  # Tetap sebagai noise
        else:
            grow_cluster(features, labels, i, neighbors, cluster_id, eps, min_samples)
            cluster_id += 1
    return labels

def region_query(features, idx, eps):
    distances = np.linalg.norm(features - features[idx], axis=1)
    neighbors = np.where(distances < eps)[0]
    return neighbors

def grow_cluster(features, labels, idx, neighbors, cluster_id, eps, min_samples):
    labels[idx] = cluster_id
    i = 0
    while i < len(neighbors):
        point = neighbors[i]
        if labels[point] == -1:
            labels[point] = cluster_id
        elif labels[point] == -2:
            labels[point] = cluster_id
            point_neighbors = region_query(features, point, eps)
            if len(point_neighbors) >= min_samples:
                neighbors = np.concatenate((neighbors, point_neighbors))
        i += 1

def process_dbscan_clustering(image, eps=0.05, min_samples=5):
    # Preprocess image
    image_resized = load_and_preprocess_image(image)
    # Ekstrak fitur piksel
    pixel_features = extract_pixel_features(image_resized)
    # Melakukan clustering DBSCAN
    cluster_labels = dbscan(pixel_features, eps, min_samples)
    # Reshape hasil clustering ke bentuk citra
    clustered_image = cluster_labels.reshape(image_resized.shape[0], image_resized.shape[1])
    
    # Menyimpan citra hasil clustering sementara
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.imsave(tmp_file.name, clustered_image, cmap='nipy_spectral')
        return tmp_file.name
