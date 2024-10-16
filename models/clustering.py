import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def process_image_clustering(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_normalized = image_resized / 255.0

    # Extract features
    features = extract_pixel_features(image_normalized)

    # Perform clustering (e.g., K-Means with k=5)
    k = 5
    cluster_labels, centroids = k_means(features, k)

    # Create the clustered image
    clustered_image = centroids[cluster_labels].reshape(image_resized.shape)

    # Save the clustering result temporarily
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.imsave(tmp_file.name, clustered_image)
        return tmp_file.name

def extract_pixel_features(image):
    h, w, c = image.shape
    RGB_values = image.reshape(-1, 3)
    return RGB_values

def k_means(features, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(features, k)
    for _ in range(max_iters):
        centroids_old = centroids.copy()
        cluster_labels = assign_clusters(features, centroids)
        centroids = update_centroids(features, cluster_labels, k)
        if np.linalg.norm(centroids - centroids_old) < tol:
            break
    return cluster_labels, centroids

def initialize_centroids(features, k):
    indices = np.random.choice(features.shape[0], k, replace=False)
    return features[indices]

def assign_clusters(features, centroids):
    distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(features, cluster_labels, k):
    return np.array([features[cluster_labels == i].mean(axis=0) for i in range(k)])

# # Example usage
# image_path = 'your_image.png'  # Path to your input image
# result_path = process_image_clustering(image_path)
# print(f"Clustered image saved at: {result_path}")