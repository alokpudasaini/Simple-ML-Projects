import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load and normalize the image
img = Image.open("example.jpg")  # Replace with your own image
img = img.resize((800, 600))     # Smaller for faster computation
img_data = np.array(img) / 255.0
original_shape = img_data.shape

# Flatten image to (m, 3)
X = img_data.reshape(-1, 3)
m, n = X.shape

def initialize_centroids(X, K):
    indices = np.random.choice(m, K, replace=False)
    return X[indices]

def find_closest_centroids(X, centroids):
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distances)
    return idx

def compute_centroids(X, idx, K):
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0) if len(points) > 0 else centroids[k]
    return centroids

def run_kmeans(X, K, max_iters=10):
    centroids = initialize_centroids(X, K)
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

K = 16  # Try with 8, 16, 32
centroids, idx = run_kmeans(X, K)
X_compressed = centroids[idx]

# Reshape back to original image
compressed_img = X_compressed.reshape(original_shape)

# Show original and compressed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_data)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title(f"Compressed Image (K={K})")

plt.show()
