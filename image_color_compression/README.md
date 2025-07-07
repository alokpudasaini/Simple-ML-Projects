# Image Color Compression using K-Means Clustering (from Scratch)

This project implements image color compression using the K-Means clustering algorithm built from scratch with NumPy.  
It reduces the number of colors in an image by clustering similar colors and replacing them with cluster centroids.

---

## Features

- Loads and normalizes an input image  
- Reshapes the image pixels for clustering  
- Implements K-Means clustering algorithm from scratch  
- Compresses the image colors to K clusters (e.g., 8, 16, 32)  
- Displays original and compressed images side-by-side  

---

## Requirements

- Python 3.x  
- NumPy  
- Pillow (PIL)  
- Matplotlib  

Install dependencies via:
pip install numpy pillow matplotlib


---

## Usage

1. Run the script:  
python image_color_compression.py


---

## How it Works

- The image is resized and normalized (values between 0 and 1)  
- Pixels are flattened into a 2D array where each row is a color (R,G,B)  
- K-Means clustering groups similar colors into K clusters  
- Each pixel is replaced with its cluster centroid color  
- The compressed image is reshaped back and displayed next to the original

---

## Experimentation

Try changing the number of clusters `K` (e.g., 8, 16, 32) to see the effect on compression quality and file size.

---

## Author

Alok Pudasaini  
Civil Engineer | ML Enthusiast | Researcher  

---

## Inspired By

Andrew Ng’s Machine Learning Specialization (K-Means Clustering lesson)

---
