import os
import cv2
import numpy as np
import csv
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd

def save_to_csv(data, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Laplacian Variance"])
        writer.writerows(data)

def calculate_laplacian_variance(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def process_images(folder_path, output_folder="./final_preprocessed", threshold=500):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = []  # To hold tuples of (image_name, laplacian_variance)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                variance = calculate_laplacian_variance(image_path)
                data.append((image_path, variance))
                # Check if the variance is higher than the threshold
                if variance > threshold:
                    # Define the output path
                    output_path = os.path.join(output_folder, os.path.basename(image_path))
                    # Save the image to the output folder
                    image = cv2.imread(image_path)
                    cv2.imwrite(output_path, image)
                    print(f"Saved: {output_path} with variance: {variance}")
    return data

def calculate_laplacian_variance(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

def find_similar_and_delete(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        images = []
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
                image_path = os.path.join(subdir, file)
                variance = calculate_laplacian_variance(image_path)
                images.append((image_path, variance))
        
        images.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                img1 = cv2.imread(images[i][0])
                img2 = cv2.imread(images[j][0])

                if img1.shape != img2.shape:
                    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
                
                sim_index = ssim(img1, img2, multichannel=True)
                
                if sim_index > 0.9:
                    os.remove(images[j][0])
                    print(f"Deleted {images[j][0]} due to similarity to {images[i][0]}")
                    break

def visualize_similar_images(image1_path, image2_path):
    image1 = cv2.cvtColor(cv2.imread(image1_path), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(image2_path), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title('Image 2')
    plt.axis('off')
    
    plt.show()

def find_similar_and_log(folder_path, csv_path="removed_images.csv", threshold=0.8):
    to_remove = []  # List to hold paths of images to remove

    for subdir, dirs, files in os.walk(folder_path):
        images = []
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
                image_path = os.path.join(subdir, file)
                try:
                    variance = calculate_laplacian_variance(image_path)
                    images.append((image_path, variance))
                except ValueError as e:
                    print(e)
                    continue

        images.sort(key=lambda x: x[1], reverse=True)

        for i in range(len(images)):
            for j in range(i+1, len(images)):
                img1 = cv2.imread(images[i][0])
                img2 = cv2.imread(images[j][0])

                if img1 is None or img2 is None:
                    print(f"Error reading one of the images: {images[i][0]} or {images[j][0]}")
                    continue

                try:
                    img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
                    img2 = cv2.resize(img2, (128, 128), interpolation=cv2.INTER_AREA)                 
                    sim_index = ssim(img1, img2, multichannel=True)
                    print("sim_index is ", sim_index)
                    if sim_index > threshold:
                        to_remove.append(images[j][0])
                        os.remove(images[j][0])  # Delete the image from the filesystem
                        visualize_similar_images(images[i][0], images[j][0])
                        break
                except ValueError as e:
                    print(f"SSIM ValueError for {images[i][0]} and {images[j][0]}: {e}")
                    continue