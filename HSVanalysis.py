import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys

def detect_single_leaf_disc(image_path):
    """
    Detects a diseased region in a single leaf disc using HSV masking and contour detection.
    Displays intermediate steps in a result figure and calculates disease percentage for the disc.
    Visualizes HSV histograms for healthy and diseased regions to refine thresholds.
    Args:
        image_path: Path to the input image file.
    """
    print("Step 1: Reading the image...")
    # Step 1: Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open {image_path}.")
        return

    # Get image name for saving the final figure with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_name = os.path.basename(image_path).split('.')[0]
    output_dir = f"output_{base_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("Step 2: Converting to HSV color space...")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    print("Step 3: Creating color masks for leaf and disease regions...")
    # Define HSV ranges for green (healthy leaf) and brown (diseased area)
    lower_green = np.array([36, 50, 50])
    upper_green = np.array([86, 255, 255])
    lower_brown = np.array([10, 40, 20])
    upper_brown = np.array([20, 200, 150])

    # Create masks for green and brown regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combine masks to isolate the leaf disc
    leaf_mask = cv2.bitwise_or(green_mask, brown_mask)

    print("Step 4: Sampling HSV values for histograms...")
    # Extract HSV values for healthy and diseased areas
    healthy_hsv_values = hsv[green_mask > 0]
    diseased_hsv_values = hsv[brown_mask > 0]

    # Plot histograms for Hue, Saturation, and Value
    def plot_histograms(values, title, output_path):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(values[:, 0], bins=30, color='r', alpha=0.7)
        plt.title(f'{title} - Hue')
        plt.subplot(1, 3, 2)
        plt.hist(values[:, 1], bins=30, color='g', alpha=0.7)
        plt.title(f'{title} - Saturation')
        plt.subplot(1, 3, 3)
        plt.hist(values[:, 2], bins=30, color='b', alpha=0.7)
        plt.title(f'{title} - Value')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    healthy_hist_path = os.path.join(output_dir, "healthy_hsv_histograms.png")
    diseased_hist_path = os.path.join(output_dir, "diseased_hsv_histograms.png")
    
    plot_histograms(healthy_hsv_values, "Healthy Region HSV", healthy_hist_path)
    plot_histograms(diseased_hsv_values, "Diseased Region HSV", diseased_hist_path)

    print(f"Saved HSV histograms: {healthy_hist_path} and {diseased_hist_path}")

    # Additional processing and visualization can follow here...
    print("Processing complete!")

# Main function to accept arguments
if __name__ == "__main__":
    print("Starting script...")
    if len(sys.argv) < 2:
        print("Usage: python hsv_analysis.py <image_path>")
    else:
        image_path = sys.argv[1]
        print(f"Input image: {image_path}")
        detect_single_leaf_disc(image_path)
    sys.exit(0)
