import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

def detect_single_leaf_disc(image_path):
    """
    Detects a diseased region in a single leaf disc using HSV masking and contour detection.
    Displays intermediate steps in a result figure and calculates disease percentage for the disc.
    Saves intermediate images after each transformation.
    Adds debugging for HSV values of sampled pixels and ensures edge detection accuracy.
    Additionally, visually highlights healthy vs diseased pixels and overlays diseased areas on the original image.
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
    hsv_image_path = os.path.join(output_dir, "hsv_image.png")
    cv2.imwrite(hsv_image_path, hsv)

    print("Step 3: Creating color masks for leaf and disease regions...")
    # Define HSV ranges for green (healthy leaf) and brown (diseased area)
    lower_green = np.array([36, 50, 50])  # Adjusted lower bounds for green
    upper_green = np.array([86, 255, 255])  # Adjusted upper bounds for green
    lower_brown = np.array([10, 40, 20])  # Adjusted lower bounds for brown
    upper_brown = np.array([20, 200, 150])  # Adjusted upper bounds for brown

    # Create masks for green and brown regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Apply morphological operations to clean the brown mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)

    # Save intermediate masks
    green_mask_path = os.path.join(output_dir, "green_mask.png")
    brown_mask_path = os.path.join(output_dir, "brown_mask.png")
    cv2.imwrite(green_mask_path, green_mask)
    cv2.imwrite(brown_mask_path, brown_mask)

    # Combine masks to isolate the leaf disc
    leaf_mask = cv2.bitwise_or(green_mask, brown_mask)
    leaf_mask_path = os.path.join(output_dir, "leaf_mask.png")
    cv2.imwrite(leaf_mask_path, leaf_mask)

    print("Step 4: Detecting contour for the largest leaf disc...")
    # Apply Gaussian blur to refine edges
    leaf_mask_blur = cv2.GaussianBlur(leaf_mask, (5, 5), 0)
    contours, _ = cv2.findContours(leaf_mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No leaf disc detected. Check your image or color ranges.")
        return

    # Assume the largest contour is the leaf disc
    largest_contour = max(contours, key=cv2.contourArea)
    mask_leaf = np.zeros_like(leaf_mask)
    cv2.drawContours(mask_leaf, [largest_contour], -1, 255, thickness=cv2.FILLED)

    mask_leaf_path = os.path.join(output_dir, "leaf_contour_mask.png")
    cv2.imwrite(mask_leaf_path, mask_leaf)

    print("Step 5: Calculating disease percentage...")
    # Isolate diseased area within the leaf disc
    diseased_area = cv2.bitwise_and(brown_mask, brown_mask, mask=mask_leaf)
    diseased_area_path = os.path.join(output_dir, "diseased_area.png")
    cv2.imwrite(diseased_area_path, diseased_area)

    # Calculate percentage of diseased pixels
    total_pixels = np.sum(mask_leaf > 0)
    diseased_pixels = np.sum(diseased_area > 0)

    # Debugging: Print pixel counts
    print(f"Total Pixels in Leaf Disc: {total_pixels}")
    print(f"Diseased Pixels: {diseased_pixels}")

    # Calculate correct percentage
    disease_percentage = (diseased_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    print(f"Corrected Disease Percentage: {disease_percentage:.2f}%")

    print("Step 6: Visualizing results...")
    # Highlight healthy vs diseased pixels
    visualization = np.zeros_like(mask_leaf, dtype=np.uint8)
    visualization[mask_leaf > 0] = 1  # Healthy area
    visualization[diseased_area > 0] = 2  # Diseased area

    # Convert visualization to RGB for display
    visualization_rgb = np.zeros((*visualization.shape, 3), dtype=np.uint8)
    visualization_rgb[visualization == 1] = [0, 255, 0]  # Green for healthy
    visualization_rgb[visualization == 2] = [0, 0, 255]  # Red for diseased

    # Ensure diseased areas align correctly with the detected leaf disc
    aligned_overlay = cv2.bitwise_and(visualization_rgb, visualization_rgb, mask=mask_leaf)

    # Overlay diseased areas on the original image
    overlay_image = image.copy()
    diseased_overlay = cv2.addWeighted(overlay_image, 0.7, aligned_overlay, 0.3, 0)
    overlay_image_path = os.path.join(output_dir, "overlay_image.png")
    cv2.imwrite(overlay_image_path, diseased_overlay)

    # Display the results
    plt.figure(figsize=(16, 12))

    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    plt.title("HSV Image")
    plt.axis("off")

    plt.subplot(3, 2, 2)
    plt.imshow(green_mask, cmap="gray")
    plt.title("Green Mask")
    plt.axis("off")

    plt.subplot(3, 2, 3)
    plt.imshow(brown_mask, cmap="gray")
    plt.title("Brown Mask")
    plt.axis("off")

    plt.subplot(3, 2, 4)
    plt.imshow(mask_leaf, cmap="gray")
    plt.title("Leaf Contour Mask")
    plt.axis("off")

    plt.subplot(3, 2, 5)
    plt.imshow(diseased_area, cmap="gray")
    plt.title("Diseased Area")
    plt.axis("off")

    plt.subplot(3, 2, 6)
    plt.imshow(cv2.cvtColor(diseased_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Diseased Overlay on Original")
    plt.axis("off")

    plt.tight_layout()
    final_figure_path = os.path.join(output_dir, "final_figure.png")
    plt.savefig(final_figure_path, dpi=300)
    plt.close()

    print(f"Saved intermediate results and final figure in: {output_dir}")
    print("Processing complete!")

# Main function to accept arguments
if __name__ == "__main__":
    print("Starting script...")
    if len(sys.argv) < 2:
        print("Usage: python disease_percent.py <image_path>")
    else:
        image_path = sys.argv[1]
        print(f"Input image: {image_path}")
        detect_single_leaf_disc(image_path)
    sys.exit(0)
