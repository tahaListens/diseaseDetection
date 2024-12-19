import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def detect_disease_percentage(image_path):
    """
    Detects diseased regions and calculates the percentage for each leaf disc.
    Saves intermediate steps for analysis. Ensures only 25 circles are processed.
    Args:
        image_path: Path to the input image file.
    """
    print("Step 1: Reading the image...")
    # Step 1: Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open {image_path}.")
        return
    
    print("Creating output folder...")
    # Create output folder to save steps
    output_folder = "output_steps"
    os.makedirs(output_folder, exist_ok=True)

    print("Converting to RGB for visualization...")
    # Convert to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_folder, "01_original_image.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    print("Step 2: Converting to Grayscale...")
    # Step 2: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, "02_grayscale.png"), gray)

    print("Step 3: Applying thresholding to isolate diseased regions...")
    # Step 3: Thresholding to isolate diseased regions
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(output_folder, "03_thresholded.png"), thresholded)

    print("Step 4: Detecting circles (leaf discs) using Hough Circle Transform...")
    # Step 4: Detect circles (leaf discs) using Hough Circle Transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, 
                               param1=50, param2=40, minRadius=30, maxRadius=60)

    if circles is None:
        print("No circles (leaf discs) detected. Adjust parameters.")
        return

    # Ensure we only process the first 25 circles
    circles = np.uint16(np.around(circles[0]))
    if len(circles) > 25:
        print(f"Detected {len(circles)} circles. Limiting to the first 25 circles.")
        circles = circles[:25]
    
    disease_percentages = []
    result_image = image_rgb.copy()

    print("Step 5: Processing detected circles...")
    # Step 5: Create a single mask for all diseased areas
    combined_mask = np.zeros_like(gray)

    for i, circle in enumerate(circles):
        x, y, r = circle  # Circle center (x, y) and radius r
        print(f"Processing leaf disc {i + 1} at ({x}, {y}) with radius {r}...")
        
        # Create a mask for the current disc
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Calculate diseased region within the disc
        diseased_area = cv2.bitwise_and(thresholded, thresholded, mask=mask)
        total_area = np.pi * (r ** 2)
        diseased_pixels = np.sum(diseased_area > 0)
        percentage = (diseased_pixels / total_area) * 100
        disease_percentages.append(percentage)
        
        # Draw the circle and disease percentage on the result image
        cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
        cv2.putText(result_image, f"{percentage:.1f}%", (x - 20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print("Saving combined mask and final result image...")
    # Step 6: Save the combined mask and result image
    cv2.imwrite(os.path.join(output_folder, "04_combined_mask.png"), combined_mask)
    cv2.imwrite(os.path.join(output_folder, "05_final_result.png"), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    print("Displaying results...")
    # Display the result plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(combined_mask, cmap="gray")
    plt.title("Combined Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(result_image)
    plt.title("Final Result")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Printing disease percentages...")
    # Print disease percentages
    for i, percent in enumerate(disease_percentages):
        print(f"Leaf Disc {i + 1}: {percent:.2f}% diseased")

    print("Processing complete!")

# Main function to accept arguments
if __name__ == "__main__":
    print("Starting script...")
    if len(sys.argv) < 2:
        print("Usage: python disease_percent.py <image_path>")
    else:
        image_path = sys.argv[1]
        print(f"Input image: {image_path}")
        detect_disease_percentage(image_path)
