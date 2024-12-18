import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def segment_and_detect_disease(image_path):
    """
    Detects diseased regions in leaf discs using a 5x5 grid segmentation approach.
    Saves a single result figure containing all intermediate steps.
    Args:
        image_path: Path to the input image file.
    """
    print("Step 1: Reading the image...")
    # Step 1: Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open {image_path}.")
        return
    
    # Get image name for saving the final figure
    image_name = os.path.basename(image_path).split('.')[0] + "_result.png"
    
    print("Step 2: Converting to RGB for visualization...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Step 3: Converting to Grayscale...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Step 4: Thresholding to isolate diseased regions...")
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    print("Step 5: Segmenting image into a 5x5 grid...")
    # Step 5: Divide the image into a 5x5 grid
    rows, cols = 5, 5
    height, width = gray.shape
    cell_height, cell_width = height // rows, width // cols

    disease_percentages = []
    result_image = image_rgb.copy()

    # Step 6: Process each cell in the grid
    for i in range(rows):
        for j in range(cols):
            print(f"Processing grid cell ({i+1}, {j+1})...")
            x_start, x_end = j * cell_width, (j + 1) * cell_width
            y_start, y_end = i * cell_height, (i + 1) * cell_height
            
            # Extract the cell
            cell = thresholded[y_start:y_end, x_start:x_end]
            cell_rgb = result_image[y_start:y_end, x_start:x_end]
            
            # Remove background by focusing on non-white regions
            diseased_pixels = np.sum(cell == 0)
            total_pixels = cell.size
            percentage = (diseased_pixels / total_pixels) * 100
            disease_percentages.append(percentage)
            
            # Draw a rectangle and annotate percentage
            cv2.rectangle(result_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(result_image, f"{percentage:.1f}%", (x_start + 5, y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print("Step 7: Saving the result figure...")
    # Display the results with all steps
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(thresholded, cmap="gray")
    plt.title("Thresholded Image")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(result_image)
    plt.title("Final Result with Grid and Percentages")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(image_name, dpi=300)
    plt.show()

    print(f"Saved result figure as: {image_name}")
    print("Disease percentages per grid cell:")
    for i, percent in enumerate(disease_percentages):
        print(f"Grid Cell {i + 1}: {percent:.2f}% diseased")

    print("Processing complete!")

# Main function to accept arguments
if __name__ == "__main__":
    print("Starting script...")
    if len(sys.argv) < 2:
        print("Usage: python disease_percent.py <image_path>")
    else:
        image_path = sys.argv[1]
        print(f"Input image: {image_path}")
        segment_and_detect_disease(image_path)
