import os
import cv2
import numpy as np
import rawpy

def calculate_dynamic_range(image_path, bits_per_pixel=14, percentile_max=99.9, percentile_min=0.1):
    """
    Calculates the dynamic range of an image in stops.

    Args:
        image_path: Path to the image file.
        bits_per_pixel: Number of bits per pixel (default: 14 - typical for many Sony sensors)
        percentile_max: Percentile to use for maximum pixel value (default: 99.9)
        percentile_min: Percentile to use for minimum pixel value (default: 0.1)

    Returns:
        Dynamic range in stops, or None if an error occurs.
    """

    try:
        # Load the image
        if image_path.lower().endswith(('.arw', '.dng', '.nef', '.cr2', '.cr3', '.raf')):  # Check for common raw formats
            with rawpy.imread(image_path) as raw:
                if len(raw.raw_image.shape) == 2:  # Grayscale raw image
                    gray = raw.raw_image.copy()
                else:
                    rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # If the image is raw, the pixel values are typically in a high bit-depth (e.g., 12-bit or 14-bit)
        # Apply bilateral filter without downscaling, using high bit-depth
        gray_filtered = cv2.bilateralFilter(gray.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=7)

        # Find minimum and maximum non-zero pixel values
        min_val = np.percentile(gray_filtered[gray_filtered > 0], percentile_min)  # Use percentile to avoid noise
        max_val = np.percentile(gray_filtered, percentile_max)  # Use percentile to avoid outliers

        # Avoid division by zero
        if min_val == 0:
            min_val = 1e-10  # Small value to avoid division by zero

        # Calculate dynamic range in stops
        dynamic_range = 20 * np.log10(max_val / min_val)

        # Clamp dynamic range to a reasonable limit
        max_dynamic_range = 120  # Adjust this limit as needed
        dynamic_range = min(dynamic_range, max_dynamic_range)

        return dynamic_range

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".arw", ".dng", ".nef", ".cr2", ".cr3", ".raf"]

    for filename in os.listdir(current_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(current_dir, filename)
            dynamic_range = calculate_dynamic_range(image_path)

            if dynamic_range:
                print(f"Estimated Dynamic Range for {filename}: {dynamic_range:.2f}")
