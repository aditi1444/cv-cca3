import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# CCA 3 - TASK 1B
# MORPHOLOGICAL PROCESSING AND SEGMENTATION
# ==========================================

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "dataset" / "cca 3.jpg"

OUTPUT_DIR = BASE_DIR / "outputs" / "unit3_segmentation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# STEP 1: LOAD ORIGINAL IMAGE
# ==========================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Check your dataset folder and image filename."
    )

# Convert BGR to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Image loaded successfully!")
print("Image dimensions:", image.shape)


# ==========================================
# STEP 2: BINARY THRESHOLDING
# ==========================================

# Otsu's thresholding automatically selects a threshold value
threshold_value, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Invert the binary image for an alternative foreground mask
binary_inverse = cv2.bitwise_not(binary)

print("Otsu threshold value:", threshold_value)


# ==========================================
# STEP 3: MORPHOLOGICAL OPERATIONS
# ==========================================

kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(
    binary,
    kernel,
    iterations=1
)

# Dilation
dilation = cv2.dilate(
    binary,
    kernel,
    iterations=1
)

# Opening: erosion followed by dilation
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

# Closing: dilation followed by erosion
closing = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)


# ==========================================
# STEP 4: CANNY EDGE DETECTION
# ==========================================

edges = cv2.Canny(
    gray,
    50,
    150
)


# ==========================================
# STEP 5: WATERSHED SEGMENTATION
# ==========================================

# Watershed works best when foreground objects
# can be distinguished from their background.

# Apply Otsu thresholding to the grayscale image.
# Choose the foreground polarity that occupies less
# than half the image, as a simple initial heuristic.

foreground = binary if np.mean(binary == 255) < 0.5 else binary_inverse

# Remove small foreground regions
sure_background = cv2.dilate(
    foreground,
    kernel,
    iterations=3
)

# Find likely foreground regions using distance transform
distance_transform = cv2.distanceTransform(
    foreground,
    cv2.DIST_L2,
    5
)

if distance_transform.max() > 0:
    _, sure_foreground = cv2.threshold(
        distance_transform,
        0.35 * distance_transform.max(),
        255,
        0
    )

    sure_foreground = sure_foreground.astype(np.uint8)

else:
    sure_foreground = np.zeros_like(gray)

# Unknown region lies between foreground and background
unknown = cv2.subtract(
    sure_background,
    sure_foreground
)

# Label connected foreground components
number_of_labels, markers = cv2.connectedComponents(
    sure_foreground
)

# Watershed marker convention:
# 0 = unknown; positive values = candidate regions
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
watershed_markers = cv2.watershed(
    image.copy(),
    markers.astype(np.int32)
)

# Mark watershed boundaries in red
watershed_result = image_rgb.copy()

watershed_result[watershed_markers == -1] = [255, 0, 0]


# ==========================================
# STEP 6: SAVE OUTPUT IMAGES
# ==========================================

def save_image(filename, output_image):
    """Save an image to the segmentation output folder."""

    output_path = OUTPUT_DIR / filename

    if output_image.ndim == 3:
        output_to_save = cv2.cvtColor(
            output_image,
            cv2.COLOR_RGB2BGR
        )
    else:
        output_to_save = output_image

    success = cv2.imwrite(
        str(output_path),
        output_to_save
    )

    if not success:
        raise IOError(f"Could not save: {output_path}")

    print("Saved:", filename)


save_image("01_original.png", image_rgb)

save_image("02_grayscale.png", gray)

save_image("03_binary_threshold.png", binary)

save_image("04_binary_inverse.png", binary_inverse)

save_image("05_erosion.png", erosion)

save_image("06_dilation.png", dilation)

save_image("07_opening.png", opening)

save_image("08_closing.png", closing)

save_image("09_canny_edges.png", edges)

save_image("10_watershed.png", watershed_result)


# ==========================================
# STEP 7: DISPLAY COMPARISON
# ==========================================

images = [
    image_rgb,
    gray,
    binary,
    erosion,
    dilation,
    opening,
    closing,
    edges,
    watershed_result
]

titles = [
    "Original Image",
    "Grayscale",
    "Binary Threshold",
    "Erosion",
    "Dilation",
    "Opening",
    "Closing",
    "Canny Edges",
    "Watershed Segmentation"
]

plt.figure(figsize=(16, 12))

for index, (output_image, title) in enumerate(
    zip(images, titles),
    start=1
):
    plt.subplot(3, 3, index)

    if output_image.ndim == 2:
        plt.imshow(output_image, cmap="gray")
    else:
        plt.imshow(output_image)

    plt.title(title)
    plt.axis("off")

plt.tight_layout()

comparison_path = OUTPUT_DIR / "11_segmentation_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# STEP 8: COMPLETION MESSAGE
# ==========================================

print("\n========================================")
print("TASK 1B COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nWatershed connected-component markers:", number_of_labels)

print("\nAll segmentation results saved in:")
print(OUTPUT_DIR.resolve())