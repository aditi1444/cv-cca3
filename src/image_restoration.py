import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from pathlib import Path


# ==========================================
# CCA 3 - TASK 1A: IMAGE RESTORATION
# ==========================================

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Corrected image filename
IMAGE_PATH = BASE_DIR / "dataset" / "cca 3.jpg"

OUTPUT_DIR = BASE_DIR / "outputs" / "unit3_restoration"

# Create output folder automatically
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# STEP 1: LOAD ORIGINAL IMAGE
# ==========================================

original_bgr = cv2.imread(str(IMAGE_PATH))

if original_bgr is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Please check the filename inside the dataset folder."
    )

# Convert BGR to RGB
original = cv2.cvtColor(
    original_bgr,
    cv2.COLOR_BGR2RGB
)

# Convert image to floating point
original_float = original.astype(np.float32) / 255.0

print("Original image loaded successfully!")
print("Image dimensions:", original.shape)


# ==========================================
# STEP 2: ADD GAUSSIAN NOISE
# ==========================================

np.random.seed(42)

gaussian_noise = np.random.normal(
    loc=0.0,
    scale=0.08,
    size=original_float.shape
).astype(np.float32)

noisy_gaussian = np.clip(
    original_float + gaussian_noise,
    0.0,
    1.0
)

# Convert noisy image to uint8
gaussian_noisy_uint8 = (
    noisy_gaussian * 255
).astype(np.uint8)


# ==========================================
# STEP 3: ADD SALT-AND-PEPPER NOISE
# ==========================================

noisy_sp = original.copy()

height, width, _ = noisy_sp.shape

random_values = np.random.random(
    (height, width)
)

# Add salt-and-pepper noise
noisy_sp[random_values < 0.02] = [0, 0, 0]

noisy_sp[random_values > 0.98] = [255, 255, 255]


# ==========================================
# STEP 4: APPLY GAUSSIAN FILTERING
# ==========================================

restored_gaussian = cv2.GaussianBlur(
    gaussian_noisy_uint8,
    (5, 5),
    1.0
)


# ==========================================
# STEP 5: APPLY MEDIAN FILTERING
# ==========================================

restored_median = cv2.medianBlur(
    noisy_sp,
    5
)


# ==========================================
# STEP 6: CALCULATE PSNR AND SSIM
# ==========================================

original_uint8 = original.copy()

# Gaussian noisy image
psnr_gaussian = peak_signal_noise_ratio(
    original_uint8,
    gaussian_noisy_uint8,
    data_range=255
)

ssim_gaussian = structural_similarity(
    original_uint8,
    gaussian_noisy_uint8,
    channel_axis=2,
    data_range=255
)

# Gaussian filtered image
psnr_gaussian_restored = peak_signal_noise_ratio(
    original_uint8,
    restored_gaussian,
    data_range=255
)

ssim_gaussian_restored = structural_similarity(
    original_uint8,
    restored_gaussian,
    channel_axis=2,
    data_range=255
)

# Salt-and-pepper noisy image
psnr_sp = peak_signal_noise_ratio(
    original_uint8,
    noisy_sp,
    data_range=255
)

ssim_sp = structural_similarity(
    original_uint8,
    noisy_sp,
    channel_axis=2,
    data_range=255
)

# Median filtered image
psnr_median = peak_signal_noise_ratio(
    original_uint8,
    restored_median,
    data_range=255
)

ssim_median = structural_similarity(
    original_uint8,
    restored_median,
    channel_axis=2,
    data_range=255
)


# ==========================================
# STEP 7: SAVE OUTPUT IMAGES
# ==========================================

def save_image(filename, image):
    """Save an RGB image to the output directory."""

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    output_path = OUTPUT_DIR / filename

    success = cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
    )

    if not success:
        raise IOError(
            f"Could not save image: {output_path}"
        )

    print("Saved:", output_path.name)


save_image(
    "01_original.png",
    original
)

save_image(
    "02_gaussian_noise.png",
    gaussian_noisy_uint8
)

save_image(
    "03_gaussian_restored.png",
    restored_gaussian
)

save_image(
    "04_salt_pepper_noise.png",
    noisy_sp
)

save_image(
    "05_median_restored.png",
    restored_median
)


# ==========================================
# STEP 8: DISPLAY IMAGE COMPARISON
# ==========================================

images = [
    original,
    gaussian_noisy_uint8,
    restored_gaussian,
    noisy_sp,
    restored_median
]

titles = [
    "Original Image",
    "Gaussian Noise",
    "Gaussian Filtered",
    "Salt-and-Pepper Noise",
    "Median Filtered"
]

plt.figure(figsize=(15, 9))

for index, (image, title) in enumerate(
    zip(images, titles),
    start=1
):
    plt.subplot(2, 3, index)

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

plt.tight_layout()

comparison_path = (
    OUTPUT_DIR / "06_restoration_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

print("\nComparison image saved successfully!")

plt.show()


# ==========================================
# STEP 9: PRINT QUALITY MEASUREMENTS
# ==========================================

print("\n" + "=" * 45)
print("       IMAGE QUALITY RESULTS")
print("=" * 45)

print("\n1. GAUSSIAN NOISE")
print(f"PSNR: {psnr_gaussian:.2f} dB")
print(f"SSIM: {ssim_gaussian:.4f}")

print("\n2. AFTER GAUSSIAN FILTERING")
print(f"PSNR: {psnr_gaussian_restored:.2f} dB")
print(f"SSIM: {ssim_gaussian_restored:.4f}")

print("\n3. SALT-AND-PEPPER NOISE")
print(f"PSNR: {psnr_sp:.2f} dB")
print(f"SSIM: {ssim_sp:.4f}")

print("\n4. AFTER MEDIAN FILTERING")
print(f"PSNR: {psnr_median:.2f} dB")
print(f"SSIM: {ssim_median:.4f}")

print("\n" + "=" * 45)
print("TASK 1A COMPLETED SUCCESSFULLY!")
print("=" * 45)

print("\nAll output files saved in:")
print(OUTPUT_DIR.resolve())