# ============================================================
# CCA 3 - UNIT 4: FEATURE EXTRACTION AND FEATURE MATCHING
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.feature import hog
from skimage import exposure


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs" / "unit4_feature_extraction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1: LOAD THE TWO CLAY PHOTOGRAPHS
# ============================================================

print("=" * 60)
print("CCA 3 - UNIT 4: FEATURE EXTRACTION AND FEATURE MATCHING")
print("=" * 60)

image1_path = DATASET_DIR / "cca 3.jpg"
image2_path = DATASET_DIR / "cca 3 cv..jpg"

# Check image files.
if not image1_path.exists():
    raise FileNotFoundError(
        f"First clay photograph not found:\n{image1_path}"
    )

if not image2_path.exists():
    raise FileNotFoundError(
        f"Second clay photograph not found:\n{image2_path}"
    )

# Read photographs.
image1 = cv2.imread(str(image1_path))
image2 = cv2.imread(str(image2_path))

if image1 is None or image2 is None:
    raise ValueError("Could not read one or both photographs.")

# Convert BGR to RGB.
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Convert to grayscale.
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

print("\nBoth clay photographs loaded successfully!")
print("First image:", image1_path.name)
print("Second image:", image2_path.name)
print("First image dimensions:", image1.shape)
print("Second image dimensions:", image2.shape)


# ============================================================
# STEP 2: HOG FEATURE EXTRACTION
# ============================================================

print("\nSTEP 2: HOG FEATURE EXTRACTION")

# Resize images for consistent HOG dimensions.
hog_image1 = cv2.resize(gray1, (256, 256))
hog_image2 = cv2.resize(gray2, (256, 256))

# Extract HOG features.
features1, hog_visual1 = hog(
    hog_image1,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True,
    feature_vector=True
)

features2, hog_visual2 = hog(
    hog_image2,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True,
    feature_vector=True
)

# Improve HOG visualization contrast.
hog_visual1 = exposure.rescale_intensity(
    hog_visual1,
    in_range="image",
    out_range=(0, 255)
).astype(np.uint8)

hog_visual2 = exposure.rescale_intensity(
    hog_visual2,
    in_range="image",
    out_range=(0, 255)
).astype(np.uint8)

print("HOG feature extraction completed!")
print("HOG feature vector 1 length:", len(features1))
print("HOG feature vector 2 length:", len(features2))

cv2.imwrite(
    str(OUTPUT_DIR / "01_hog_image1.png"),
    hog_visual1
)

cv2.imwrite(
    str(OUTPUT_DIR / "02_hog_image2.png"),
    hog_visual2
)


# ============================================================
# STEP 3: SIFT KEYPOINT DETECTION
# ============================================================

print("\nSTEP 3: SIFT KEYPOINT DETECTION")

sift = cv2.SIFT_create(nfeatures=1500)

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

print("SIFT keypoint detection completed!")
print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))

# Draw keypoints.
keypoint_image1 = cv2.drawKeypoints(
    image1_rgb,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoint_image2 = cv2.drawKeypoints(
    image2_rgb,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.imsave(
    OUTPUT_DIR / "03_sift_keypoints_image1.png",
    keypoint_image1
)

plt.imsave(
    OUTPUT_DIR / "04_sift_keypoints_image2.png",
    keypoint_image2
)


# ============================================================
# STEP 4: SIFT FEATURE MATCHING
# ============================================================

print("\nSTEP 4: FEATURE MATCHING")

if (
    descriptors1 is None
    or descriptors2 is None
    or len(descriptors1) < 2
    or len(descriptors2) < 2
):
    raise RuntimeError(
        "Not enough SIFT descriptors were detected. "
        "Try photographs with more visible details."
    )

# Create a Brute-Force matcher.
bf = cv2.BFMatcher(cv2.NORM_L2)

# Find two nearest matches.
matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)

# Lowe's ratio test.
good_matches = []

for pair in matches:
    if len(pair) == 2:
        match, second_match = pair

        if match.distance < 0.75 * second_match.distance:
            good_matches.append(match)

print("Total candidate matches:", len(matches))
print("Good matches after ratio test:", len(good_matches))

# Select up to 100 best matches.
display_matches = sorted(
    good_matches,
    key=lambda match: match.distance
)[:100]

matched_image = cv2.drawMatches(
    image1_rgb,
    keypoints1,
    image2_rgb,
    keypoints2,
    display_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.imsave(
    OUTPUT_DIR / "05_sift_feature_matching.png",
    matched_image
)


# ============================================================
# STEP 5: DISPLAY FEATURE EXTRACTION COMPARISON
# ============================================================

print("\nSTEP 5: DISPLAYING FEATURE EXTRACTION RESULTS")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(image1_rgb)
axes[0, 0].set_title("Clay Photograph 1")

axes[0, 1].imshow(hog_visual1, cmap="gray")
axes[0, 1].set_title("HOG Features - Image 1")

axes[0, 2].imshow(keypoint_image1)
axes[0, 2].set_title("SIFT Keypoints - Image 1")

axes[1, 0].imshow(image2_rgb)
axes[1, 0].set_title("Clay Photograph 2")

axes[1, 1].imshow(hog_visual2, cmap="gray")
axes[1, 1].set_title("HOG Features - Image 2")

axes[1, 2].imshow(keypoint_image2)
axes[1, 2].set_title("SIFT Keypoints - Image 2")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_feature_extraction_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 6: DISPLAY ORIGINAL FEATURE MATCHING
# ============================================================

print("\nSTEP 6: DISPLAYING FEATURE MATCHING")

plt.figure(figsize=(18, 9))
plt.imshow(matched_image)

plt.title(
    f"SIFT Feature Matching: {len(good_matches)} Good Matches"
)

plt.axis("off")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_feature_matching_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 7: FEATURE MATCHING AFTER ROTATION
# ============================================================

print("\nSTEP 7: FEATURE MATCHING AFTER ROTATION")

# Rotate the second photograph by 15 degrees.
height, width = gray2.shape

rotation_matrix = cv2.getRotationMatrix2D(
    (width // 2, height // 2),
    15,
    1.0
)

rotated_image = cv2.warpAffine(
    image2_rgb,
    rotation_matrix,
    (width, height),
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(255, 255, 255)
)

# Convert the rotated image to grayscale.
rotated_gray = cv2.cvtColor(
    rotated_image,
    cv2.COLOR_RGB2GRAY
)

# Detect SIFT features in the rotated image.
rotated_keypoints, rotated_descriptors = (
    sift.detectAndCompute(rotated_gray, None)
)

print("Rotated image keypoints:", len(rotated_keypoints))

rotated_good_matches = []

if (
    rotated_descriptors is not None
    and len(rotated_descriptors) >= 2
):
    rotated_matches = bf.knnMatch(
        descriptors1,
        rotated_descriptors,
        k=2
    )

    # Apply Lowe's ratio test.
    for pair in rotated_matches:
        if len(pair) == 2:
            match, second_match = pair

            if match.distance < 0.75 * second_match.distance:
                rotated_good_matches.append(match)

    print(
        "Good matches after rotation:",
        len(rotated_good_matches)
    )

    # Select the best 100 matches.
    rotated_display_matches = sorted(
        rotated_good_matches,
        key=lambda match: match.distance
    )[:100]

    # Draw rotated-image feature matches.
    rotated_matched_image = cv2.drawMatches(
        image1_rgb,
        keypoints1,
        rotated_image,
        rotated_keypoints,
        rotated_display_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save the rotated photograph.
    plt.imsave(
        OUTPUT_DIR / "08_rotated_clay_image.png",
        rotated_image
    )

    # Save matching visualization.
    plt.imsave(
        OUTPUT_DIR / "09_rotated_feature_matching.png",
        rotated_matched_image
    )

    # Display the result.
    plt.figure(figsize=(18, 9))
    plt.imshow(rotated_matched_image)

    plt.title(
        f"SIFT Matching After Rotation: "
        f"{len(rotated_good_matches)} Good Matches"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.show()

else:
    print("Not enough SIFT descriptors in the rotated image.")


# ============================================================
# STEP 8: SAVE FEATURE DATA
# ============================================================

print("\nSTEP 8: SAVING FEATURE DATA")

np.save(
    OUTPUT_DIR / "hog_features_image1.npy",
    features1
)

np.save(
    OUTPUT_DIR / "hog_features_image2.npy",
    features2
)

if rotated_descriptors is not None:
    np.save(
        OUTPUT_DIR / "sift_descriptors_rotated_image.npy",
        rotated_descriptors
    )


# ============================================================
# STEP 9: GENERATE FEATURE EXTRACTION REPORT
# ============================================================

report_path = OUTPUT_DIR / "feature_extraction_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "CCA 3 - UNIT 4: FEATURE EXTRACTION REPORT\n"
    )

    report.write("=" * 55 + "\n\n")

    report.write(f"Image 1: {image1_path.name}\n")
    report.write(f"Image 2: {image2_path.name}\n\n")

    report.write(f"Image 1 dimensions: {image1.shape}\n")
    report.write(f"Image 2 dimensions: {image2.shape}\n\n")

    report.write(f"HOG feature length, image 1: {len(features1)}\n")
    report.write(f"HOG feature length, image 2: {len(features2)}\n\n")

    report.write(f"SIFT keypoints, image 1: {len(keypoints1)}\n")
    report.write(f"SIFT keypoints, image 2: {len(keypoints2)}\n\n")

    report.write(f"Candidate matches: {len(matches)}\n")
    report.write(f"Good matches before rotation: {len(good_matches)}\n")
    report.write(
        f"Good matches after rotation: {len(rotated_good_matches)}\n"
    )


# ============================================================
# STEP 10: COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nAll results saved in:")
print(OUTPUT_DIR)

print("\nGenerated output files:")

for output_file in sorted(OUTPUT_DIR.iterdir()):
    if output_file.is_file():
        print(output_file.name)

print("\nTask 2 feature extraction experiment completed!")