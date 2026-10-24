import cv2
import numpy as np
from pathlib import Path
import csv


# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "dataset" / "cca 3 cvv.mp4"

OUTPUT_DIR = BASE_DIR / "outputs" / "unit5_motion_estimation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_DIFF_DIR = OUTPUT_DIR / "frame_differencing"
GMM_DIR = OUTPUT_DIR / "gmm_background_subtraction"
FLOW_DIR = OUTPUT_DIR / "optical_flow"

FRAME_DIFF_DIR.mkdir(exist_ok=True)
GMM_DIR.mkdir(exist_ok=True)
FLOW_DIR.mkdir(exist_ok=True)


# ============================================
# 2. CHECK INPUT VIDEO
# ============================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Unable to open the input video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 55)
print("UNIT 5 - VIDEO MOTION ESTIMATION")
print("=" * 55)
print(f"Input video: {VIDEO_PATH.name}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")


# ============================================
# 3. INITIALIZE ALGORITHMS
# ============================================

# Gaussian Mixture Model background subtractor
gmm = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25,
    detectShadows=True
)

# Lucas-Kanade optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)


# ============================================
# 4. READ FIRST FRAME
# ============================================

success, previous_frame = cap.read()

if not success:
    cap.release()
    raise RuntimeError("Unable to read the first video frame.")

previous_gray = cv2.cvtColor(
    previous_frame,
    cv2.COLOR_BGR2GRAY
)

previous_points = cv2.goodFeaturesToTrack(
    previous_gray,
    mask=None,
    **feature_params
)

# Used for frame differencing
previous_blurred = cv2.GaussianBlur(
    previous_gray,
    (5, 5),
    0
)

# Used to generate the first GMM background mask
gmm.apply(previous_frame)

frame_number = 1

diff_count = 0
gmm_count = 0
flow_count = 0

comparison_rows = []

# Save original first frame
cv2.imwrite(
    str(OUTPUT_DIR / "original_frame.jpg"),
    previous_frame
)


# ============================================
# 5. PROCESS CONSECUTIVE FRAMES
# ============================================

print("\nProcessing video frames...")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ----------------------------------------
    # A. FRAME DIFFERENCING
    # ----------------------------------------

    difference = cv2.absdiff(
        previous_blurred,
        blurred
    )

    _, frame_mask = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)

    frame_mask = cv2.morphologyEx(
        frame_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    frame_mask = cv2.dilate(
        frame_mask,
        kernel,
        iterations=2
    )

    frame_diff_vis = cv2.cvtColor(
        frame_mask,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        frame_diff_vis,
        "Frame Differencing",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    diff_pixels = int(
        cv2.countNonZero(frame_mask)
    )

    diff_count += diff_pixels

    # ----------------------------------------
    # B. GMM BACKGROUND SUBTRACTION
    # ----------------------------------------

    gmm_mask = gmm.apply(frame)

    # Remove detected shadow pixels.
    _, gmm_mask = cv2.threshold(
        gmm_mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    gmm_mask = cv2.morphologyEx(
        gmm_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    gmm_vis = cv2.cvtColor(
        gmm_mask,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        gmm_vis,
        "GMM Background Subtraction",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    gmm_pixels = int(
        cv2.countNonZero(gmm_mask)
    )

    gmm_count += gmm_pixels

    # ----------------------------------------
    # C. LUCAS-KANADE OPTICAL FLOW
    # ----------------------------------------

    flow_vis = frame.copy()

    if (
        previous_points is None
        or len(previous_points) < 10
    ):

        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            mask=None,
            **feature_params
        )

    if previous_points is not None:

        next_points, status, errors = (
            cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                previous_points,
                None,
                **lk_params
            )
        )

        if next_points is not None and status is not None:

            good_new = next_points[
                status.ravel() == 1
            ]

            good_old = previous_points[
                status.ravel() == 1
            ]

            motion_magnitudes = []

            for new, old in zip(good_new, good_old):

                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                motion = np.sqrt(
                    (x_new - x_old) ** 2
                    + (y_new - y_old) ** 2
                )

                motion_magnitudes.append(float(motion))

                cv2.arrowedLine(
                    flow_vis,
                    (int(x_old), int(y_old)),
                    (int(x_new), int(y_new)),
                    (0, 255, 0),
                    2,
                    tipLength=0.3
                )

                cv2.circle(
                    flow_vis,
                    (int(x_new), int(y_new)),
                    3,
                    (0, 0, 255),
                    -1
                )

            valid_tracks = len(good_new)
            flow_count += valid_tracks

            mean_motion = (
                float(np.mean(motion_magnitudes))
                if motion_magnitudes
                else 0.0
            )

            # Track valid features into the next frame.
            previous_points = good_new.reshape(-1, 1, 2)

        else:
            valid_tracks = 0
            mean_motion = 0.0
            previous_points = None

    else:
        valid_tracks = 0
        mean_motion = 0.0

    cv2.putText(
        flow_vis,
        "Lucas-Kanade Optical Flow",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # ----------------------------------------
    # D. SAVE SELECTED OUTPUTS
    # ----------------------------------------

    # Save one result every 30 frames and the last frame.
    if frame_number % 30 == 0 or frame_number == total_frames:

        cv2.imwrite(
            str(FRAME_DIFF_DIR / f"frame_{frame_number:04d}.jpg"),
            frame_diff_vis
        )

        cv2.imwrite(
            str(GMM_DIR / f"frame_{frame_number:04d}.jpg"),
            gmm_vis
        )

        cv2.imwrite(
            str(FLOW_DIR / f"frame_{frame_number:04d}.jpg"),
            flow_vis
        )

    comparison_rows.append([
        frame_number,
        diff_pixels,
        gmm_pixels,
        valid_tracks,
        round(mean_motion, 3)
    ])

    # ----------------------------------------
    # E. UPDATE PREVIOUS FRAME
    # ----------------------------------------

    previous_gray = gray.copy()

    previous_blurred = blurred.copy()

    if frame_number % 100 == 0:
        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# ============================================
# 6. RELEASE VIDEO
# ============================================

cap.release()


# ============================================
# 7. SAVE NUMERICAL RESULTS
# ============================================

csv_path = OUTPUT_DIR / "motion_estimation_results.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    csv_writer = csv.writer(csvfile)

    csv_writer.writerow([
        "Frame Number",
        "Frame Difference Foreground Pixels",
        "GMM Foreground Pixels",
        "Valid Optical Flow Tracks",
        "Mean Optical Flow Magnitude"
    ])

    csv_writer.writerows(comparison_rows)


# ============================================
# 8. SAVE SUMMARY REPORT
# ============================================

processed_frames = len(comparison_rows)

with open(
    OUTPUT_DIR / "motion_estimation_report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write("VIDEO MOTION ESTIMATION REPORT\n")
    report.write("=" * 45 + "\n\n")

    report.write(f"Input video: {VIDEO_PATH.name}\n")
    report.write(f"FPS: {fps:.2f}\n")
    report.write(f"Frames processed: {processed_frames}\n\n")

    report.write("METHODS IMPLEMENTED\n")
    report.write("1. Frame Differencing\n")
    report.write("2. GMM Background Subtraction (MOG2)\n")
    report.write("3. Lucas-Kanade Optical Flow\n\n")

    report.write("AGGREGATE RESULTS\n")
    report.write(
        f"Total foreground pixels from frame differencing: "
        f"{diff_count}\n"
    )

    report.write(
        f"Total foreground pixels from GMM: "
        f"{gmm_count}\n"
    )

    report.write(
        f"Total valid optical-flow tracks across frames: "
        f"{flow_count}\n\n"
    )

    report.write("METHOD COMPARISON\n")
    report.write(
        "Frame differencing compares consecutive frames and "
        "highlights changes between them.\n"
    )

    report.write(
        "GMM background subtraction models the background over "
        "time to identify foreground regions.\n"
    )

    report.write(
        "Lucas-Kanade optical flow estimates displacement of "
        "tracked feature points between consecutive frames.\n"
    )

    report.write(
        "\nNote: Aggregate counts are not accuracy measurements. "
        "Results depend on camera motion, lighting, scene content "
        "and parameter settings.\n"
    )


# ============================================
# 9. FINAL SUMMARY
# ============================================

print("\n" + "=" * 55)
print("MOTION ESTIMATION COMPLETED SUCCESSFULLY!")
print("=" * 55)

print(f"Frames processed: {processed_frames}")
print(f"Frame differencing foreground pixels: {diff_count}")
print(f"GMM foreground pixels: {gmm_count}")
print(f"Optical flow tracks: {flow_count}")

print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"CSV results: {csv_path}")
print(
    f"Text report: "
    f"{OUTPUT_DIR / 'motion_estimation_report.txt'}"
)