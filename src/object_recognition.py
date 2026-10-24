import cv2
import os
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "dataset" / "cca 3 cvv.mp4"

OUTPUT_DIR = BASE_DIR / "outputs" / "unit5_object_recognition"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "object_recognition_output.mp4"

OUTPUT_IMAGE = OUTPUT_DIR / "object_detection_sample.jpg"

REPORT_FILE = OUTPUT_DIR / "object_recognition_report.txt"


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

# Downloads the pretrained model automatically on first use.
model = YOLO("yolov8n.pt")


# ==========================================
# 3. LOAD HAAR CASCADE CLASSIFIER
# ==========================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Unable to load Haar Cascade classifier.")


# ==========================================
# 4. OPEN INPUT VIDEO
# ==========================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Unable to open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo Information")
print("-----------------")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")


# ==========================================
# 5. INITIALIZE OUTPUT VIDEO
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Unable to create output video.")


# ==========================================
# 6. PROCESS VIDEO FRAMES
# ==========================================

frame_number = 0
sample_saved = False
total_detected_objects = 0
total_detected_faces = 0

object_counts = {}

print("\nProcessing video...")
print("Please wait; YOLO may take some time.")


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------
    # A. YOLO OBJECT DETECTION
    # --------------------------------------

    results = model.predict(
        source=frame,
        conf=0.35,
        verbose=False
    )

    annotated_frame = frame.copy()

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Draw bounding box.
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display detected class and confidence.
            label = f"{class_name}: {confidence:.2f}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            total_detected_objects += 1

            object_counts[class_name] = (
                object_counts.get(class_name, 0) + 1
            )

    # --------------------------------------
    # B. VIOLA-JONES FACE DETECTION
    # --------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            "Face - Viola-Jones",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    total_detected_faces += len(faces)

    # --------------------------------------
    # C. ADD FRAME INFORMATION
    # --------------------------------------

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}/{total_frames}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Save the first processed frame as a sample.
    if not sample_saved:

        cv2.imwrite(
            str(OUTPUT_IMAGE),
            annotated_frame
        )

        sample_saved = True

    # Write processed frame to output video.
    writer.write(annotated_frame)

    if frame_number % 50 == 0:

        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# ==========================================
# 7. RELEASE RESOURCES
# ==========================================

cap.release()
writer.release()


# ==========================================
# 8. SAVE EXPERIMENT REPORT
# ==========================================

with open(REPORT_FILE, "w", encoding="utf-8") as report:

    report.write("OBJECT RECOGNITION USING YOLO AND VIOLA-JONES\n")
    report.write("=" * 50 + "\n\n")

    report.write(f"Input video: {VIDEO_PATH.name}\n")
    report.write(f"Video width: {width}\n")
    report.write(f"Video height: {height}\n")
    report.write(f"Frame rate: {fps:.2f}\n")
    report.write(f"Frames processed: {frame_number}\n\n")

    report.write("YOLO OBJECT DETECTION\n")
    report.write("-" * 30 + "\n")

    report.write(
        f"Total object detections across frames: "
        f"{total_detected_objects}\n"
    )

    for name, count in sorted(object_counts.items()):

        report.write(f"{name}: {count} detections\n")

    report.write("\nVIOLA-JONES FACE DETECTION\n")
    report.write("-" * 30 + "\n")

    report.write(
        f"Total face detections across frames: "
        f"{total_detected_faces}\n"
    )

    report.write(
        "\nNote: Detection totals count detections across frames, "
        "not unique objects.\n"
    )


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 45)
print("OBJECT RECOGNITION COMPLETED!")
print("=" * 45)

print(f"Frames processed: {frame_number}")
print(f"Total YOLO detections: {total_detected_objects}")
print(f"Total face detections: {total_detected_faces}")

print(f"\nOutput video: {OUTPUT_VIDEO}")
print(f"Sample image: {OUTPUT_IMAGE}")
print(f"Experiment report: {REPORT_FILE}")

print("\nOpen the output folder to view your results.")