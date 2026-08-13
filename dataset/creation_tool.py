import csv
from pathlib import Path
import time
import urllib.request
import cv2
import mediapipe as mp

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not MODEL_PATH.exists():
    print("No model file. Downloading from official MediaPipe server...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download completed successfully!\n")

CSV_FILE = SCRIPT_DIR / "gestures_data.csv"

GESTURES = {
    0: {"file": "none.png",          "name": "NONE / IDLE"},
    1: {"file": "like.png",          "name": "LIKE / THUMBS UP"},
    2: {"file": "hey.png",           "name": "HEY / OPEN HAND"},
    3: {"file": "point_up.png",      "name": "POINT UP"},
    4: {"file": "heart.png",         "name": "HEART / OK"},
    5: {"file": "thinking.png",      "name": "THINKING"},
    6: {"file": "evil.png",          "name": "EVIL / CLENCHED FIST"},
    7: {"file": "middle_finger.png", "name": "MIDDLE FINGER"},
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

header = ["label"]
for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

try:
    with open(CSV_FILE, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
except FileExistsError:
    pass

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

cap = cv2.VideoCapture(0)

print("--- GUIDE ---")
print("1. Click numbers 1-8 (classes) to capture current hand position.")
print("2. Click 'q' to exit.")

for i in GESTURES:
    print(f"{i}: {GESTURES[i]['name']}")

with HandLandmarker.create_from_options(options) as landmarker:
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms = int((time.time() - start_time) * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_landmarks = result.hand_landmarks[0]
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                
        cv2.imshow("Dataset Creator", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Closing program...")
            break

        if chr(key) in ["1", "2", "3", "4", "5", "6"]:
            if result.hand_world_landmarks and len(result.hand_world_landmarks) > 0:
                label = chr(key)
                world_hand = result.hand_world_landmarks[0]

                row = [label]
                for lm in world_hand:
                    row.extend([lm.x, lm.y, lm.z])

                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print(f"Added sample for class {label}!")

cap.release()
cv2.destroyAllWindows()