import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import cv2, time, os, csv
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", "hand_landmarker.task")

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


SEQUENCE_LENGTH = 30  # nombre de frames par mot

latest_result = None
recording = False
sequence = []
current_label = None

# ─── LANDMARKER ───────────────────────────────────────────
def callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=callback
)

landmarker = vision.HandLandmarker.create_from_options(options)

# ─── UTILS ───────────────────────────────────────────────
def extract_landmarks(result):
    if not result or not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    return [coord for p in lm for coord in (p.x, p.y, p.z)]

# ─── WEBCAM ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)

print("R = start | S = stop | ESC = quit")


current_label = input("Quel mot vas-tu enregistrer ? ").upper()
print(f"Prêt pour : {current_label}. Appuie sur 'R' pour lancer une séquence.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    landmarker.detect_async(mp_image, int(time.time() * 1000))

    # affichage
    status = f"{current_label or 'NONE'} | Frames: {len(sequence)}/{SEQUENCE_LENGTH}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # enregistrement
    if recording and latest_result:
        lm = extract_landmarks(latest_result)
        if lm:
            sequence.append(lm)

        if len(sequence) >= SEQUENCE_LENGTH:
            label_dir = os.path.join(RAW_DIR, current_label)
            os.makedirs(label_dir, exist_ok=True)

            filename = f"{int(time.time())}.npy"
            path = os.path.join(label_dir, filename)
            np.save(path, np.array(sequence))

            print(f"✅ Séquence sauvegardée : {filename}")

            recording = False
            sequence = []

    cv2.imshow("Capture mots", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        print(f"▶️ Recording: {current_label}")
        recording = True
        sequence = []

    elif key == ord('s'):
        print("⏹ Stop")
        recording = False
        sequence = []

cap.release()
cv2.destroyAllWindows()