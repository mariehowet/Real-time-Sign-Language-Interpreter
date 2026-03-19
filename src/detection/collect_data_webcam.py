import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import cv2, csv, time, os
import numpy as np
# ─── Config ───────────────────────────────────────────────────────────────────
latest_result = None
MODEL_ASSET_PATH = "hand_landmarker.task"
DATA_FILE = "data.csv"
current_label = None
sample_count = 0
SAMPLES_PER_LETTER = 200
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# ─── Hand Landmarker ───────────────────────────────────────────────────────────────────
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_ASSET_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM, # TODO: a changer si image, video ou webcam
    num_hands=1,
    result_callback=result_callback # TODO: a changer si image, video ou webcam
)
landmarker = vision.HandLandmarker.create_from_options(options)


# ─── Draw landmarks ───────────────────────────────────────────────────────────────────
mp_hands = vision.HandLandmarksConnections
mp_drawing = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# ─── Data file ───────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{axis}{i}" for i in range(21) for axis in ["x","y","z"]] + ["label"]
        writer.writerow(header)

# ─── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Appuie sur une lettre (A-Z) pour commencer à collecter. ESC pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) # TODO: a changer si image, video ou webcam
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms) # TODO: a changer si image, video ou webcam

    # Affichage du statut
    status = f"Lettre: {current_label or '?'}  |  Échantillons: {sample_count}/{SAMPLES_PER_LETTER}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 100), 1)

    # Enregistrement si une lettre est active et une main détectée
    if current_label and latest_result and latest_result.hand_landmarks:
        annotated_frame = draw_landmarks_on_image(frame, latest_result)
        lm = latest_result.hand_landmarks[0]
        row = [coord for p in lm for coord in (p.x, p.y, p.z)]
        row.append(current_label)

        with open(DATA_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)

        sample_count += 1
        if sample_count >= SAMPLES_PER_LETTER:
            print(f"✅ {current_label} terminé !")
            current_label = None
            sample_count = 0

    else:
        annotated_frame = frame

    cv2.imshow("Collecte ASL", annotated_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z ou a-z
        current_label = chr(key).upper()
        sample_count = 0
        print(f"→ Collecte pour la lettre: {current_label}")

cap.release()
cv2.destroyAllWindows()