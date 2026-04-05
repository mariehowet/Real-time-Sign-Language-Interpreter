import mediapipe as mp
import torch
from mediapipe.tasks.python import vision, BaseOptions
import cv2, time, numpy as np

from src.config.config import Config
from src.models.test_landmarks import LandmarkClassifier

# ─── Config ───────────────────────────────────────────────────────────────────
latest_result = None
MODEL_ASSET_PATH = str(Config.HAND_LANDMARKER)
MODEL_PATH = str(Config.MODEL_LANDMARKER)
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
SIGN_TEXT_COLOR = (0, 0, 255)

# ─── Mapping index → lettre/signe ─────────────────────────────────────────────
idx_to_sign = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'Space',  # insérer un espace ici
    20: 'T',
    21: 'U',
    22: 'V',
    23: 'W',
    24: 'X',
    25: 'Y',
    26: 'Z'
}

# ─── Load model ───────────────────────────────────────────────────────────────────
model = LandmarkClassifier(63, 27) # ta classe du modèle
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))# TODO : a changer en fonction
model.eval()  # mode inference

# ─── Hand Landmarker ───────────────────────────────────────────────────────────────────
def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
    print('hand landmarker result: {}'.format(result))


options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_ASSET_PATH),
    running_mode= vision.RunningMode.LIVE_STREAM, # TODO: a changer si image, video ou webcam
    num_hands=2,
    result_callback=result_callback # TODO: a changer si image, video ou webcam
)

landmarker = vision.HandLandmarker.create_from_options(options)

# ─── Utilitaires ───────────────────────────────────────────────────────────────────
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

def landmarks_to_tensor(hand_landmarks):
    """Convertit 21 landmarks en tensor [1, 63] exactement comme le dataset"""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])  # normalisé entre 0 et 1 comme dans ton dataset
    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # shape [1,63]

# ─── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.flip(frame, 1)  # flip pour effet miroir
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) # TODO: a changer si image, video ou webcam
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms) # TODO: a changer si image, video ou webcam

    annotated_frame = frame.copy()
    if latest_result and latest_result.hand_landmarks:
        annotated_frame = draw_landmarks_on_image(frame, latest_result)
        # Prédiction signe pour chaque main
        for hand_landmarks in latest_result.hand_landmarks:
            input_tensor = landmarks_to_tensor(hand_landmarks)
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                predicted_sign = idx_to_sign.get(predicted_class, "?")

            # Affichage sur la vidéo
            cv2.putText(annotated_frame, f"Signe: {predicted_sign}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, SIGN_TEXT_COLOR, 2)

    cv2.imshow("Hand Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27: # Esc pour quitter
        break

cap.release()
cv2.destroyAllWindows()

