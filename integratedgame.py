import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import time
import joblib

# Load scaler and classes
scaler = joblib.load('scaler.pkl')
classes = np.load("gesture_classes.npy", allow_pickle=True)
num_classes = len(classes)

# Model definition
class GestureClassifier(nn.Module):
    def __init__(self, input_size=10, num_classes=4):
        super(GestureClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Load model
optimized_model = torch.jit.load("gesture_classifier_traced.pt")
optimized_model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Gesture to direction mapping
gesture_to_direction = {
    "swipe_left": (-20, 0),
    "swipe_right": (20, 0),
    "jump": (0, -20),
    "slide": (0, 20)
}

# Extract only x, y from 5 keypoints of index finger
def extract_keypoints(results):
    index_finger_indices = [0, 5, 6, 7, 8]
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = []
        for idx in index_finger_indices:
            lm = hand.landmark[idx]
            keypoints.extend([lm.x, lm.y])
        return np.array(keypoints)
    return None

# Game config
win_size = (640, 480)
player_color = (0, 255, 0)
bg_color = (50, 50, 50)
player_pos = [320, 240]
player_size = 40

# Start camera
cap = cv2.VideoCapture(0)
print("[Game] Use hand gestures to move the square!")

gesture_cooldown = 0.1
last_gesture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process hand
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    keypoints = extract_keypoints(results)
    if keypoints is not None:
        keypoints_scaled = scaler.transform([keypoints])
        input_tensor = torch.tensor(keypoints_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = optimized_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            gesture = classes[predicted_class]

        # Update position if valid gesture
        if gesture in gesture_to_direction and (time.time() - last_gesture_time) > gesture_cooldown:
            dx, dy = gesture_to_direction[gesture]
            player_pos[0] = np.clip(player_pos[0] + dx, 0, win_size[0] - player_size)
            player_pos[1] = np.clip(player_pos[1] + dy, 0, win_size[1] - player_size)
            last_gesture_time = time.time()

    # Draw game
    game_frame = np.full((win_size[1], win_size[0], 3), bg_color, dtype=np.uint8)
    cv2.rectangle(game_frame,
                  (player_pos[0], player_pos[1]),
                  (player_pos[0] + player_size, player_pos[1] + player_size),
                  player_color,
                  -1)

    # UI text
    cv2.putText(game_frame, "Use gestures: Left, Right, Jump, Slide",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Show window
    cv2.imshow("Gesture Controlled Game", game_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
