import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import pyautogui
import time
import joblib

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')

# Load class labels
classes = np.load("gesture_classes.npy", allow_pickle=True)
num_classes = len(classes)

# Define the model for 10 input features (5 landmarks × 2 for x and y)
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

# Initialize and load model weights
model = GestureClassifier(input_size=10, num_classes=num_classes)
# 🔁 Load the TorchScript model
optimized_model = torch.jit.load("gesture_classifier_traced.pt")
optimized_model.eval()

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Gesture to keypress mapping
gesture_to_key = {
    "swipe_left": "left",
    "swipe_right": "right",
    "jump": "up",
    "slide": "down"
}

# Function to extract only x, y of index finger keypoints (0, 5, 6, 7, 8)
def extract_keypoints(results):
    index_finger_indices = [0, 5, 6, 7, 8]
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = []
        for idx in index_finger_indices:
            lm = hand.landmark[idx]
            keypoints.extend([lm.x, lm.y])
        return np.array(keypoints)  # Shape: (10,)
    return None

# Start webcam
cap = cv2.VideoCapture(0)
print("[Game] Starting gesture control... Show your hand to control Temple Run!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    keypoints = extract_keypoints(results)
    if keypoints is not None:
        keypoints_scaled = scaler.transform([keypoints])  # Shape: (1, 10)

        # Convert to tensor
        input_tensor = torch.tensor(keypoints_scaled, dtype=torch.float32)
        with torch.no_grad():
            output = optimized_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            gesture = classes[predicted_class]

        # Trigger key press
        if gesture in gesture_to_key:
            pyautogui.press(gesture_to_key[gesture])
            time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()
