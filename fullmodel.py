import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import pyautogui
import time
import joblib
scaler = joblib.load('scaler.pkl')

# Load the trained model
class GestureClassifier(nn.Module):
    def __init__(self, input_size=63, num_classes=4):
        super(GestureClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load label classes
classes = np.load("gesture_classes.npy", allow_pickle=True)
num_classes = len(classes)

# Load model
model = GestureClassifier(input_size=63, num_classes=num_classes)
model.load_state_dict(torch.load("gesture_classifier_model.pth", map_location=torch.device('cpu')))
model.eval()

# Setup MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Map class labels to keypresses
gesture_to_key = {
    "swipe_left": "left",
    "swipe_right": "right",
    "jump": "up",
    "slide": "down"
}

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints[:63])  # use only first 21 landmarks × 3 = 63
    return None

# Start webcam
cap = cv2.VideoCapture(0)
print("[Game] Starting gesture control... Show your hand to control Temple Run!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural mirror effect
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    keypoints = extract_keypoints(results)
    if keypoints is not None:
        keypoints_scaled = scaler.transform([keypoints])  # Shape: (1, 63)

        # Convert to tensor
        input_tensor = torch.tensor(keypoints_scaled, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            gesture = classes[predicted_class]
            # probs = torch.softmax(output, dim=1)
            # confidence = torch.max(probs).item()
            # print(f'The predicted class is: {gesture}, and confidence is: {confidence}')



        # Display prediction
        # cv2.putText(frame, f'Gesture: {gesture}', (10, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Simulate keypress if recognized
        if gesture in gesture_to_key:
            key = gesture_to_key[gesture]
            pyautogui.press(key)
            time.sleep(0.1)  # avoid repeated actions

    # Show hand landmarks
    # if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show camera feed
    # cv2.imshow("Gesture Controller", frame)

    # Exit on 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

cap.release()
cv2.destroyAllWindows()
