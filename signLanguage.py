import cv2
import mediapipe as mp
import os

# Specify the input image path and output path
input_path = 'frame_175.jpg'  # Change this to your image filename
output_path = 'hand_annotated.jpg'

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize Hands with default parameters
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    # Read the image from disk
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {input_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to find hands
    result = hands.process(img_rgb)

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Save the annotated image
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to: {output_path}")
