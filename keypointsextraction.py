import os
import cv2
import csv
import mediapipe as mp

# Path to dataset root folder (change this to your actual path)
DATASET_DIR = "videos"

# Output CSV file
CSV_PATH = "hand_keypoints_dataset.csv"

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Prepare CSV
with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)

    # Write header: x1, y1, z1, ..., x21, y21, z21, label
    header = []
    for i in range(21):
        header += [f'x{i+1}', f'y{i+1}', f'z{i+1}']
    header.append("label")
    writer.writerow(header)

    total_frames = 0
    total_videos = 0

    # Walk through each class directory
    for class_dir in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue

        print(f"[INFO] Processing class: {class_dir}")

        for file in os.listdir(class_path):
            if not file.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            video_path = os.path.join(class_path, file)
            print(f"  → Reading video: {file}")

            cap = cv2.VideoCapture(video_path)

            while True:
                success, img = cap.read()
                if not success:
                    break

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                keypoints = []
                if result.multi_hand_landmarks:
                    # Use only the first detected hand
                    hand_landmarks = result.multi_hand_landmarks[0]
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    # If no hand detected, use zeroes
                    keypoints = [0] * 63

                # Add class label
                keypoints.append(class_dir)
                writer.writerow(keypoints)
                total_frames += 1

            cap.release()
            total_videos += 1

print(f"\n✅ Extracted keypoints from {total_videos} videos ({total_frames} frames total)")
print(f"📁 Saved to: {CSV_PATH}")
