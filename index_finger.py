import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Define finger keypoints (0 is wrist, 5-8 are index finger joints to tip)
finger_indices = [0, 5, 6, 7, 8]

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            h, w, c = img.shape
            for idx in finger_indices:
                lm = handLms.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"ID {idx}: (x={cx}, y={cy})")  # You can store this in a list instead of printing

                # Draw circle on each keypoint
                cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

            # Optional: draw full hand (comment out if not needed)
            # mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
