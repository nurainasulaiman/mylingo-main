import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle

# Load the trained model and label encoder
model = load_model('hand_landmarks_cnn_model.h5')
with open('label_encoder.p', 'rb') as f:
    label_encoder = pickle.load(f)

# Assuming each hand landmark has 21 coordinates (x, y for 21 landmarks)
num_landmarks = 21

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# OpenCV setup
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Create a window for the user interface
cv2.namedWindow('Hand Landmarks Classification', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Landmarks Classification', 800, 600)

# Initial screen with a start button
start_button_pressed = False

while not start_button_pressed:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Check camera access and connection.")
        break

    # Display the start screen with the "Start" button
    cv2.putText(frame, 'Hand Landmarks Classification', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Press "S" to Start', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Hand Landmarks Classification', frame)

    key = cv2.waitKey(1)
    if key == ord('s') or key == ord('S'):
        start_button_pressed = True

# Main loop for hand landmarks classification
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Check camera access and connection.")
        break

    # Convert the frame to RGB for input to MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                landmarks.extend([x, y])

            # Reshape and normalize hand landmarks
            hand_input = np.array(landmarks).reshape(1, num_landmarks, 2, 1) / frame.shape[1]

            # Predict the label using the CNN model
            label_idx = np.argmax(model.predict(hand_input))
            predicted_label = label_encoder.inverse_transform([label_idx])[0]

            # Display the predicted label on the frame
            cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display predicted label in the user interface window
            ui_text = f'Predicted Label: {predicted_label}'
            cv2.putText(frame, ui_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw hand landmarks on the frame
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Landmarks Classification', frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or 'Esc' key to exit
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

