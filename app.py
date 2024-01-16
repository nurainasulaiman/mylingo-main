from flask import Flask, render_template, jsonify
import pickle
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp
import logging

# Configure logging settings for error handling
logging.basicConfig(filename='app.log', level=logging.ERROR)

app = Flask(__name__)

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

# Variable to store the predicted label
predicted_label = "No prediction yet"

# Main loop for hand landmarks classification
def classify_hand_landmarks():
    global predicted_label
    try:    
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
                x_max, x_min = np.max(landmarks[::2]), np.min(landmarks[::2])
                hand_input = np.array(landmarks).reshape(1, num_landmarks, 2, 1) / (x_max - x_min)

                # Predict the label using the CNN model
                label_idx = np.argmax(model.predict(hand_input))
                predicted_label = label_encoder.inverse_transform([label_idx])[0]

            # Break the loop if 'q' or 'Esc' key is pressed
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    break
    
    except Exception as e:
        # Log the exception
        logging.error(f"An error occurred during hand sign detection: {e}")

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

# Add this route to your Flask app
@app.route('/hand_landmarks_classification', methods=['GET'])
def hand_landmarks_classification():
    try:
        classify_hand_landmarks()
        return jsonify({'predicted_label': predicted_label})
    
    except Exception as e:
        # Log the exception
        logging.error(f"An error occurred in the hand_landmarks_classification route: {e}")

        # Return an error response
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/translate.html')
def translate():
    return render_template('translate.html')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/parallax.html')
def other_page():
    return render_template('parallax.html')

@app.route('/choosepath.html')
def choosepath():
    return render_template('choosepath.html')

@app.route('/signlogin.html')
def signlogin():
    return render_template('signlogin.html')

@app.route('/basics.html')
def basic():
    return render_template('basics.html')

@app.route('/lesson.html')
def lesson():
    return render_template('lesson.html')

@app.route('/ch1.html')
def ch1():
    return render_template('ch1.html')

@app.route('/ch2.html')
def ch2():
    return render_template('ch2.html')

# @app.route('/temp.html')
# def temp():
#     return render_template('temp.html')

if __name__ == '__main__':
    app.run(debug=True)