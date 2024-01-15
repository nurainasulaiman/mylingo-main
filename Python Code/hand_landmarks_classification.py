import pickle
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import mediapipe as mp

# Load hand landmarks data
data_dict = pickle.load(open('C:/Users/nurai/Favorites/mylingo-main/data.pickle', 'rb'))

# Assuming each hand landmark has 21 coordinates (x, y for 21 landmarks)
num_landmarks = 21

# Pad or truncate hand landmarks to a fixed size
max_sequence_length = num_landmarks * 2  # Assuming x, y for each landmark
data = [sample[:max_sequence_length] + [0] * (max_sequence_length - len(sample)) if len(sample) < max_sequence_length else sample[:max_sequence_length] for sample in data_dict['data']]

data = np.asarray(data)
labels = np.asarray(data_dict['labels'])

# Convert string labels to numerical labels using LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Reshape data for CNN input
x_train = x_train.reshape(-1, num_landmarks, 2, 1)
x_test = x_test.reshape(-1, num_landmarks, 2, 1)

# Define the CNN model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 2), activation='relu', input_shape=(num_landmarks, 2, 1)))
model.add(layers.MaxPooling2D((2, 1)))

model.add(layers.Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 1)))

model.add(layers.Conv2D(128, (2, 2), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 1)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))  # Adding dropout for regularization

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Adding dropout for regularization

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(np.max(labels_encoded) + 1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model without data augmentation
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))

# Evaluate the model
y_predict_prob = model.predict(x_test)
y_predict_encoded = np.argmax(y_predict_prob, axis=1)

# Convert back to original labels for evaluation
y_predict = label_encoder.inverse_transform(y_predict_encoded)

score = accuracy_score(y_predict, label_encoder.inverse_transform(y_test))
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model and label encoder
model.save('hand_landmarks_cnn_model.keras')
# model.save('hand_landmarks_cnn_model.h5')
with open('label_encoder.keras', 'wb') as f: # Save the label encoder using the native Keras format
    pickle.dump(label_encoder, f)


# Plot training and validation loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()

# import pickle
# import os
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# import mediapipe as mp

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
    cv2.putText(frame, 'Hand Landmarks Classification', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
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
            x_max, x_min = np.max(landmarks[::2]), np.min(landmarks[::2])
            hand_input = np.array(landmarks).reshape(1, num_landmarks, 2, 1) / (x_max - x_min)

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
