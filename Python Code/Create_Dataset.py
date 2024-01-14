import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'C:/Users/nurai/Favorites/FYP/train'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

            # Normalize coordinates
            x_min, x_max = min(x_), max(x_)
            y_min, y_max = min(y_), max(y_)

            for i in range(len(x_)):
                x = (x_[i] - x_min) / (x_max - x_min)
                y = (y_[i] - y_min) / (y_max - y_min)

                data_aux.append(x)
                data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
f.close()

current_directory = os.getcwd()
pickle_file_path = os.path.join(current_directory, 'data.pickle')
print(f"Data pickle file is located at: {pickle_file_path}")