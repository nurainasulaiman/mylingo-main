import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load hand landmarks data
data_dict = pickle.load(open('C:/Users/nurai/Favorites/FYP/data.pickle', 'rb'))

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
history = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_data=(x_test, y_test))

# Evaluate the model
y_predict_prob = model.predict(x_test)
y_predict_encoded = np.argmax(y_predict_prob, axis=1)

# Convert back to original labels for evaluation
y_predict = label_encoder.inverse_transform(y_predict_encoded)

score = accuracy_score(y_predict, label_encoder.inverse_transform(y_test))
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model and label encoder
model.save('hand_landmarks_cnn_model.h5')
with open('label_encoder.p', 'wb') as f:
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
