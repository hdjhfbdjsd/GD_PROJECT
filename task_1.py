import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ===== Path Setup =====
DATASET_ROOT = r"D:\Desktop\GD-PRO\dataset of voice recognition"
train_dir = os.path.join(DATASET_ROOT, "train")
test_dir = os.path.join(DATASET_ROOT, "test")

# ===== Feature Extraction =====
def extract_features(file_path, mfcc_dim=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_dim)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# ===== Data Preparation =====
def load_data(base_dir):
    X, y = [], []
    for speaker in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(speaker)
    return np.array(X), np.array(y)

print("Loading training data...")
X_train, y_train = load_data(train_dir)
print("Loading test data...")
X_test, y_test = load_data(test_dir)

# ===== Label Encoding =====
le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_test_enc = to_categorical(le.transform(y_test))

# ===== Data Shape =====
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# ===== Model Building =====
model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
history = model.fit(X_train, y_train_enc, epochs=50, batch_size=8, validation_data=(X_test, y_test_enc))

# ===== Evaluation =====
print("\nModel Evaluation:")
test_loss, test_acc = model.evaluate(X_test, y_test_enc)
print(f"Test Accuracy: {test_acc:.2f}")

# ===== OUTPUT 1: Accuracy Plot =====
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# ===== OUTPUT 2: Confusion Matrix =====
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_enc, axis=1)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_true, y_pred_classes), annot=True, fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()