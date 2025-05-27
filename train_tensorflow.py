import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

print('TensorFlow version:', tf.__version__)

def load_data():
    images, labels = [], []
    
    print('Loading helipad images...')
    helipad_path = 'data/helipad'
    for img_file in os.listdir(helipad_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(helipad_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(1)
    
    print('Loading no-helipad images...')
    no_helipad_path = 'data/no_helipad'
    for img_file in os.listdir(no_helipad_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(no_helipad_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(0)
    
    return np.array(images), np.array(labels)

def create_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print('Loading data...')
X, y = load_data()
print(f'Total images: {len(X)}')

if len(X) == 0:
    print('No images found!')
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model()
print('Training model...')

model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy:.4f}')

os.makedirs('models', exist_ok=True)
model.save('models/helipad_model.h5')
print('Model saved!')
