# Helipad Detection Setup and Training
Write-Host "=== Helipad Detection Setup ===" -ForegroundColor Green

# Create directories
New-Item -ItemType Directory -Force -Path "data\helipad"
New-Item -ItemType Directory -Force -Path "data\no_helipad" 
New-Item -ItemType Directory -Force -Path "models"

Write-Host "Directories created!" -ForegroundColor Green

# Create sample images
$sampleCode = @"
import os
from PIL import Image, ImageDraw
import random

def create_helipad_image(i):
    img = Image.new('RGB', (224, 224), color=(34, 139, 34))
    draw = ImageDraw.Draw(img)
    draw.ellipse([75, 75, 149, 149], fill=(128, 128, 128))
    draw.rectangle([100, 90, 124, 134], fill=(255, 255, 255))
    img.save(f'data/helipad/helipad_{i}.jpg')

def create_no_helipad_image(i):
    colors = [(34, 139, 34), (139, 69, 19), (70, 130, 180)]
    img = Image.new('RGB', (224, 224), color=random.choice(colors))
    draw = ImageDraw.Draw(img)
    for _ in range(3):
        x1, y1 = random.randint(0, 112), random.randint(0, 112)
        x2, y2 = random.randint(112, 224), random.randint(112, 224)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    img.save(f'data/no_helipad/no_helipad_{i}.jpg')

for i in range(20):
    create_helipad_image(i)
    create_no_helipad_image(i)

print("Sample images created!")
"@

$sampleCode | Out-File -FilePath "create_samples.py"

# Create training script
$trainCode = @"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

def load_data():
    images, labels = [], []
    
    # Load helipad images
    for img_file in os.listdir('data/helipad'):
        img = cv2.imread(f'data/helipad/{img_file}')
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        images.append(img)
        labels.append(1)  # helipad
    
    # Load no-helipad images  
    for img_file in os.listdir('data/no_helipad'):
        img = cv2.imread(f'data/no_helipad/{img_file}')
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        images.append(img)
        labels.append(0)  # no helipad
    
    return np.array(images), np.array(labels)

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print('Loading data...')
X, y = load_data()
print(f'Loaded {len(X)} images')

model = create_model()
print('Training...')
model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2)

model.save('models/helipad_model.h5')
print('Model saved!')
"@

$trainCode | Out-File -FilePath "train.py"

# Install packages and run
pip install tensorflow pillow opencv-python
python create_samples.py
python train.py

Write-Host "Training complete!" -ForegroundColor Green
