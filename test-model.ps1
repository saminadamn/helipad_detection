# Test the trained helipad detection model

$testCode = @"
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

# Load the trained model
print('Loading trained model...')
model = tf.keras.models.load_model('models/helipad_model.h5')

def predict_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    confidence = np.max(prediction)
    class_idx = np.argmax(prediction)
    
    class_names = ['No Helipad', 'Helipad']
    result = class_names[class_idx]
    
    return result, confidence

# Test on sample images
print('Testing model on sample images...')
print('-' * 40)

# Test helipad images
helipad_files = os.listdir('data/helipad')[:5]  # Test first 5
for img_file in helipad_files:
    img_path = f'data/helipad/{img_file}'
    result, confidence = predict_image(img_path)
    print(f'{img_file}: {result} (confidence: {confidence:.2f})')

print('-' * 40)

# Test no-helipad images  
no_helipad_files = os.listdir('data/no_helipad')[:5]  # Test first 5
for img_file in no_helipad_files:
    img_path = f'data/no_helipad/{img_file}'
    result, confidence = predict_image(img_path)
    print(f'{img_file}: {result} (confidence: {confidence:.2f})')

print('-' * 40)
print('Testing complete!')
"@

$testCode | Out-File -FilePath "test_model.py"
python test_model.py
