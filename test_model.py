import tensorflow as tf
import cv2
import numpy as np
import os

model = tf.keras.models.load_model('models/helipad_model.h5')

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img, verbose=0)
    confidence = np.max(prediction)
    class_idx = np.argmax(prediction)
    
    class_names = ['No Helipad', 'Helipad']
    return class_names[class_idx], confidence

print('Testing helipad images:')
for img_file in os.listdir('data/helipad')[:5]:
    result, conf = predict_image(f'data/helipad/{img_file}')
    print(f'{img_file}: {result} ({conf:.3f})')

print('Testing no-helipad images:')
for img_file in os.listdir('data/no_helipad')[:5]:
    result, conf = predict_image(f'data/no_helipad/{img_file}')
    print(f'{img_file}: {result} ({conf:.3f})')
