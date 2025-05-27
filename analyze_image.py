import tensorflow as tf
import cv2
import numpy as np
import sys

model = tf.keras.models.load_model('models/helipad_model.h5')

def analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f'Error: Could not load {image_path}')
        return
    
    # Preprocess
    processed = cv2.resize(img, (224, 224))
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    processed = processed.astype(np.float32) / 255.0
    processed = np.expand_dims(processed, axis=0)
    
    # Predict
    prediction = model.predict(processed, verbose=0)
    confidence = np.max(prediction)
    class_idx = np.argmax(prediction)
    
    result = 'HELIPAD DETECTED' if class_idx == 1 else 'NO HELIPAD'
    
    print(f'Image: {image_path}')
    print(f'Result: {result}')
    print(f'Confidence: {confidence:.3f} ({confidence*100:.1f}%)')
    
    # Show image with result
    color = (0, 255, 0) if class_idx == 1 else (0, 0, 255)
    cv2.putText(img, f'{result} ({confidence:.2f})', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Analysis Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python analyze_image.py <image_path>')
        print('Example: python analyze_image.py data/helipad/helipad_001.jpg')
    else:
        analyze_image(sys.argv[1])
