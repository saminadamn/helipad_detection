import tensorflow as tf
import cv2
import numpy as np
import os

def test_professional_model():
    print('Testing Professional Helipad Detection Model')
    print('=' * 50)
    
    # Load the professional model
    try:
        model = tf.keras.models.load_model('models/professional_helipad_model.h5')
        print('✅ Model loaded successfully!')
    except:
        print('❌ Model not found. Please run training first.')
        return
    
    def predict_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, 0
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img, verbose=0)
        confidence = np.max(prediction)
        class_idx = np.argmax(prediction)
        
        return 'HELIPAD DETECTED' if class_idx == 1 else 'NO HELIPAD', confidence
    
    # Test on your real image
    if os.path.exists('Aug_Illustration.PNG'):
        result, confidence = predict_image('Aug_Illustration.PNG')
        if result:
            print(f'🚁 Aug_Illustration.PNG: {result}')
            print(f'📊 Confidence: {confidence:.4f} ({confidence*100:.1f}%)')
            
            if result == 'HELIPAD DETECTED' and confidence > 0.8:
                print('🎉 EXCELLENT! High confidence helipad detection!')
            elif result == 'HELIPAD DETECTED':
                print('✅ GOOD! Helipad detected with moderate confidence.')
            else:
                print('⚠️ Model classified as no helipad.')
        else:
            print('❌ Error processing image')
    else:
        print('❌ Aug_Illustration.PNG not found')
    
    # Test on any other images in the directory
    print('\nTesting other images in directory:')
    for file in os.listdir('.'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file != 'Aug_Illustration.PNG':
            result, confidence = predict_image(file)
            if result:
                print(f'📸 {file}: {result} ({confidence:.3f})')

if __name__ == '__main__':
    test_professional_model()
