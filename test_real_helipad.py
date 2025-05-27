import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your model
model = tf.keras.models.load_model('models/helipad_model.h5')

def analyze_real_helipad_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f'Could not load: {image_path}')
        return
    
    print(f'Analyzing: {image_path}')
    print(f'Image size: {img.shape[1]}x{img.shape[0]} pixels')
    
    # Create a copy for display
    display_img = img.copy()
    
    # Preprocess for AI model
    processed = cv2.resize(img, (224, 224))
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    processed = processed.astype(np.float32) / 255.0
    processed = np.expand_dims(processed, axis=0)
    
    # Get prediction
    prediction = model.predict(processed, verbose=0)
    confidence = np.max(prediction)
    class_idx = np.argmax(prediction)
    
    # Determine result
    if class_idx == 1:
        result = 'HELIPADS DETECTED!'
        color = (0, 255, 0)  # Green
        status = '✅ SUCCESS'
    else:
        result = 'NO HELIPADS DETECTED'
        color = (0, 0, 255)  # Red  
        status = '❌ MISSED'
    
    # Add result text to image
    cv2.putText(display_img, result, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    cv2.putText(display_img, f'Confidence: {confidence:.3f} ({confidence*100:.1f}%)', 
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(display_img, 'AI Analysis of Real Helipad Image', 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show the result
    cv2.imshow('Real Helipad Detection Test', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print detailed results
    print('=' * 60)
    print('🚁 REAL HELIPAD IMAGE ANALYSIS RESULTS')
    print('=' * 60)
    print(f'Status: {status}')
    print(f'Prediction: {result}')
    print(f'Confidence: {confidence:.4f} ({confidence*100:.2f}%)')
    print(f'Raw prediction scores: {prediction[0]}')
    print('=' * 60)
    
    # Expected result analysis
    print('📊 ANALYSIS:')
    print('This image contains REAL helipads (visible as square pads with H markings)')
    if class_idx == 1 and confidence > 0.7:
        print('🎉 EXCELLENT! Your AI correctly identified the helipads with high confidence!')
    elif class_idx == 1 and confidence > 0.5:
        print('✅ GOOD! Your AI detected helipads but with moderate confidence.')
    elif class_idx == 1:
        print('⚠️  DETECTED but with low confidence. Model might need more training data.')
    else:
        print('❌ MISSED! The AI did not detect the helipads. This suggests:')
        print('   - The synthetic training data differs from real imagery')
        print('   - More diverse training data needed')
        print('   - Model may need fine-tuning for aerial/satellite images')
    
    return class_idx, confidence

# Test the real helipad image
if __name__ == '__main__':
    analyze_real_helipad_image('Aug_Illustration.PNG')
