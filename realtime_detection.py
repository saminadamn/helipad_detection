import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('models/helipad_model.h5')

def detect_helipad_realtime():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = model.predict(img, verbose=0)
        confidence = np.max(prediction)
        class_idx = np.argmax(prediction)
        
        result = 'HELIPAD DETECTED!' if class_idx == 1 else 'No Helipad'
        color = (0, 255, 0) if class_idx == 1 else (0, 0, 255)
        
        # Draw result
        cv2.putText(frame, f'{result} ({confidence:.2f})', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Helipad Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('Starting real-time helipad detection...')
    print('Press Q to quit')
    detect_helipad_realtime()
