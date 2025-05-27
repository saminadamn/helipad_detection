import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageEnhance
import random
import json
from datetime import datetime

print("🚁 SIMPLE HELIPAD DETECTION TRAINING")
print("=" * 50)

# Configuration
HELIPAD_DATA_PATH = r'C:\helipad_detection\data_image'
MODEL_OUTPUT_PATH = 'models'
IMG_SIZE = (224, 224)
EPOCHS = 20
BATCH_SIZE = 16

def load_helipad_images(data_path):
    """Load all helipad images"""
    
    print(f"📂 Loading images from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Directory not found: {data_path}")
        return np.array([]), np.array([])
    
    images = []
    labels = []
    
    # Get image files
    image_files = []
    for file in os.listdir(data_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    print(f"📊 Found {len(image_files)} image files")
    
    # Load images
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(data_path, img_file)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(1)  # Helipad
            
            if (i + 1) % 50 == 0:
                print(f"✅ Loaded {i + 1} images...")
                
        except Exception as e:
            print(f"❌ Error loading {img_file}: {e}")
    
    print(f"✅ Loaded {len(images)} helipad images")
    return np.array(images), np.array(labels)

def create_simple_augmentations(images, target_count=300):
    """Create simple augmented versions"""
    
    print(f"🔄 Creating {target_count} augmented images...")
    
    aug_images = []
    aug_labels = []
    
    # Add originals
    for img in images:
        aug_images.append(img)
        aug_labels.append(1)
    
    # Create augmented versions
    while len(aug_images) < target_count:
        idx = random.randint(0, len(images) - 1)
        original = images[idx]
        
        # Simple augmentations
        augmented = original.copy()
        
        # Random flip
        if random.random() > 0.5:
            augmented = np.fliplr(augmented)
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            augmented = np.clip(augmented * factor, 0, 1)
        
        # Random noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)
        
        aug_images.append(augmented)
        aug_labels.append(1)
    
    return np.array(aug_images), np.array(aug_labels)

def create_negative_samples(count=300):
    """Create simple negative samples"""
    
    print(f"🔄 Creating {count} negative samples...")
    
    images = []
    labels = []
    
    for i in range(count):
        # Create random colored rectangles (non-helipad scenes)
        img = np.random.uniform(0, 1, (*IMG_SIZE, 3)).astype(np.float32)
        
        # Add some structure
        for _ in range(random.randint(2, 5)):
            x1 = random.randint(0, IMG_SIZE[0] - 30)
            y1 = random.randint(0, IMG_SIZE[1] - 30)
            x2 = x1 + random.randint(20, 50)
            y2 = y1 + random.randint(20, 50)
            
            color = random.uniform(0, 1)
            img[y1:y2, x1:x2] = color
        
        images.append(img)
        labels.append(0)  # No helipad
        
        if (i + 1) % 100 == 0:
            print(f"✅ Created {i + 1} negative samples...")
    
    return np.array(images), np.array(labels)

def create_basic_model():
    """Create a basic CNN model"""
    
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_image(model, image_path):
    """Test model on an image"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()
    
    # Preprocess
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    
    # Predict
    prediction = model.predict(np.expand_dims(img, 0), verbose=0)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    
    # Show results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title(f"Test Image: {os.path.basename(image_path)}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    classes = ['No Helipad', 'Helipad']
    colors = ['red', 'green']
    bars = plt.bar(classes, prediction[0], color=colors, alpha=0.7)
    plt.title(f"Prediction: {classes[predicted_class]}\nConfidence: {confidence:.2%}")
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    
    for bar, prob in zip(bars, prediction[0]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Result: {classes[predicted_class]} ({confidence:.2%} confidence)")

def main():
    print("🔄 Starting simple training process...")
    
    # Load helipad images
    helipad_images, helipad_labels = load_helipad_images(HELIPAD_DATA_PATH)
    
    if len(helipad_images) == 0:
        print("❌ No images loaded! Check your path.")
        return
    
    print(f"✅ Loaded {len(helipad_images)} real helipad images")
    
    # Create augmented data
    aug_helipad_images, aug_helipad_labels = create_simple_augmentations(
        helipad_images, target_count=300
    )
    
    # Create negative samples
    negative_images, negative_labels = create_negative_samples(count=300)
    
    # Combine data
    all_images = np.concatenate([aug_helipad_images, negative_images])
    all_labels = np.concatenate([aug_helipad_labels, negative_labels])
    
    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    print(f"📊 Total dataset: {len(all_images)} images")
    print(f"   Helipad: {np.sum(all_labels == 1)}")
    print(f"   No helipad: {np.sum(all_labels == 0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"🔄 Training: {len(X_train)} samples")
    print(f"🔄 Testing: {len(X_test)} samples")
    
    # Create model
    model = create_basic_model()
    print("🏗️  Model created")
    model.summary()
    
    # Train (no callbacks to avoid errors)
    print("🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    print("🔍 Evaluating...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n" + "="*50)
    print("📊 RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['No Helipad', 'Helipad']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    # Save model
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    model.save(f'{MODEL_OUTPUT_PATH}/simple_helipad_model.h5')
    
    # Save info
    info = {
        'model_name': 'simple_helipad_model',
        'timestamp': datetime.now().isoformat(),
        'test_accuracy': float(test_accuracy),
        'real_images_used': len(helipad_images),
        'total_training_samples': len(all_images),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }
    
    with open(f'{MODEL_OUTPUT_PATH}/model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Training history
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Confusion matrix
    plt.subplot(2, 2, 3)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Sample predictions
    plt.subplot(2, 2, 4)
    if len(X_test) > 0:
        sample_idx = 0
        plt.imshow(X_test[sample_idx])
        pred = model.predict(np.expand_dims(X_test[sample_idx], 0), verbose=0)
        pred_class = 'Helipad' if np.argmax(pred) == 1 else 'No Helipad'
        actual_class = 'Helipad' if y_test[sample_idx] == 1 else 'No Helipad'
        plt.title(f'Sample\nPred: {pred_class}\nActual: {actual_class}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_OUTPUT_PATH}/training_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"✅ Model saved: simple_helipad_model.h5")
    print(f"✅ Used {len(helipad_images)} real helipad images")
    
    # Test on available images
    test_files = ["Aug_Illustration.PNG", "test.jpg", "test.png"]
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🔍 Testing on {test_file}...")
            test_image(model, test_file)
            break
    else:
        print("\n💡 Place a test image in the current directory to test it!")

if __name__ == '__main__':
    main()
