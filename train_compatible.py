import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import random
import json
from datetime import datetime

print("🚁 HELIPAD DETECTION - COMPATIBLE TRAINING")
print("=" * 60)

# Configuration
HELIPAD_DATA_PATH = r'C:\helipad_detection\data_image'
MODEL_OUTPUT_PATH = 'models'
IMG_SIZE = (224, 224)
EPOCHS = 30
BATCH_SIZE = 16  # Reduced batch size for stability
LEARNING_RATE = 0.001

def load_helipad_images(data_path):
    """Load all helipad images from the specified directory"""
    
    print(f"📂 Loading helipad images from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Directory not found: {data_path}")
        return np.array([]), np.array([])
    
    images = []
    labels = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(data_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"📊 Found {len(image_files)} image files")
    
    # Load and preprocess images
    loaded_count = 0
    for img_file in image_files:
        img_path = os.path.join(data_path, img_file)
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️  Could not load: {img_file}")
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            img = cv2.resize(img, IMG_SIZE)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(1)  # All images are helipads
            loaded_count += 1
            
            if loaded_count % 50 == 0:
                print(f"✅ Loaded {loaded_count} images...")
                
        except Exception as e:
            print(f"❌ Error loading {img_file}: {e}")
    
    print(f"✅ Successfully loaded {loaded_count} helipad images")
    return np.array(images), np.array(labels)

def create_augmented_helipads(helipad_images, target_count=400):
    """Create augmented versions of helipad images"""
    
    print(f"🔄 Creating {target_count} augmented helipad images...")
    
    augmented_images = []
    augmented_labels = []
    
    # Add original images
    for img in helipad_images:
        augmented_images.append(img)
        augmented_labels.append(1)
    
    # Create augmented versions
    while len(augmented_images) < target_count:
        # Pick random original image
        idx = random.randint(0, len(helipad_images) - 1)
        original_img = helipad_images[idx]
        
        # Apply numpy-based augmentations
        augmented_img = apply_numpy_augmentations(original_img)
        
        augmented_images.append(augmented_img)
        augmented_labels.append(1)
    
    return np.array(augmented_images), np.array(augmented_labels)

def apply_numpy_augmentations(img):
    """Apply augmentations using numpy operations"""
    
    # Convert to PIL for some operations
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    
    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        pil_img = pil_img.rotate(angle, fillcolor=(128, 128, 128))
    
    # Random brightness
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Random contrast
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Convert back to numpy
    augmented_np = np.array(pil_img).astype(np.float32) / 255.0
    
    # Random flip
    if random.random() > 0.5:
        augmented_np = np.fliplr(augmented_np)
    
    # Random noise
    if random.random() > 0.7:
        noise = np.random.normal(0, 0.02, augmented_np.shape)
        augmented_np = np.clip(augmented_np + noise, 0, 1)
    
    return augmented_np

def create_simple_negative_samples(count=400):
    """Create simple negative samples"""
    
    print(f"🔄 Creating {count} negative samples...")
    
    images = []
    labels = []
    
    for i in range(count):
        # Create different scene types
        scene_type = random.choice(['urban', 'rural', 'water', 'random'])
        
        if scene_type == 'urban':
            img = create_simple_urban()
        elif scene_type == 'rural':
            img = create_simple_rural()
        elif scene_type == 'water':
            img = create_simple_water()
        else:
            img = create_random_pattern()
        
        images.append(img)
        labels.append(0)  # No helipad
        
        if (i + 1) % 100 == 0:
            print(f"✅ Created {i + 1} negative samples...")
    
    return np.array(images), np.array(labels)

def create_simple_urban():
    """Create simple urban scene"""
    img = np.full((*IMG_SIZE, 3), 0.5, dtype=np.float32)  # Gray base
    
    # Add some rectangles (buildings)
    for _ in range(random.randint(3, 6)):
        x1 = random.randint(0, IMG_SIZE[0] - 50)
        y1 = random.randint(0, IMG_SIZE[1] - 50)
        x2 = x1 + random.randint(20, 60)
        y2 = y1 + random.randint(30, 80)
        
        color = random.uniform(0.3, 0.7)
        img[y1:y2, x1:x2] = color
    
    return img

def create_simple_rural():
    """Create simple rural scene"""
    img = np.full((*IMG_SIZE, 3), [0.2, 0.6, 0.2], dtype=np.float32)  # Green base
    
    # Add some patches
    for _ in range(random.randint(2, 4)):
        x1 = random.randint(0, IMG_SIZE[0] - 40)
        y1 = random.randint(0, IMG_SIZE[1] - 40)
        x2 = x1 + random.randint(30, 70)
        y2 = y1 + random.randint(30, 70)
        
        color = [random.uniform(0.1, 0.4), random.uniform(0.4, 0.8), random.uniform(0.1, 0.4)]
        img[y1:y2, x1:x2] = color
    
    return img

def create_simple_water():
    """Create simple water scene"""
    img = np.full((*IMG_SIZE, 3), [0.2, 0.4, 0.7], dtype=np.float32)  # Blue base
    
    # Add some variation
    noise = np.random.normal(0, 0.05, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    return img

def create_random_pattern():
    """Create random pattern"""
    img = np.random.uniform(0, 1, (*IMG_SIZE, 3)).astype(np.float32)
    
    # Smooth it a bit
    from scipy import ndimage
    try:
        img = ndimage.gaussian_filter(img, sigma=1.0)
    except:
        pass  # If scipy not available, use as is
    
    return img

def create_simple_model():
    """Create a simple, compatible CNN model"""
    
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        
        # First block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        
        # Classifier
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_on_custom_image(model, image_path):
    """Test the model on a custom image"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load and preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, 0), verbose=0)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(f"Original Image\n{os.path.basename(image_path)}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.title("Processed for Model")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
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
    plt.savefig('models/test_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Prediction: {classes[predicted_class]} ({confidence:.2%} confidence)")

def main():
    print("🔄 Loading real helipad dataset...")
    
    # Load real helipad images
    helipad_images, helipad_labels = load_helipad_images(HELIPAD_DATA_PATH)
    
    if len(helipad_images) == 0:
        print("❌ No helipad images loaded! Please check the path.")
        return
    
    print(f"✅ Loaded {len(helipad_images)} real helipad images")
    
    # Create augmented helipad data
    print("🔄 Creating augmented helipad dataset...")
    aug_helipad_images, aug_helipad_labels = create_augmented_helipads(
        helipad_images, target_count=400
    )
    
    # Create negative samples
    print("🔄 Creating negative samples...")
    negative_images, negative_labels = create_simple_negative_samples(count=400)
    
    # Combine all data
    all_images = np.concatenate([aug_helipad_images, negative_images])
    all_labels = np.concatenate([aug_helipad_labels, negative_labels])
    
    # Shuffle the data
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    print(f"📊 Final dataset: {len(all_images)} images")
    print(f"   Helipad images: {np.sum(all_labels == 1)}")
    print(f"   No-helipad images: {np.sum(all_labels == 0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"🔄 Training samples: {len(X_train)}")
    print(f"🔄 Test samples: {len(X_test)}")
    
    # Create model
    model = create_simple_model()
    print("🏗️  Simple model architecture created")
    model.summary()
    
    # Training callbacks
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'{MODEL_OUTPUT_PATH}/best_helipad_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("🔍 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Detailed predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n" + "="*60)
    print("📊 CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['No Helipad', 'Helipad']))
    
    print("\n" + "="*60)
    print("🎯 CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    # Save final model
    model.save(f'{MODEL_OUTPUT_PATH}/final_helipad_model.h5')
    
    # Save metadata
    metadata = {
        'model_name': 'final_helipad_model',
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'real_helipad_images': len(helipad_images),
            'total_helipad_samples': int(np.sum(all_labels == 1)),
            'total_negative_samples': int(np.sum(all_labels == 0)),
            'total_samples': len(all_images)
        },
        'performance': {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss)
        },
        'training_config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'image_size': IMG_SIZE
        },
        'class_names': ['No Helipad', 'Helipad']
    }
    
    with open(f'{MODEL_OUTPUT_PATH}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Training history
    plt.subplot(2, 3, 1)
    plt.plot(history.history['accuracy'], label='Training', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['loss'], label='Training', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Confusion matrix
    plt.subplot(2, 3, 3)
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    class_names = ['No Helipad', 'Helipad']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    # Sample predictions
    for i in range(3):
        plt.subplot(2, 3, 4 + i)
        if i < len(X_test):
            plt.imshow(X_test[i])
            pred = model.predict(np.expand_dims(X_test[i], 0), verbose=0)
            pred_class = 'Helipad' if np.argmax(pred) == 1 else 'No Helipad'
            actual_class = 'Helipad' if y_test[i] == 1 else 'No Helipad'
            confidence = np.max(pred)
            color = 'green' if pred_class == actual_class else 'red'
            plt.title(f'P: {pred_class}\nA: {actual_class}\nConf: {confidence:.2f}', 
                     fontsize=8, color=color)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_OUTPUT_PATH}/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"✅ Used {len(helipad_images)} real helipad images")
    print("✅ Model saved as: final_helipad_model.h5")
    
    # Test on custom image if available
    test_images = ["Aug_Illustration.PNG", "test_helipad.jpg", "test_image.png"]
    for test_img in test_images:
        if os.path.exists(test_img):
            print(f"\n🔍 Testing on: {test_img}")
            test_on_custom_image(model, test_img)
            break
    else:
        print("\n⚠️  No test images found. Place a test image in the current directory!")

if __name__ == '__main__':
    main()
