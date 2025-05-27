import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import random
import json
from datetime import datetime
import seaborn as sns

print("🚁 HELIPAD DETECTION - REAL DATASET TRAINING")
print("=" * 60)

# Configuration
HELIPAD_DATA_PATH = r'C:\helipad_detection\data_image'
MODEL_OUTPUT_PATH = 'models'
IMG_SIZE = (224, 224)
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

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

def create_augmented_helipads(helipad_images, target_count=500):
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
        
        # Convert to PIL for augmentation
        pil_img = Image.fromarray((original_img * 255).astype(np.uint8))
        
        # Apply augmentations
        augmented_pil = apply_helipad_augmentations(pil_img)
        
        # Convert back to numpy
        augmented_np = np.array(augmented_pil).astype(np.float32) / 255.0
        
        augmented_images.append(augmented_np)
        augmented_labels.append(1)
    
    return np.array(augmented_images), np.array(augmented_labels)

def apply_helipad_augmentations(pil_img):
    """Apply realistic augmentations for helipad images"""
    
    # Random rotation (helipads can be viewed from any angle)
    if random.random() > 0.3:
        angle = random.uniform(-45, 45)
        pil_img = pil_img.rotate(angle, fillcolor=(128, 128, 128))
    
    # Random brightness (different lighting conditions)
    if random.random() > 0.4:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.6, 1.4))
    
    # Random contrast (weather conditions)
    if random.random() > 0.4:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Random color saturation
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random blur (atmospheric conditions)
    if random.random() > 0.6:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    
    # Random crop and resize (different distances)
    if random.random() > 0.4:
        width, height = pil_img.size
        crop_factor = random.uniform(0.7, 1.0)
        new_width = int(width * crop_factor)
        new_height = int(height * crop_factor)
        
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        pil_img = pil_img.crop((left, top, left + new_width, top + new_height))
        pil_img = pil_img.resize(IMG_SIZE)
    
    return pil_img

def create_diverse_negative_samples(count=500):
    """Create diverse negative samples (non-helipad images)"""
    
    print(f"🔄 Creating {count} diverse negative samples...")
    
    images = []
    labels = []
    
    for i in range(count):
        # Create different types of backgrounds
        img_type = random.choice(['urban', 'rural', 'water', 'forest', 'desert'])
        
        if img_type == 'urban':
            img = create_urban_scene()
        elif img_type == 'rural':
            img = create_rural_scene()
        elif img_type == 'water':
            img = create_water_scene()
        elif img_type == 'forest':
            img = create_forest_scene()
        else:  # desert
            img = create_desert_scene()
        
        # Convert to numpy and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        images.append(img_array)
        labels.append(0)  # No helipad
        
        if (i + 1) % 100 == 0:
            print(f"✅ Created {i + 1} negative samples...")
    
    return np.array(images), np.array(labels)

def create_urban_scene():
    """Create urban scene without helipad"""
    img = Image.new('RGB', IMG_SIZE, color=(120, 120, 120))  # Gray base
    draw = ImageDraw.Draw(img)
    
    # Add buildings
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, IMG_SIZE[0] - 50)
        y1 = random.randint(0, IMG_SIZE[1] - 50)
        x2 = x1 + random.randint(30, 80)
        y2 = y1 + random.randint(40, 100)
        color = (random.randint(80, 160), random.randint(80, 160), random.randint(80, 160))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Add roads
    for _ in range(random.randint(1, 3)):
        if random.random() > 0.5:  # Horizontal road
            y = random.randint(50, IMG_SIZE[1] - 50)
            draw.rectangle([0, y, IMG_SIZE[0], y + 20], fill=(60, 60, 60))
        else:  # Vertical road
            x = random.randint(50, IMG_SIZE[0] - 50)
            draw.rectangle([x, 0, x + 20, IMG_SIZE[1]], fill=(60, 60, 60))
    
    return img

def create_rural_scene():
    """Create rural scene without helipad"""
    img = Image.new('RGB', IMG_SIZE, color=(34, 139, 34))  # Green base
    draw = ImageDraw.Draw(img)
    
    # Add fields with different colors
    for _ in range(random.randint(2, 5)):
        x1 = random.randint(0, IMG_SIZE[0] // 2)
        y1 = random.randint(0, IMG_SIZE[1] // 2)
        x2 = x1 + random.randint(50, 100)
        y2 = y1 + random.randint(50, 100)
        color = (random.randint(20, 80), random.randint(100, 180), random.randint(20, 80))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    return img

def create_water_scene():
    """Create water scene without helipad"""
    img = Image.new('RGB', IMG_SIZE, color=(70, 130, 180))  # Blue base
    draw = ImageDraw.Draw(img)
    
    # Add waves/ripples
    for _ in range(random.randint(5, 15)):
        x = random.randint(0, IMG_SIZE[0])
        y = random.randint(0, IMG_SIZE[1])
        r = random.randint(5, 20)
        color = (random.randint(50, 100), random.randint(110, 150), random.randint(160, 200))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    return img

def create_forest_scene():
    """Create forest scene without helipad"""
    img = Image.new('RGB', IMG_SIZE, color=(34, 80, 34))  # Dark green base
    draw = ImageDraw.Draw(img)
    
    # Add trees (circles)
    for _ in range(random.randint(10, 20)):
        x = random.randint(0, IMG_SIZE[0])
        y = random.randint(0, IMG_SIZE[1])
        r = random.randint(10, 30)
        color = (random.randint(20, 60), random.randint(80, 140), random.randint(20, 60))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    return img

def create_desert_scene():
    """Create desert scene without helipad"""
    img = Image.new('RGB', IMG_SIZE, color=(194, 178, 128))  # Sandy base
    draw = ImageDraw.Draw(img)
    
    # Add sand dunes
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, IMG_SIZE[0])
        y1 = random.randint(0, IMG_SIZE[1])
        x2 = x1 + random.randint(30, 80)
        y2 = y1 + random.randint(20, 50)
        color = (random.randint(180, 220), random.randint(160, 200), random.randint(100, 150))
        draw.ellipse([x1, y1, x2, y2], fill=color)
    
    return img

def create_advanced_model():
    """Create an advanced CNN model for helipad detection"""
    
    model = keras.Sequential([
        keras.layers.Input(shape=(*IMG_SIZE, 3)),
        
        # Data augmentation layers
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomFlip('vertical'),
        keras.layers.RandomRotation(0.3),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2),
        
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Third convolutional block
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Fourth convolutional block
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.GlobalAveragePooling2D(),
        
        # Classifier head
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile with advanced optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
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
    plt.savefig('models/real_image_prediction.png', dpi=300, bbox_inches='tight')
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
        helipad_images, target_count=600
    )
    
    # Create diverse negative samples
    print("🔄 Creating diverse negative samples...")
    negative_images, negative_labels = create_diverse_negative_samples(count=600)
    
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
    model = create_advanced_model()
    print("🏗️  Advanced model architecture created")
    model.summary()
    
    # Training callbacks
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'{MODEL_OUTPUT_PATH}/best_real_helipad_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("🚀 Starting advanced training...")
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
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_accuracy, test_precision, test_recall = test_results
    
    # Detailed predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n" + "="*60)
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['No Helipad', 'Helipad']))
    
    print("\n" + "="*60)
    print("🎯 CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    # Save final model
    model.save(f'{MODEL_OUTPUT_PATH}/final_real_helipad_model.h5')
    
    # Save comprehensive metadata
    metadata = {
        'model_name': 'final_real_helipad_model',
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'real_helipad_images': len(helipad_images),
            'total_helipad_samples': int(np.sum(all_labels == 1)),
            'total_negative_samples': int(np.sum(all_labels == 0)),
            'total_samples': len(all_images)
        },
        'performance': {
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
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
    
    with open(f'{MODEL_OUTPUT_PATH}/real_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create comprehensive visualization
    plt.figure(figsize=(20, 12))
    
    # Training history
    plt.subplot(2, 4, 1)
    plt.plot(history.history['accuracy'], label='Training', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 2)
    plt.plot(history.history['loss'], label='Training', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Confusion matrix
    plt.subplot(2, 4, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Helipad', 'Helipad'],
                yticklabels=['No Helipad', 'Helipad'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Sample predictions
    for i in range(5):
        plt.subplot(2, 4, 4 + i)
        if i < len(X_test):
            plt.imshow(X_test[i])
            pred = model.predict(np.expand_dims(X_test[i], 0), verbose=0)
            pred_class = 'Helipad' if np.argmax(pred) == 1 else 'No Helipad'
            actual_class = 'Helipad' if y_test[i] == 1 else 'No Helipad'
            confidence = np.max(pred)
            plt.title(f'P: {pred_class}\nA: {actual_class}\nConf: {confidence:.2f}', fontsize=8)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_OUTPUT_PATH}/comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("🎉 REAL DATASET TRAINING COMPLETE!")
    print("="*80)
    print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"✅ Test Precision: {test_precision:.4f}")
    print(f"✅ Test Recall: {test_recall:.4f}")
    print(f"✅ Used {len(helipad_images)} real helipad images")
    print("✅ Model saved as: final_real_helipad_model.h5")
    
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
