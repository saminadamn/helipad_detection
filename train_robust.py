import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random

print('Enhanced Helipad Detection Training with Data Augmentation')
print('TensorFlow version:', tf.__version__)

def create_augmented_data(images, labels, target_count=200):
    """Create augmented versions of existing images"""
    
    augmented_images = []
    augmented_labels = []
    
    # Add original images
    for img, label in zip(images, labels):
        augmented_images.append(img)
        augmented_labels.append(label)
    
    # Create augmented versions
    while len(augmented_images) < target_count:
        # Pick a random original image
        idx = random.randint(0, len(images) - 1)
        original_img = images[idx]
        original_label = labels[idx]
        
        # Convert to PIL for augmentation
        pil_img = Image.fromarray((original_img * 255).astype(np.uint8))
        
        # Apply random augmentations
        augmented_pil = apply_augmentations(pil_img)
        
        # Convert back to numpy
        augmented_np = np.array(augmented_pil).astype(np.float32) / 255.0
        
        augmented_images.append(augmented_np)
        augmented_labels.append(original_label)
    
    return np.array(augmented_images), np.array(augmented_labels)

def apply_augmentations(pil_img):
    """Apply various augmentations to a PIL image"""
    
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
    
    # Random color adjustment
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random blur
    if random.random() > 0.3:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
    
    # Random crop and resize
    if random.random() > 0.5:
        width, height = pil_img.size
        crop_size = random.uniform(0.8, 1.0)
        new_width = int(width * crop_size)
        new_height = int(height * crop_size)
        
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        pil_img = pil_img.crop((left, top, left + new_width, top + new_height))
        pil_img = pil_img.resize((224, 224))
    
    return pil_img

def create_negative_samples(count=100):
    """Create synthetic negative samples (non-helipad images)"""
    images = []
    labels = []
    
    for i in range(count):
        # Create random background
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        pil_img = Image.fromarray(img)
        
        # Add random shapes (buildings, roads, etc.)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(pil_img)
        
        # Add random rectangles (buildings)
        for _ in range(random.randint(2, 8)):
            x1 = random.randint(0, 180)
            y1 = random.randint(0, 180)
            x2 = x1 + random.randint(20, 60)
            y2 = y1 + random.randint(20, 60)
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Convert to numpy and normalize
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        images.append(img_array)
        labels.append(0)  # No helipad
    
    return np.array(images), np.array(labels)

def load_sample_data():
    """Load the sample helipad data"""
    
    sample_df = pd.read_csv('data/raw/csv/Sample_Helipad_Data.csv')
    print(f'Sample dataset: {len(sample_df)} entries')
    
    images = []
    labels = []
    
    # Process sample dataset (local files)
    print('Processing sample dataset...')
    for idx, row in sample_df.iterrows():
        image_filename = row['url']
        
        # Try to find the image file
        possible_paths = [
            f'data/raw/{image_filename}',
            f'data/raw/images/{image_filename}',
            f'data/processed/{image_filename}',
            f'data/{image_filename}',
            f'data/helipad/{image_filename}'  # Check synthetic folder too
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path:
            img = load_and_preprocess_image(image_path)
            if img is not None:
                images.append(img)
                labels.append(1)  # Helipad
        else:
            print(f'Image not found: {image_filename}')
    
    print(f'Loaded {len(images)} real helipad images')
    return np.array(images), np.array(labels)

def load_and_preprocess_image(img_path):
    """Load and preprocess image file"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return img
    except Exception as e:
        print(f'Error loading {img_path}: {e}')
        return None

def create_robust_model():
    """Create a robust CNN model"""
    
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        
        # Built-in data augmentation
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomFlip('vertical'),
        keras.layers.RandomRotation(0.3),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2),
        
        # Convolutional base
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        
        # Classifier
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print('Loading and augmenting helipad data...')
    
    # Load sample data
    sample_images, sample_labels = load_sample_data()
    
    if len(sample_images) == 0:
        print('No sample images loaded! Please check your data paths.')
        return
    
    print(f'Loaded {len(sample_images)} sample images')
    
    # Create augmented helipad data
    print('Creating augmented helipad data...')
    helipad_images, helipad_labels = create_augmented_data(
        sample_images, sample_labels, target_count=200
    )
    
    # Create negative samples
    print('Creating negative samples...')
    negative_images, negative_labels = create_negative_samples(count=200)
    
    # Combine all data
    all_images = np.concatenate([helipad_images, negative_images])
    all_labels = np.concatenate([helipad_labels, negative_labels])
    
    print(f'Total dataset: {len(all_images)} images')
    print(f'Helipad images: {np.sum(all_labels == 1)}')
    print(f'No-helipad images: {np.sum(all_labels == 0)}')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f'Training samples: {len(X_train)}')
    print(f'Test samples: {len(X_test)}')
    
    # Create model
    model = create_robust_model()
    print('Robust model architecture:')
    model.summary()
    
    # Training callbacks
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
            'models/robust_helipad_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print('Starting robust training...')
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    model.save('models/professional_helipad_model.h5')
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Show sample images
    fig_samples = plt.figure(figsize=(10, 8))
    for i in range(min(16, len(X_test))):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[i])
        pred = model.predict(np.expand_dims(X_test[i], 0), verbose=0)
        pred_class = 'Helipad' if np.argmax(pred) == 1 else 'No Helipad'
        actual_class = 'Helipad' if y_test[i] == 1 else 'No Helipad'
        plt.title(f'P:{pred_class}\nA:{actual_class}', fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('models/sample_predictions.png', dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.savefig('models/robust_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('=' * 60)
    print('🎉 ROBUST TRAINING COMPLETE!')
    print('=' * 60)
    print(f'Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
    print('Model saved as: professional_helipad_model.h5')
    print('Sample predictions saved as: sample_predictions.png')

if __name__ == '__main__':
    main()
