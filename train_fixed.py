import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import requests
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

print('Fixed Professional Helipad Detection Training')
print('TensorFlow version:', tf.__version__)

def load_helipad_data():
    """Load and process the helipad datasets"""
    
    # Load the sample dataset (local images)
    sample_df = pd.read_csv('data/raw/csv/Sample_Helipad_Data.csv')
    print(f'Sample dataset: {len(sample_df)} entries')
    
    # Load the large dataset (URL images)  
    large_df = pd.read_csv('data/raw/csv/Helipad_DataBase_annotated.csv')
    print(f'Large dataset: {len(large_df)} entries')
    
    images = []
    labels = []
    
    # Process sample dataset (local files)
    print('Processing sample dataset...')
    for idx, row in sample_df.iterrows():
        image_filename = row['url']  # This contains local filenames
        
        # Try to find the image file
        possible_paths = [
            f'data/raw/{image_filename}',
            f'data/raw/images/{image_filename}',
            f'data/processed/{image_filename}',
            f'data/{image_filename}'
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
                labels.append(1)  # All sample entries are helipads
        else:
            print(f'Local image not found: {image_filename}')
    
    print(f'Loaded {len(images)} images from sample dataset')
    
    # Process subset of large dataset (download from URLs)
    print('Processing large dataset (downloading images)...')
    
    # Filter for valid helipads and create negative samples
    helipad_entries = large_df[large_df['groundtruth'] == 1].head(50)  # 50 positive samples
    non_helipad_entries = large_df[large_df['groundtruth'] == 0].head(50)  # 50 negative samples
    
    # Download helipad images
    downloaded = 0
    for idx, row in helipad_entries.iterrows():
        if downloaded >= 50:  # Limit downloads
            break
        img = download_and_preprocess_image(row['url'])
        if img is not None:
            images.append(img)
            labels.append(1)  # Helipad
            downloaded += 1
        
        if downloaded % 10 == 0:
            print(f'Downloaded {downloaded} helipad images...')
    
    # Download non-helipad images
    downloaded = 0
    for idx, row in non_helipad_entries.iterrows():
        if downloaded >= 50:  # Limit downloads
            break
        img = download_and_preprocess_image(row['url'])
        if img is not None:
            images.append(img)
            labels.append(0)  # No helipad
            downloaded += 1
        
        if downloaded % 10 == 0:
            print(f'Downloaded {downloaded} non-helipad images...')
    
    return np.array(images), np.array(labels)

def download_and_preprocess_image(url, timeout=5):
    """Download image from URL and preprocess it"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img = np.array(img)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            return img
        else:
            return None
    except Exception as e:
        return None

def load_and_preprocess_image(img_path):
    """Load and preprocess local image file"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return img
    except Exception as e:
        return None

def create_fixed_model():
    """Create a fixed CNN model without problematic metrics"""
    
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(224, 224, 3)),
        
        # Data augmentation
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.1),
        
        # Convolutional layers
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
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile with only accuracy metric to avoid shape issues
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Only use accuracy to avoid shape conflicts
    )
    
    return model

def main():
    print('Loading professional helipad dataset...')
    
    # Load data from CSV files
    X, y = load_helipad_data()
    
    if len(X) == 0:
        print('No data loaded! Please check your CSV files and image paths.')
        return
    
    print(f'Total images loaded: {len(X)}')
    print(f'Helipad images: {np.sum(y == 1)}')
    print(f'No-helipad images: {np.sum(y == 0)}')
    
    # Ensure we have enough data for splitting
    if len(X) < 10:
        print('Not enough data for training. Need at least 10 images.')
        return
    
    # Split data
    test_size = min(0.2, 2/len(X))  # Ensure at least 2 samples for test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f'Training samples: {len(X_train)}')
    print(f'Test samples: {len(X_test)}')
    
    # Create model
    model = create_fixed_model()
    print('Fixed model architecture:')
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
            'models/fixed_helipad_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print('Starting fixed training...')
    
    # Use smaller batch size to avoid shape issues
    batch_size = min(16, len(X_train) // 4) if len(X_train) >= 4 else 1
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=batch_size,
        validation_split=0.2 if len(X_train) > 5 else 0,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    if len(X_test) > 0:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test accuracy: {test_accuracy:.4f}')
    else:
        test_accuracy = 0
        print('No test data available')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/professional_helipad_model.h5')
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/fixed_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('=' * 60)
    print('🎉 FIXED TRAINING COMPLETE!')
    print('=' * 60)
    print(f'Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
    print('Model saved as: professional_helipad_model.h5')

if __name__ == '__main__':
    main()
