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
from urllib.parse import urlparse
import time

print('Professional Helipad Detection Training')
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
                # All entries in sample dataset are helipads
                labels.append(1)
        else:
            print(f'Local image not found: {image_filename}')
    
    print(f'Loaded {len(images)} images from sample dataset')
    
    # Process subset of large dataset (download from URLs)
    print('Processing large dataset (downloading images)...')
    
    # Filter for valid helipads and create negative samples
    helipad_entries = large_df[large_df['groundtruth'] == 1].head(100)  # 100 positive samples
    non_helipad_entries = large_df[large_df['groundtruth'] == 0].head(100)  # 100 negative samples
    
    # Download helipad images
    for idx, row in helipad_entries.iterrows():
        img = download_and_preprocess_image(row['url'])
        if img is not None:
            images.append(img)
            labels.append(1)  # Helipad
        
        if len(images) % 10 == 0:
            print(f'Downloaded {len(images)} images...')
    
    # Download non-helipad images
    for idx, row in non_helipad_entries.iterrows():
        img = download_and_preprocess_image(row['url'])
        if img is not None:
            images.append(img)
            labels.append(0)  # No helipad
        
        if len(images) % 10 == 0:
            print(f'Downloaded {len(images)} images...')
    
    return np.array(images), np.array(labels)

def download_and_preprocess_image(url, timeout=10):
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
        print(f'Error downloading {url}: {e}')
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
        print(f'Error loading {img_path}: {e}')
        return None

def create_professional_model():
    """Create a professional-grade CNN for helipad detection"""
    
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(224, 224, 3)),
        
        # Data augmentation for robustness
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
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Global pooling and classifier
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile with advanced optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
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
    
    # Split data strategically
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f'Training samples: {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    print(f'Test samples: {len(X_test)}')
    
    # Create professional model
    model = create_professional_model()
    print('Professional model architecture:')
    model.summary()
    
    # Advanced training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-8,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/professional_helipad_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print('Starting professional training...')
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Comprehensive evaluation
    print('Evaluating professional model...')
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_accuracy, test_precision, test_recall = test_results
    
    # Calculate F1 score
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    
    print('=' * 60)
    print('🎉 PROFESSIONAL TRAINING COMPLETE!')
    print('=' * 60)
    print(f'Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall:    {test_recall:.4f}')
    print(f'F1 Score:       {f1_score:.4f}')
    print('=' * 60)
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    model.save('models/helipad_professional_model.h5')
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy plot
    axes[0,0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0,0].set_title('Model Accuracy', fontsize=14)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0,1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0,1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0,1].set_title('Model Loss', fontsize=14)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Precision plot
    axes[1,0].plot(history.history['precision'], label='Training Precision', linewidth=2)
    axes[1,0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
    axes[1,0].set_title('Model Precision', fontsize=14)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Recall plot
    axes[1,1].plot(history.history['recall'], label='Training Recall', linewidth=2)
    axes[1,1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    axes[1,1].set_title('Model Recall', fontsize=14)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/professional_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('Models and results saved!')
    print('Ready for professional helipad detection! 🚁✨')

if __name__ == '__main__':
    main()
