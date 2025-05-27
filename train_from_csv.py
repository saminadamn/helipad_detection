import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print('CSV-Based Helipad Detection Training')
print('TensorFlow version:', tf.__version__)

def find_csv_file():
    # Look for CSV files in common locations
    possible_paths = [
        'data/dataset.yaml',
        'data/raw/annotations.csv',
        'data/raw/labels.csv', 
        'data/raw/dataset.csv',
        'data/processed/annotations.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f'Found data file: {path}')
            return path
    
    # List all CSV files in data directory
    csv_files = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if csv_files:
        print('Found CSV files:')
        for i, csv_file in enumerate(csv_files):
            print(f'{i}: {csv_file}')
        return csv_files[0]  # Use first one found
    
    return None

def load_data_from_csv(csv_path, image_base_path='data/raw'):
    print(f'Loading data from: {csv_path}')
    
    # Try to read the CSV file
    try:
        if csv_path.endswith('.yaml'):
            # Handle YAML file (common in ML datasets)
            import yaml
            with open(csv_path, 'r') as f:
                data = yaml.safe_load(f)
            print('YAML file detected. Please convert to CSV format or provide CSV file.')
            return None, None
        
        df = pd.read_csv(csv_path)
        print(f'CSV loaded successfully. Shape: {df.shape}')
        print('Columns:', df.columns.tolist())
        print('First few rows:')
        print(df.head())
        
    except Exception as e:
        print(f'Error reading CSV: {e}')
        return None, None
    
    # Try to identify image and label columns
    image_col = None
    label_col = None
    
    # Common column names for images
    image_columns = ['image', 'filename', 'file', 'image_path', 'path', 'img']
    for col in df.columns:
        if col.lower() in image_columns:
            image_col = col
            break
    
    # Common column names for labels
    label_columns = ['label', 'class', 'category', 'helipad', 'target', 'y']
    for col in df.columns:
        if col.lower() in label_columns:
            label_col = col
            break
    
    if image_col is None:
        print('Could not find image column. Available columns:', df.columns.tolist())
        image_col = input('Enter the column name for images: ')
    
    if label_col is None:
        print('Could not find label column. Available columns:', df.columns.tolist())
        label_col = input('Enter the column name for labels: ')
    
    print(f'Using image column: {image_col}')
    print(f'Using label column: {label_col}')
    
    # Load images and labels
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        image_path = os.path.join(image_base_path, row[image_col])
        
        # Try different image path combinations
        if not os.path.exists(image_path):
            # Try without base path
            image_path = row[image_col]
        if not os.path.exists(image_path):
            # Try in processed folder
            image_path = os.path.join('data/processed', row[image_col])
        if not os.path.exists(image_path):
            # Try in raw folder
            image_path = os.path.join('data/raw', os.path.basename(row[image_col]))
        
        if os.path.exists(image_path):
            img = load_and_preprocess_image(image_path)
            if img is not None:
                images.append(img)
                
                # Process label
                label = row[label_col]
                if isinstance(label, str):
                    # Convert string labels to numeric
                    if label.lower() in ['helipad', 'yes', 'true', '1', 'positive']:
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    # Assume numeric label
                    labels.append(int(label))
        else:
            print(f'Image not found: {image_path}')
    
    print(f'Successfully loaded {len(images)} images')
    return np.array(images), np.array(labels)

def load_and_preprocess_image(img_path):
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

def create_enhanced_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        
        # Data augmentation
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomFlip('vertical'),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2),
        
        # Convolutional base
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Classifier
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Find and load CSV data
    csv_path = find_csv_file()
    if csv_path is None:
        print('No CSV file found! Please ensure you have a CSV file with image paths and labels.')
        return
    
    # Load data from CSV
    X, y = load_data_from_csv(csv_path)
    
    if X is None or len(X) == 0:
        print('No data loaded! Please check your CSV file and image paths.')
        return
    
    print(f'Total images loaded: {len(X)}')
    print(f'Helipad images: {np.sum(y == 1)}')
    print(f'No-helipad images: {np.sum(y == 0)}')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f'Training samples: {len(X_train)}')
    print(f'Test samples: {len(X_test)}')
    
    # Create and train model
    model = create_enhanced_model()
    print('Model architecture:')
    model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'models/csv_trained_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print('Starting training with CSV data...')
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Final test accuracy: {test_accuracy:.4f}')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/csv_helipad_model.h5')
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/csv_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('=' * 60)
    print('🎉 CSV-BASED TRAINING COMPLETE!')
    print('=' * 60)
    print(f'Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
    print('Model saved as: csv_helipad_model.h5')

if __name__ == '__main__':
    main()
