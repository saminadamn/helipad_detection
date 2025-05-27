import pandas as pd
import os

def explore_data():
    print('Data Explorer for Helipad Detection')
    print('=' * 50)
    
    # Find all CSV files
    csv_files = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print('No CSV files found in data directory!')
        return
    
    print('Found CSV files:')
    for i, csv_file in enumerate(csv_files):
        print(f'{i+1}. {csv_file}')
    
    # Examine each CSV file
    for csv_file in csv_files:
        print(f'\n📊 Examining: {csv_file}')
        print('-' * 40)
        
        try:
            df = pd.read_csv(csv_file)
            print(f'Shape: {df.shape}')
            print(f'Columns: {df.columns.tolist()}')
            print('\nFirst 5 rows:')
            print(df.head())
            
            # Check for image files
            image_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['image', 'file', 'path', 'img']):
                    image_columns.append(col)
            
            if image_columns:
                print(f'\nPossible image columns: {image_columns}')
                
                # Check if image files exist
                for img_col in image_columns:
                    sample_path = df[img_col].iloc[0] if len(df) > 0 else None
                    if sample_path:
                        # Try different base paths
                        paths_to_try = [
                            sample_path,
                            os.path.join('data/raw', sample_path),
                            os.path.join('data/processed', sample_path),
                            os.path.join('data/raw', os.path.basename(sample_path))
                        ]
                        
                        for path in paths_to_try:
                            if os.path.exists(path):
                                print(f'✅ Sample image found: {path}')
                                break
                        else:
                            print(f'❌ Sample image not found: {sample_path}')
            
            # Check for label columns
            label_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['label', 'class', 'category', 'helipad', 'target']):
                    label_columns.append(col)
            
            if label_columns:
                print(f'\nPossible label columns: {label_columns}')
                for label_col in label_columns:
                    unique_values = df[label_col].unique()
                    print(f'{label_col} unique values: {unique_values}')
            
        except Exception as e:
            print(f'Error reading {csv_file}: {e}')
    
    # Check image directories
    print(f'\n📁 Image Directory Contents:')
    print('-' * 40)
    
    image_dirs = ['data/raw', 'data/processed']
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f'{img_dir}: {len(image_files)} image files')
            if image_files:
                print(f'  Sample files: {image_files[:3]}')
        else:
            print(f'{img_dir}: Directory not found')

if __name__ == '__main__':
    explore_data()
