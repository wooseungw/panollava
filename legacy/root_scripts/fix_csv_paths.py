import pandas as pd
import os

def fix_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    # Check if 'url' column exists
    if 'url' in df.columns:
        # Prepend 'CORA/' to 'url' column if not already there
        df['url'] = df['url'].apply(lambda x: os.path.join('CORA', x) if not x.startswith('CORA') else x)
        
        # Rename columns to match config
        df = df.rename(columns={'url': 'image_path', 'query': 'instruction', 'annotation': 'output'})
        
        # Save to new csv
        df.to_csv(output_path, index=False)
        print(f"Fixed CSV saved to {output_path}")
    else:
        print(f"Error: 'url' column not found in {input_path}")

# Fix train.csv
fix_csv('CORA/data/quic360/train.csv', 'CORA/data/quic360/train_fixed.csv')

# Fix test.csv
fix_csv('CORA/data/quic360/test.csv', 'CORA/data/quic360/test_fixed.csv')
