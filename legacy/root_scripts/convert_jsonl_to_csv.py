import pandas as pd
import json
import sys
from pathlib import Path

def convert_jsonl_to_csv(jsonl_path):
    jsonl_path = Path(jsonl_path)
    csv_path = jsonl_path.with_suffix('.csv')
    
    print(f"Converting {jsonl_path} to {csv_path}...")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Ensure required columns exist
    if 'prediction' not in df.columns or 'reference' not in df.columns:
        print("Error: JSONL must contain 'prediction' and 'reference' fields.")
        return None
        
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Successfully created {csv_path} with {len(df)} rows.")
    return csv_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_jsonl_to_csv.py <jsonl_file>")
        sys.exit(1)
        
    convert_jsonl_to_csv(sys.argv[1])
