"""Build Stage 1 (Vision) CSV: image-only dataset from Stanford pano/rgb + Zind + SUN360."""
import glob
import os
import pandas as pd

OUTPUT_DIR = "/data/1_personal/4_SWWOO/panollava/data"

# Dataset paths
DATASETS = {
    "stanford_pano": {
        "pattern": "/data/1_personal/4_SWWOO/stanford2d3d/extracted_data/area_*/area_*/pano/rgb/*.png",
        "desc": "Stanford2D3D panoramic RGB",
    },
    "zind": {
        "pattern": "/data/1_personal/4_SWWOO/zind/zind/*/panos/*.jpg",
        "desc": "Zillow Indoor (Zind) panoramas",
    },
    "sun360_train": {
        "pattern": "/data/1_personal/4_SWWOO/SUN360/train/RGB/*.jpg",
        "desc": "SUN360 train set",
    },
}

# For validation, use a small held-out portion
VAL_RATIO = 0.05


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_paths = []
    for name, info in DATASETS.items():
        paths = sorted(glob.glob(info["pattern"]))
        # Verify files exist
        valid = [p for p in paths if os.path.isfile(p)]
        print(f"{name}: {len(valid)} images found ({info['desc']})")
        all_paths.extend(valid)
    
    print(f"\nTotal Stage 1 images: {len(all_paths)}")
    
    # Create DataFrame with dummy query/annotation for VICReg-only training
    df = pd.DataFrame({
        "url": all_paths,
        "query": ["Describe this panoramic scene."] * len(all_paths),
        "annotation": ["A panoramic indoor scene."] * len(all_paths),
    })
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = max(int(len(df) * VAL_RATIO), 100)
    
    df_val = df.iloc[:val_size]
    df_train = df.iloc[val_size:]
    
    # Save
    train_path = os.path.join(OUTPUT_DIR, "stage1_train.csv")
    val_path = os.path.join(OUTPUT_DIR, "stage1_val.csv")
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Train: {len(df_train)} rows -> {train_path}")
    print(f"  Val:   {len(df_val)} rows -> {val_path}")
    
    # Also save SUN360 test as eval set
    sun360_test = sorted(glob.glob("/data/1_personal/4_SWWOO/SUN360/test/RGB/*.jpg"))
    if sun360_test:
        df_test = pd.DataFrame({
            "url": sun360_test,
            "query": ["Describe this panoramic scene."] * len(sun360_test),
            "annotation": ["A panoramic scene."] * len(sun360_test),
        })
        test_path = os.path.join(OUTPUT_DIR, "stage1_test.csv")
        df_test.to_csv(test_path, index=False)
        print(f"  Test:  {len(df_test)} rows -> {test_path}")


if __name__ == "__main__":
    main()
