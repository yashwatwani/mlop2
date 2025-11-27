import json
import os
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_PATH = "data/raw_data.jsonl"
OUTPUT_DIR = "data/mlx_format"
# Split ratios: 80% train, 10% validation, 10% test
TRAIN_RATIO = 0.8
VALID_TEST_RATIO = 0.2  # 10% of total
VALID_RATIO_OF_REMAINING = 0.5 # 10% of total is 50% of 20% remaining

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Loading data from: {RAW_DATA_PATH}")

# 1. Load the entire dataset using the Hugging Face 'datasets' library
try:
    # Load data as a Dataset object (assuming JSONL format)
    full_dataset = load_dataset("json", data_files=RAW_DATA_PATH, split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure raw_data.jsonl is valid JSONL format.")
    exit()

# Convert to Pandas DataFrame for easier splitting (Optional, but robust)
df = full_dataset.to_pandas()
print(f"Total samples loaded: {len(df)}")

# 2. First Split: Separate the Test Set (10%)
# We use sklearn's train_test_split function twice for a three-way split.
# test_size=0.1 means 10% for the final test set.
train_val_df, test_df = train_test_split(
    df, 
    test_size=0.1, 
    random_state=42,
    shuffle=True
)
print(f"Train/Validation samples: {len(train_val_df)}, Test samples: {len(test_df)}")

# 3. Second Split: Separate Train (80%) and Validation (10%)
# We split the remaining 90% (train_val_df) into a 90% train set (8/9 ≈ 88.8%) 
# and a 10% validation set (1/9 ≈ 11.1%). 
# test_size=0.1111 is approx 10% of the original 90%
train_df, validation_df = train_test_split(
    train_val_df, 
    test_size=len(test_df), # Use the same number of samples as the test set for symmetry
    random_state=42,
    shuffle=True
)

# Convert back to Hugging Face Dataset objects
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
valid_ds = Dataset.from_pandas(validation_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

# 4. Save the three splits as JSONL files
train_ds.to_json(os.path.join(OUTPUT_DIR, "train.jsonl"), orient="records", lines=True)
valid_ds.to_json(os.path.join(OUTPUT_DIR, "valid.jsonl"), orient="records", lines=True)
test_ds.to_json(os.path.join(OUTPUT_DIR, "test.jsonl"), orient="records", lines=True)

print("\n--- Split Summary ---")
print(f"✅ train.jsonl saved: {len(train_df)} samples")
print(f"✅ valid.jsonl saved: {len(validation_df)} samples")
print(f"✅ test.jsonl saved: {len(test_df)} samples")
print(f"Files saved to: {OUTPUT_DIR}/")