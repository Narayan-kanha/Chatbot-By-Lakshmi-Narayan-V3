# tokenize_data_memory_efficient.py
# This script prepares a large dataset for training without loading the entire file into memory.

import os
import pickle
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Make sure this points to your final, largest dataset file.
INPUT_FILE_PATH = "ULTIMATE_DATASET.txt" 
# The size of chunks to read from the file (in characters)
CHUNK_SIZE = 8192  # 8KB
train_size = 0.7  # 70% for training, 30% for validation
total_train_size = train_size * 100

# --- End Configuration ---


def process_dataset_memory_efficient():
    """
    Reads a large text file in chunks to build a vocabulary and save tokenized
    train/validation splits to disk, minimizing memory usage.
    """
    print(f"--- Starting memory-efficient data processing for '{INPUT_FILE_PATH}' ---")

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"\n!!! FATAL ERROR !!!")
        print(f"Dataset file not found at '{INPUT_FILE_PATH}'.")
        print("Please create this file by combining all your text sources,")
        print("and make sure the filename is correct in this script.")
        return

    # --- First Pass: Build Vocabulary and Get Total Character Count ---
    print("\n--- Pass 1: Building vocabulary and counting characters ---")
    
    unique_chars = set()
    total_chars = 0
    
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        # Using tqdm to show progress while reading the file
        pbar = tqdm(total=os.path.getsize(INPUT_FILE_PATH), unit='B', unit_scale=True, desc="Reading file")
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            unique_chars.update(chunk)
            total_chars += len(chunk)
            pbar.update(len(chunk.encode('utf-8', errors='ignore')))
        pbar.close()

    chars = sorted(list(unique_chars))
    vocab_size = len(chars)
    print(f"Successfully read {total_chars:,} characters.")
    print(f"Vocabulary size: {vocab_size} unique characters.")

    # Create character-to-integer mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Save the metadata (vocabulary)
    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print("Successfully saved the master dictionary to 'meta.pkl'.")

    # --- Second Pass: Tokenize and Write to Binary Files ---
    print("\n--- Pass 2: Tokenizing and creating binary split files ---")

    # Define the split point
    train_split_size = int(train_size * total_chars)
    val_split_size = total_chars - train_split_size
    print(f"Splitting data into {total_train_size:,}% train ({train_split_size:,} tokens) and {100 - total_train_size:,.0f}% validation ({val_split_size:,} tokens).")
    
    # Open binary files for writing
    train_file = open('train.bin', 'wb')
    val_file = open('val.bin', 'wb')
    
    processed_chars = 0
    current_file = train_file

    with open(INPUT_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        pbar = tqdm(total=total_chars, unit='char', desc="Tokenizing")
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            for char in chunk:
                if processed_chars == train_split_size:
                    # Switch to writing to the validation file
                    print("\nReached split point. Now writing to 'val.bin'.")
                    current_file = val_file

                # Encode the character and write it as a 16-bit unsigned integer
                token = stoi.get(char, 0) # Default to 0 if char somehow not in vocab
                current_file.write(np.array(token, dtype=np.uint16).tobytes())
                processed_chars += 1
            
            pbar.update(len(chunk))
        pbar.close()

    # Close the binary files
    train_file.close()
    val_file.close()

    print("\nSuccessfully saved 'train.bin' and 'val.bin'.")
    print("\n--- Pre-processing Complete! ✅ ---")
    print("You are now ready to run the training script.")


if __name__ == '__main__':
    process_dataset_memory_efficient()