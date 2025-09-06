import os
import pandas as pd
from tqdm import tqdm
import kagglehub
from sklearn.model_selection import GroupShuffleSplit
from config import DATA_PATH, EMOTIONS_TO_KEEP, TEST_SIZE_VAL_TEST, TEST_SIZE_FINAL
from utils import parse_ravdess_filename

def main():
    """Downloads, parses, and splits the RAVDESS dataset."""
    # Create necessary directories
    os.makedirs(DATA_PATH, exist_ok=True)
    print("Directories created.")

    # 1. Download and extract dataset
    print("Downloading RAVDESS dataset...")
    path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    print(f"Dataset downloaded to: {path}")

    # 2. Parse filenames and create metadata DataFrame
    all_metadata = []
    actor_folders = [f for f in os.listdir(path) if f.startswith("Actor_")]
    
    for actor_folder in tqdm(actor_folders, desc="Parsing Actor Folders"):
        actor_path = os.path.join(path, actor_folder)
        wav_files = [f for f in os.listdir(actor_path) if f.lower().endswith(".wav")]
        for wav_file in wav_files:
            metadata = parse_ravdess_filename(wav_file, EMOTIONS_TO_KEEP)
            if metadata:
                metadata["full_path"] = os.path.join(actor_path, wav_file)
                all_metadata.append(metadata)

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df_path = os.path.join(DATA_PATH, 'metadata.pkl')
    metadata_df.to_pickle(metadata_df_path)
    print(f"\nParsed {len(metadata_df)} valid audio files.")
    print(f"Metadata DataFrame saved to {metadata_df_path}")
    print("\nEmotion distribution:\n", metadata_df['emotion_label'].value_counts())

    # 3. Speaker-independent data split
    X = metadata_df.drop(columns=['emotion_label'])
    y = metadata_df['emotion_label']
    groups = metadata_df['speaker_id']

    # Split 1: train vs. (validation + test)
    gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE_VAL_TEST, random_state=42)
    train_idx, temp_idx = next(gss_train_temp.split(X, y, groups))
    train_df = metadata_df.iloc[train_idx]
    temp_df = metadata_df.iloc[temp_idx]

    # Split 2: validation vs. test
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE_FINAL, random_state=42)
    val_idx, test_idx = next(gss_val_test.split(
        temp_df.drop(columns=['emotion_label']), temp_df['emotion_label'], temp_df['speaker_id']
    ))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    # Save splits
    train_df.to_pickle(os.path.join(DATA_PATH, 'train_df.pkl'))
    val_df.to_pickle(os.path.join(DATA_PATH, 'val_df.pkl'))
    test_df.to_pickle(os.path.join(DATA_PATH, 'test_df.pkl'))

    print("\nData splitting complete:")
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print("Speakers in train:", sorted(train_df['speaker_id'].unique()))
    print("Speakers in val:", sorted(val_df['speaker_id'].unique()))
    print("Speakers in test:", sorted(test_df['speaker_id'].unique()))

if __name__ == '__main__':
    main()