import zipfile
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Configuration
ZIP_PATH = 'archive (3).zip'
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
TARGET_FILES = ['ADMISSIONS.csv', 'PATIENTS.csv', 'LABEVENTS.csv']

def extract_data():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_files = z.namelist()
        for target in TARGET_FILES:
            # Find file (handling potential subfolders in zip)
            match = next((f for f in all_files if f.endswith(target)), None)
            if match and not os.path.exists(os.path.join(RAW_DIR, target)):
                print(f"Extracting {match}...")
                z.extract(match, RAW_DIR)
                # Rename to flat filename if extracted simply
                extracted_path = os.path.join(RAW_DIR, match)
                target_path = os.path.join(RAW_DIR, target)
                if extracted_path != target_path:
                    # Move/Rename if needed (e.g. if zip structure is complex)
                    # For now assuming simple extract or flattening
                    pass

def load_and_process():
    print("Loading extracted CSVs...")
    # Helper to find file path regardless of extraction folder structure
    def get_path(fname):
        for root, dirs, files in os.walk(RAW_DIR):
            if fname in files:
                return os.path.join(root, fname)
        return None

    # Load Admissions
    adm_path = get_path('ADMISSIONS.csv')
    df_adm = pd.read_csv(adm_path)
    
    # Load Patients
    pat_path = get_path('PATIENTS.csv')
    df_pat = pd.read_csv(pat_path)
    
    # Load Lab Events (Chunk/Sample if too large, but for 11MB zip it fits in memory)
    lab_path = get_path('LABEVENTS.csv')
    df_lab = pd.read_csv(lab_path)

    print("Merging Admissions and Patients...")
    # Merge
    df = pd.merge(df_adm, df_pat, on='subject_id', how='inner')
    
    # Date conversion
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    # Calculate Age
    # Use Year difference to avoid OverflowError with standard pandas Timestamps (ns precision)
    df['age'] = df['admittime'].dt.year - df['dob'].dt.year
    df.loc[df['age'] < 0, 'age'] = 90
    df.loc[df['age'] > 90, 'age'] = 90
    
    # Target Variable: Mortality
    # HOSPITAL_EXPIRE_FLAG exists in ADMISSIONS usually.
    if 'hospital_expire_flag' not in df.columns:
        print("Creating target from deathtime...")
        df['hospital_expire_flag'] = df['deathtime'].notnull().astype(int)
    
    print("Processing Lab Events...")
    # Feature Engineering from Labs
    # 1. Count of labs per admission
    lab_counts = df_lab.groupby('hadm_id').size().reset_index(name='lab_count')
    
    # 2. Count of abnormal labs
    # 'flag' column usually contains 'abnormal'
    if 'flag' in df_lab.columns:
        abnormal_labs = df_lab[df_lab['flag'].astype(str).str.lower() == 'abnormal']
        abnormal_counts = abnormal_labs.groupby('hadm_id').size().reset_index(name='abnormal_count')
    else:
        abnormal_counts = pd.DataFrame({'hadm_id': [], 'abnormal_count': []})

    # Merge Lab features
    df = pd.merge(df, lab_counts, on='hadm_id', how='left')
    df = pd.merge(df, abnormal_counts, on='hadm_id', how='left')
    
    # Fill missing lab counts with 0 (no labs)
    df['lab_count'] = df['lab_count'].fillna(0)
    df['abnormal_count'] = df['abnormal_count'].fillna(0)

    # Clean other features
    df['gender'] = df['gender'].map({'M': 0, 'F': 1}).fillna(0)
    
    # Select Features
    features = ['age', 'gender', 'lab_count', 'abnormal_count'] # Add more if available (e.g. Admission Type)
    
    # Add One-Hot for Admission Type
    if 'admission_type' in df.columns:
        dummies = pd.get_dummies(df['admission_type'], prefix='type')
        df = pd.concat([df, dummies], axis=1)
        features.extend(dummies.columns)

    target = 'hospital_expire_flag'
    
    # Final Dataset
    final_df = df[features + [target]].dropna()
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Class balance (Target=1): {final_df[target].mean():.2%}")
    
    return final_df

def save_data(df):
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    train.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)
    print(f"Saved processed data to {PROCESSED_DIR}")

if __name__ == "__main__":
    extract_data()
    df = load_and_process()
    save_data(df)
