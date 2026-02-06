import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
DATASET_PATH = "dataset"  # Aapka folder name
MODEL_FILE = "voice_model.pkl"

def extract_features(file_path):
    # Audio se features nikalna
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None

def train_model():
    X = []
    y = []
    
    print("Files load ho rahi hain...")

    # 1. HUMAN Data Load (Label 0)
    path_human = os.path.join(DATASET_PATH, "human")
    if os.path.exists(path_human):
        for f in os.listdir(path_human):
            if f.endswith('.mp3') or f.endswith('.wav'):
                data = extract_features(os.path.join(path_human, f))
                if data is not None:
                    X.append(data)
                    y.append(0) # 0 = Human
    else:
        print("Warning: 'human' folder nahi mila!")

    # 2. AI Data Load (Label 1)
    path_ai = os.path.join(DATASET_PATH, "ai")
    if os.path.exists(path_ai):
        for f in os.listdir(path_ai):
            if f.endswith('.mp3') or f.endswith('.wav'):
                data = extract_features(os.path.join(path_ai, f))
                if data is not None:
                    X.append(data)
                    y.append(1) # 1 = AI
    else:
        print("Warning: 'ai' folder nahi mila!")

    # 3. Training
    if len(X) > 0:
        print(f"Training started on {len(X)} files...")
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)
        joblib.dump(clf, MODEL_FILE)
        print(f"SUCCESS: Model save ho gaya -> {MODEL_FILE}")
    else:
        print("ERROR: Koi audio files nahi mili training ke liye!")

if __name__ == "__main__":
    train_model()