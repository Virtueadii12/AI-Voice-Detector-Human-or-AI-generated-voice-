# 🎙️ AI-Generated Voice Detection System

An end-to-end Machine Learning pipeline designed to distinguish between **Human** and **AI-Generated** voices. This system analyzes audio frequency patterns using MFCC (Mel-frequency cepstral coefficients) and classifies them using a Random Forest model, exposed via a high-performance FastAPI server.

## 🚀 Features
- **Audio Feature Extraction:** Uses `Librosa` to extract MFCC features from audio files.
- **Machine Learning Model:** Trained using `Random Forest Classifier` for robust detection.
- **Real-time API:** Fast and asynchronous API built with `FastAPI`.
- **Base64 Support:** Accepts Base64 encoded audio strings for direct integration.
- **Security:** Implements API Key authentication (`x-api-key`) for secure access.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **ML & Data:** Scikit-Learn, NumPy, Joblib
- **Audio Processing:** Librosa
- **Backend Framework:** FastAPI, Uvicorn

## 📂 Project Structure
```bash
AI-Voice-Detector/
│
├── dataset/             # (Not uploaded) Folder containing 'human' and 'ai' audio files
├── train.py             # Script to extract features and train the model
├── main.py              # FastAPI server code for inference
├── voice_model.pkl      # Trained ML model (saved after training)
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
