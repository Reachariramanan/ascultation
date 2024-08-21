from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from tensorflow.keras.models import load_model
import threading

app = FastAPI()
lock = threading.Lock()

# Models loading functions similar to Flask example
def load_heart_model(model_path):
    class CustomMobileNetV2(nn.Module):
        def __init__(self):
            super(CustomMobileNetV2, self).__init__()
            self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
            self.mobilenet = mobilenet_v3_small(pretrained=True)
            self.classifier = nn.Linear(1000, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.mobilenet(x)
            x = self.classifier(x)
            return x
    
    model = CustomMobileNetV2()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_melspectrogram_db(file_path, sr=None, n_fft=1024, hop_length=256, n_mels=256, fmin=10, fmax=2000, top_db=80):
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0] < 18 * sr:
        wav = np.pad(wav, int(np.ceil((18 * sr - wav.shape[0]) / 2)), mode='reflect')
    else:
        wav = wav[:18 * sr]
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def load_lung_model(model_path):
    return load_model(model_path)

def extract_mfccs(file_path, sample_rate=16000, n_mfcc=13):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc).T
    max_length = 100
    if mfccs.shape[0] < max_length:
        mfccs = np.pad(mfccs, ((0, max_length - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:max_length, :]
    return mfccs

# Load models at the start
heart_model = load_heart_model("models/heart_model.pth")
lung_model = load_lung_model("models/lung_model.h5")

# Define a list of disease names for lung predictions
disease_names = [
    "Asthma", "Bronchitis", "COPD", "Heart Failure", 
    "Lung Fibrosis", "Normal", "Pleural Effusion", "Pneumonia"
]

@app.post("/heart")
async def predict_heart(file: UploadFile = File(...)):
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    try:
        spec = get_melspectrogram_db(file_location)
        spec = spec_to_image(spec)
        spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            output = heart_model(spec)
        
        probabilities = F.softmax(output, dim=1)
        _, predicted_label = torch.max(output, 1)
        output_label = 'Murmur Present' if predicted_label.item() == 1 else 'Murmur Absent'

        return {
            "predicted_label": predicted_label.item(),
            "probabilities": probabilities.squeeze().tolist(),
            "output_label": output_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lungs")
async def predict_lung_disease(file: UploadFile = File(...)):
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        mfccs = extract_mfccs(file_location)
        prediction = lung_model.predict(np.array([mfccs]))
        predicted_class = np.argmax(prediction)
        predicted_disease = disease_names[predicted_class]

        return {
            "predicted_class": int(predicted_class),
            "predicted_disease": predicted_disease
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# The rest of your FastAPI code stays the same, including the home route and API key validation.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
