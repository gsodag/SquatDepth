from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from .toStick import ToCSV
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Globalne zmienne
model = None
training_mean = None
training_std = None

# PARAMETRY Z TRENINGU
n_timesteps = 120
n_features = 11

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ścieżki do plików modelu
PATH_TO_MODEL = os.path.join(BASE_DIR, "results/Squat_CNN_LSTM_v2_20250813_012229/mdl_wts.keras")
PATH_TO_NORMALIZATION = os.path.join(BASE_DIR, "results/Squat_CNN_LSTM_v2_20250813_012229/normalization_params.json")


def load_model():
    """Ładowanie modelu i parametrów normalizacji"""
    global model, training_mean, training_std

    try:
        # Załaduj model
        if os.path.exists(PATH_TO_MODEL):
            model = tf.keras.models.load_model(PATH_TO_MODEL)
            logger.info(f"Model loaded successfully from {PATH_TO_MODEL}")
        else:
            logger.error(f"Model file not found: {PATH_TO_MODEL}")
            model = None
            return

        # Załaduj parametry normalizacji
        if os.path.exists(PATH_TO_NORMALIZATION):
            with open(PATH_TO_NORMALIZATION, 'r') as f:
                norm_params = json.load(f)
                training_mean = np.array(norm_params['mean'])
                training_std = np.array(norm_params['std'])
                logger.info(f"Loaded normalization parameters:")
                logger.info(f"  Method: {norm_params.get('method', 'unknown')}")
                logger.info(f"  Mean shape: {training_mean.shape}")
                logger.info(f"  Std shape: {training_std.shape}")
                logger.info(f"  Mean preview: {training_mean[:3]}")
                logger.info(f"  Std preview: {training_std[:3]}")
        else:
            logger.error(f"Normalization parameters not found: {PATH_TO_NORMALIZATION}")
            training_mean = None
            training_std = None

    except Exception as e:
        logger.error(f"Error loading model or normalization parameters: {str(e)}")
        model = None
        training_mean = None
        training_std = None


def normalize_data_like_training(data):
    """
    Normalizuje dane używając parametrów z treningu (Z-Score)
    """
    global training_mean, training_std

    if training_mean is None or training_std is None:
        raise ValueError("Training normalization parameters not loaded!")

    logger.info(f"Input data shape: {data.shape}")
    logger.info(f"Input data range: [{data.min():.2f}, {data.max():.2f}]")

    # Normalizacja: (x - mean) / std dla każdej cechy
    normalized_data = np.zeros_like(data)

    for i in range(data.shape[0]):  # dla każdego timestep
        normalized_data[i] = (data[i] - training_mean) / training_std

    logger.info(f"After normalization - data range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
    logger.info(f"After normalization - mean: {normalized_data.mean():.4f}, std: {normalized_data.std():.4f}")

    return normalized_data


def preprocess_data_like_training(X):
    """
    Przygotowuje dane dokładnie tak samo jak w treningu
    """
    logger.info(f"Raw input data shape: {X.shape}")

    # 1. Usuń wiersze z NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        logger.warning(f"Removing {nan_mask.sum()} rows with NaN values")
        X = X[~nan_mask]

    if X.size == 0:
        raise ValueError("No valid data after removing NaN values")

    # 2. Przytnij/dopełnij do n_timesteps (120 ramek)
    if X.shape[0] < n_timesteps:
        # Dopełnij zerami
        padding = np.zeros((n_timesteps - X.shape[0], n_features))
        X = np.vstack((X, padding))
        logger.info(f"Padded data to {n_timesteps} timesteps")
    elif X.shape[0] > n_timesteps:
        # Przytnij
        X = X[:n_timesteps, :]
        logger.info(f"Truncated data to {n_timesteps} timesteps")

    logger.info(f"After timestep adjustment: {X.shape}")

    # 3. Normalizacja Z-Score używając parametrów z treningu
    X_normalized = normalize_data_like_training(X)

    # 4. Reshape do formatu batch (1, timesteps, features)
    X_batch = X_normalized.reshape(1, X_normalized.shape[0], X_normalized.shape[1])
    logger.info(f"Final shape for model: {X_batch.shape}")

    return X_batch


# Załaduj model przy starcie
load_model()


@app.get("/health")
async def health_check():
    """Endpoint sprawdzający stan aplikacji"""
    return JSONResponse({
        "status": "running",
        "model_loaded": model is not None,
        "normalization_params_loaded": training_mean is not None and training_std is not None,
        "training_mean_preview": training_mean[:3].tolist() if training_mean is not None else None,
        "training_std_preview": training_std[:3].tolist() if training_std is not None else None
    })


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    temp_video_path = None
    temp_input_dir = None

    try:
        logger.info(f"Received file: {file.filename}")

        # Sprawdź czy model i parametry są załadowane
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if training_mean is None or training_std is None:
            raise HTTPException(status_code=500, detail="Normalization parameters not loaded")

        # Sprawdź typ pliku
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
            )

        # Utwórz tymczasowy katalog
        temp_input_dir = tempfile.mkdtemp()
        logger.info(f"Created temp directory: {temp_input_dir}")

        # Zapisz przesłany plik
        temp_video_path = os.path.join(temp_input_dir, file.filename)
        with open(temp_video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Saved video to: {temp_video_path} ({len(content)} bytes)")

        # Preprocessing wideo
        logger.info("Starting video preprocessing...")
        outputDirectory = ToCSV(temp_input_dir)

        # Znajdź CSV
        csv_files = [f for f in os.listdir(outputDirectory) if f.endswith('_height_analysis.csv')]
        if not csv_files:
            raise HTTPException(status_code=500, detail="No CSV file generated from preprocessing")

        csv_path = os.path.join(outputDirectory, csv_files[-1])
        logger.info(f"Using CSV file: {csv_path}")

        # Wczytaj i przygotuj dane
        logger.info("Reading CSV and preparing data...")
        df = pd.read_csv(csv_path)

        logger.info(f"CSV shape: {df.shape}")
        logger.info(f"CSV columns: {list(df.columns)}")

        # Użyj pierwszych 11 kolumn (bez czasu)
        X = df.iloc[:, :n_features].values

        logger.info(f"Extracted features shape: {X.shape}")

        if X.size == 0:
            raise HTTPException(status_code=500, detail="No data extracted from video")

        # Przygotuj dane jak w treningu
        X_processed = preprocess_data_like_training(X)

        # Sprawdź kształt
        expected_shape = (1, n_timesteps, n_features)
        if X_processed.shape != expected_shape:
            raise HTTPException(
                status_code=500,
                detail=f"Data shape mismatch. Expected {expected_shape}, got {X_processed.shape}"
            )

        # Predykcja
        logger.info("Making prediction...")
        logger.info(
            f"Input to model: shape={X_processed.shape}, range=[{X_processed.min():.4f}, {X_processed.max():.4f}]")

        prediction = model.predict(X_processed, verbose=0)
        logger.info(f"Raw prediction: {prediction}")

        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        class_probabilities = prediction[0]
        prob_incorrect = float(class_probabilities[0])  # klasa 0
        prob_correct = float(class_probabilities[1])  # klasa 1

        result = "PASS" if predicted_class == 1 else "FAIL"

        logger.info(f"FINAL RESULT: {result}")
        logger.info(f"Class probabilities - Incorrect: {prob_incorrect:.4f}, Correct: {prob_correct:.4f}")
        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

        # Sprzątanie
        try:
            if os.path.exists(csv_path):
                os.remove(csv_path)
        except Exception as e:
            logger.warning(f"Could not remove CSV file: {e}")

        return JSONResponse({
            "prediction": result,
            "confidence": confidence,
            "predicted_class": int(predicted_class),
            "probabilities": {
                "incorrect": prob_incorrect,
                "correct": prob_correct
            },
            "debug_info": {
                "data_shape": list(X_processed.shape),
                "raw_data_range": [float(X.min()), float(X.max())],
                "normalized_data_range": [float(X_processed.min()), float(X_processed.max())],
                "csv_rows": len(df),
                "normalization_method": "z_score"
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # Sprzątanie plików tymczasowych
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.info("Cleaned up temp video file")
            except Exception as e:
                logger.warning(f"Could not remove temp video file: {e}")

        if temp_input_dir and os.path.exists(temp_input_dir):
            try:
                for file in os.listdir(temp_input_dir):
                    file_path = os.path.join(temp_input_dir, file)
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                os.rmdir(temp_input_dir)
                logger.info("Cleaned up temp directory")
            except Exception as e:
                logger.warning(f"Could not remove temp directory: {e}")


@app.get("/")
async def root():
    return {"message": "Squats Analysis API with proper Z-Score normalization"}


if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, port=8001, log_level="info")