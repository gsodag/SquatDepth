from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import cv2
from contextlib import asynccontextmanager
from .toStick import ToCSV as ToCSV_Comfort
from .toStickAcc import ToCSV as ToCSV_Accuracy
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_CONFIG = {
    "comfort": {
        "path_model": os.path.join(BASE_DIR, "results/comfort/mdl_wts.keras"),
        "path_norm": os.path.join(BASE_DIR, "results/comfort/normalization_params.json"),
        "n_features": 11,
        "processor": ToCSV_Comfort,
        "preprocessing_method": "pad_truncate",
        "loaded_model": None,
        "norm_mean": None,
        "norm_std": None
    },
    "accuracy": {
        "path_model": os.path.join(BASE_DIR, "results/accuracy/mdl_wts.keras"),
        "path_norm": os.path.join(BASE_DIR, "results/accuracy/normalization_params.json"),
        "n_features": 10,
        "processor": ToCSV_Accuracy,
        "preprocessing_method": "resample",
        "loaded_model": None,
        "norm_mean": None,
        "norm_std": None
    }
}

n_timesteps = 120

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- STARTUP: Rozpoczynam ładowanie modeli ---")

    for model_name, config in MODELS_CONFIG.items():
        try:
            if not os.path.exists(config["path_model"]):
                logger.error(f"[{model_name}] PLIK MODELU NIE ISTNIEJE w: {config['path_model']}")
                continue

            logger.info(f"[{model_name}] Ładowanie modelu z dysku...")
            config["loaded_model"] = tf.keras.models.load_model(config["path_model"])

            if os.path.exists(config["path_norm"]):
                with open(config["path_norm"], 'r') as f:
                    norm_params = json.load(f)
                    config["norm_mean"] = np.array(norm_params['mean'])
                    config["norm_std"] = np.array(norm_params['std'])
            else:
                logger.warning(f"[{model_name}] Brak pliku normalizacji!")

            logger.info(f"[{model_name}] GOTOWY DO PRACY.")

        except Exception as e:
            logger.error(f"[{model_name}] KRYTYCZNY BŁĄD PODCZAS ŁADOWANIA: {str(e)}")
            logger.error(traceback.format_exc())

    yield

    logger.info("--- SHUTDOWN: Czyszczenie pamięci TensorFlow ---")
    tf.keras.backend.clear_session()


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSED_VIDEOS_DIR = os.path.join(BASE_DIR, "processed_videos")
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=PROCESSED_VIDEOS_DIR), name="videos")

def preprocess_data(X, config):
    mean = config["norm_mean"]
    std = config["norm_std"]
    expected_features = config["n_features"]
    method = config["preprocessing_method"]

    X = X[~np.isnan(X).any(axis=1)]
    if X.size == 0: raise ValueError("Dane są puste (same wartości NaN)")

    if X.shape[1] != expected_features:
        X = X[:, :expected_features]

    if method == "resample":
        if X.shape[0] != n_timesteps:
            X = X.astype(np.float32)
            X = cv2.resize(X, (X.shape[1], n_timesteps), interpolation=cv2.INTER_LINEAR)
    elif method == "pad_truncate":
        if X.shape[0] < n_timesteps:
            padding = np.zeros((n_timesteps - X.shape[0], expected_features))
            X = np.vstack((X, padding))
        elif X.shape[0] > n_timesteps:
            X = X[:n_timesteps, :]

    X_normalized = (X - mean) / std
    return X_normalized.reshape(1, n_timesteps, expected_features)
@app.get("/debug")
async def debug_status():
    status = {}
    for name, config in MODELS_CONFIG.items():
        status[name] = {
            "path": config["path_model"],
            "exists_on_disk": os.path.exists(config["path_model"]),
            "is_loaded_in_memory": config["loaded_model"] is not None
        }
    return status

@app.post("/upload")
async def upload_video(file: UploadFile = File(...), model: str = Form("comfort")):
    temp_dir = None
    try:
        logger.info(f"Otrzymano wideo: {file.filename}, Model: {model}")

        if model not in MODELS_CONFIG:
            raise HTTPException(400, f"Nieznany model: {model}")

        cfg = MODELS_CONFIG[model]

        if cfg["loaded_model"] is None:
            logger.error(f"Błąd: Model {model} jest None. Ścieżka: {cfg['path_model']}")
            raise HTTPException(500,
                                f"Model '{model}' nie załadował się poprawnie przy starcie serwera. Sprawdź logi konsoli.")

        temp_dir = tempfile.mkdtemp()
        v_path = os.path.join(temp_dir, file.filename)
        with open(v_path, "wb") as f:
            f.write(await file.read())

        out_dir = cfg["processor"](temp_dir, PROCESSED_VIDEOS_DIR)

        base = os.path.splitext(file.filename)[0]
        csv_suf = '_height_analysis.csv' if model == 'comfort' else '_acc_analysis.csv'
        csv_file = next((f for f in os.listdir(out_dir) if base in f and f.endswith(csv_suf)), None)
        vid_file = next((f for f in os.listdir(out_dir) if base in f and f.endswith('.mp4') and 'stick' in f), None)

        if not csv_file: raise HTTPException(500, "Nie udało się wygenerować analizy (brak pliku CSV).")

        df = pd.read_csv(os.path.join(out_dir, csv_file))
        X = df.iloc[:, :cfg["n_features"]].values

        X_proc = preprocess_data(X, cfg)
        pred = cfg["loaded_model"].predict(X_proc, verbose=0)

        p_idx = np.argmax(pred, axis=1)[0]
        result = "PASS" if p_idx == 1 else "FAIL"

        try:
            os.remove(os.path.join(out_dir, csv_file))
        except:
            pass

        return JSONResponse({
            "prediction": result,
            "confidence": float(np.max(pred)),
            "probabilities": {"incorrect": float(pred[0][0]), "correct": float(pred[0][1])},
            "video_url": f"http://localhost:8001/videos/{vid_file}" if vid_file else None,
            "model_used": model
        })

    except Exception as e:
        logger.error(f"Błąd endpointu: {traceback.format_exc()}")
        raise HTTPException(500, str(e))
    finally:
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
