import gc
import os
import random
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from typing import NamedTuple


class ModelResult(NamedTuple):
    folder: str
    accuracy: float
    optimizer: object
    initial_lr: float


# Params
data_source_path = '../stickFigures'
models_folder_path = './results'
n_timesteps = 120
n_features = 11
classes = 2
shuffle_seed = 100
train_size = 0.80
val_size = 0.2
batch_size = 32
epochs = 200


def load_data():
    features_list = []
    labels_list = []
    ids_list = []

    for file_name in os.listdir(data_source_path):
        if not file_name.lower().endswith('.csv'):
            continue
        csv_path = os.path.join(data_source_path, file_name)
        df = pd.read_csv(csv_path)

        if 'incorrect' in file_name.lower():
            label = 0
        elif 'correct' in file_name.lower():
            label = 1
        else:
            continue

        features = df.iloc[:, :n_features].values

        # Przygotowanie danych do odpowiedniego kształtu
        if features.shape[0] < n_timesteps:
            padding = np.zeros((n_timesteps - features.shape[0], n_features))
            features = np.vstack((features, padding))
        elif features.shape[0] > n_timesteps:
            features = features[:n_timesteps, :]

        features_list.append(features)
        labels_list.append(label)
        ids_list.append(file_name)

    return np.array(features_list), np.array(labels_list), np.array(ids_list)


def train_model(data, labels, testOptimizer):
    # Mieszanie danych
    c = list(zip(data, labels, current_ids))
    random.Random(shuffle_seed).shuffle(c)
    data, labels, ids = zip(*c)
    data = np.array(data)
    labels = np.array(labels)
    ids = np.array(ids)

    # Podział na zbiory treningowy i testowy PRZED normalizacją
    bound = int(len(labels) * train_size)
    x_train = data[:bound]
    y_train = labels[:bound]
    x_test = data[bound:]
    y_test = labels[bound:]
    test_ids = ids[bound:]

    print(f"Train data shape before normalization: {x_train.shape}")
    print(f"Test data shape before normalization: {x_test.shape}")

    # WŁAŚCIWA NORMALIZACJA
    # Oblicz parametry normalizacji TYLKO z danych treningowych
    train_data_flattened = x_train.reshape(-1, n_features)  # (samples*timesteps, features)

    # Opcja 1: Z-Score normalization (ZALECANE)
    train_mean = np.mean(train_data_flattened, axis=0)  # średnia dla każdej cechy
    train_std = np.std(train_data_flattened, axis=0)  # odchylenie dla każdej cechy

    # Zabezpieczenie przed dzieleniem przez zero
    train_std = np.where(train_std == 0, 1, train_std)

    print(f"Training data - Mean: {train_mean[:3]}")  # Pokaż pierwsze 3 cechy
    print(f"Training data - Std: {train_std[:3]}")

    # Normalizuj dane treningowe i testowe używając parametrów z treningu
    x_train_norm = normalize_data(x_train, train_mean, train_std)
    x_test_norm = normalize_data(x_test, train_mean, train_std)

    print(f"After normalization - Train mean: {x_train_norm.mean():.4f}, std: {x_train_norm.std():.4f}")
    print(f"After normalization - Test mean: {x_test_norm.mean():.4f}, std: {x_test_norm.std():.4f}")

    # Kodowanie etykiet do one-hot encoding
    y_train_encoded = pd.get_dummies(y_train).values
    y_test_encoded = pd.get_dummies(y_test).values

    # Model
    model = Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(classes, activation='softmax')
    ])

    model.compile(optimizer=testOptimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"Squat_CNN_LSTM_v2_{timestamp}"

    # Callbacki
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001
    )

    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(models_folder_path, model_name, 'mdl_wts.keras'),
        save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min'
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, start_from_epoch=20, verbose=1, mode='min'
    )

    # Trenowanie
    hist = model.fit(
        x=x_train_norm, y=y_train_encoded,
        batch_size=batch_size, epochs=epochs, verbose=1,
        callbacks=[reduce_lr, checkpoint, early_stopping],
        validation_split=val_size, shuffle=True
    )

    return x_train_norm, y_train_encoded, x_test_norm, y_test_encoded, test_ids, hist, model, model_name, train_mean, train_std


def normalize_data(data, mean, std):
    """Normalizuje dane używając podanych parametrów"""
    # data shape: (samples, timesteps, features)
    normalized_data = np.zeros_like(data)

    for i in range(data.shape[0]):  # dla każdej próbki
        for j in range(data.shape[1]):  # dla każdego timestep
            normalized_data[i, j] = (data[i, j] - mean) / std

    return normalized_data


def evaluate_model(x_test, y_test, hist, model, model_name, test_ids, train_mean, train_std):
    # Zapisz model
    os.makedirs(os.path.join(models_folder_path, model_name), exist_ok=True)
    model.save(os.path.join(models_folder_path, model_name, f"{model_name}.keras"))

    # ZAPISZ PARAMETRY NORMALIZACJI
    normalization_params = {
        "method": "z_score",
        "mean": train_mean.tolist(),
        "std": train_std.tolist(),
        "n_timesteps": n_timesteps,
        "n_features": n_features,
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(models_folder_path, model_name, 'normalization_params.json'), 'w') as f:
        json.dump(normalization_params, f, indent=2)

    print(f"Saved normalization parameters to normalization_params.json")

    # Zapisz historię treningu
    with open(os.path.join(models_folder_path, model_name, 'history_dict.pickle'), 'wb') as f:
        pickle.dump(hist.history, f, protocol=4)

    # Ewaluacja
    results = model.evaluate(x_test, y_test, verbose=1)
    print("Evaluation results:", results)

    # Predykcje i raport
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes))

    # Macierz pomyłek
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print(f"Confusion Matrix:\n{cm}")

    acc = results[1]
    if hasattr(acc, 'numpy'):
        acc = float(acc.numpy())

    return acc


# MAIN EXECUTION
learning_rate = 0.001
result_list = []
misclassified_counter = {}

# Uruchom tylko jeden trening dla testów
optimizer_instance = RMSprop(learning_rate=learning_rate)
current_data, current_labels, current_ids = load_data()

trainX, trainY, testX, testY, test_ids, Hist, Model, modelName, mean_params, std_params = train_model(
    data=current_data,
    labels=current_labels,
    testOptimizer=optimizer_instance
)

accu = evaluate_model(testX, testY, Hist, Model, modelName, test_ids, mean_params, std_params)

print(f"\nModel accuracy: {accu:.3f}")
print(f"Model saved to: {modelName}")
print(f"Normalization method: Z-Score with mean/std calculated from training data only")