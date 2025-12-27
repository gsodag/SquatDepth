import gc
import os
import random
import pickle
import numpy as np
import pandas as pd
import json
import cv2
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


data_source_path = r'C:\Users\piotr\SquatDepth\backend\squatAI\acc_training_slim'
models_folder_path = './results'
n_timesteps = 120
n_features = 10
classes = 2
shuffle_seed = 100
train_size = 0.80
val_size = 0.2
batch_size = 32
epochs = 200


def resample_to_fixed_timesteps(data, target_timesteps):

    if data.shape[0] == target_timesteps:
        return data

    data = data.astype(np.float32)

    resampled_data = cv2.resize(data, (data.shape[1], target_timesteps), interpolation=cv2.INTER_LINEAR)

    return resampled_data


def load_data():
    features_list = []
    labels_list = []
    ids_list = []

    if not os.path.exists(data_source_path):
        print(f"BŁĄD: Ścieżka {data_source_path} nie istnieje!")
        return np.array([]), np.array([]), np.array([])

    print(f"Ładowanie danych z: {data_source_path}")
    files = [f for f in os.listdir(data_source_path) if f.lower().endswith('.csv')]

    for file_name in files:
        csv_path = os.path.join(data_source_path, file_name)

        try:
            df = pd.read_csv(csv_path)

            if 'incorrect' in file_name.lower():
                label = 0
            elif 'correct' in file_name.lower():
                label = 1
            else:
                continue

            features = df.iloc[:, :n_features].values

            if features.shape[0] > 10:
                features = resample_to_fixed_timesteps(features, n_timesteps)

                features_list.append(features)
                labels_list.append(label)
                ids_list.append(file_name)

        except Exception as e:
            print(f"Błąd przy przetwarzaniu pliku {file_name}: {e}")

    print(f"Załadowano {len(features_list)} próbek.")
    return np.array(features_list), np.array(labels_list), np.array(ids_list)


def normalize_data(data, mean, std):
    normalized_data = np.zeros_like(data)

    for i in range(data.shape[0]):
        normalized_data[i] = (data[i] - mean) / std

    return normalized_data


def train_model(data, labels, ids, testOptimizer):
    if len(data) == 0:
        print("Brak danych do treningu.")
        return None, None, None, None, None, None, None, None, None, None

    c = list(zip(data, labels, ids))
    random.Random(shuffle_seed).shuffle(c)
    data, labels, ids = zip(*c)
    data = np.array(data)
    labels = np.array(labels)
    ids = np.array(ids)

    bound = int(len(labels) * train_size)
    x_train = data[:bound]
    y_train = labels[:bound]
    x_test = data[bound:]
    y_test = labels[bound:]
    test_ids = ids[bound:]

    print(f"Train data shape before normalization: {x_train.shape}")
    print(f"Test data shape before normalization: {x_test.shape}")

    train_data_flattened = x_train.reshape(-1, n_features)

    train_mean = np.mean(train_data_flattened, axis=0)
    train_std = np.std(train_data_flattened, axis=0)

    train_std = np.where(train_std == 0, 1, train_std)

    print(f"Training data - Mean (first 3): {train_mean[:3]}")
    print(f"Training data - Std (first 3): {train_std[:3]}")

    x_train_norm = normalize_data(x_train, train_mean, train_std)
    x_test_norm = normalize_data(x_test, train_mean, train_std)

    print(f"After normalization - Train mean: {x_train_norm.mean():.4f}, std: {x_train_norm.std():.4f}")
    print(f"After normalization - Test mean: {x_test_norm.mean():.4f}, std: {x_test_norm.std():.4f}")

    y_train_encoded = pd.get_dummies(y_train).values
    y_test_encoded = pd.get_dummies(y_test).values

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
    model_name = f"Squat_CNN_LSTM_ACCURACY_{timestamp}"

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001
    )

    os.makedirs(os.path.join(models_folder_path, model_name), exist_ok=True)

    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(models_folder_path, model_name, 'mdl_wts.keras'),
        save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min'
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=15, start_from_epoch=10, verbose=1, mode='min'
    )

    print("Rozpoczynam trening...")
    hist = model.fit(
        x=x_train_norm, y=y_train_encoded,
        batch_size=batch_size, epochs=epochs, verbose=1,
        callbacks=[reduce_lr, checkpoint, early_stopping],
        validation_split=val_size, shuffle=True
    )

    return x_train_norm, y_train_encoded, x_test_norm, y_test_encoded, test_ids, hist, model, model_name, train_mean, train_std


def evaluate_model(x_test, y_test, hist, model, model_name, test_ids, train_mean, train_std):
    model.save(os.path.join(models_folder_path, model_name, f"{model_name}.keras"))

    normalization_params = {
        "method": "z_score_resampled",
        "mean": train_mean.tolist(),
        "std": train_std.tolist(),
        "n_timesteps": n_timesteps,
        "n_features": n_features,
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(models_folder_path, model_name, 'normalization_params.json'), 'w') as f:
        json.dump(normalization_params, f, indent=2)

    with open(os.path.join(models_folder_path, model_name, 'history_dict.pickle'), 'wb') as f:
        pickle.dump(hist.history, f, protocol=4)

    results = model.evaluate(x_test, y_test, verbose=1)
    print("Evaluation results:", results)

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes))

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print(f"Confusion Matrix:\n{cm}")

    print("\n" + "=" * 40)
    print("      MISCLASSIFIED VIDEOS DETAILED")
    print("=" * 40)

    false_positives = []
    false_negatives = []

    for i in range(len(test_ids)):
        actual = y_test_classes[i]
        predicted = y_pred_classes[i]
        filename = test_ids[i]

        if actual != predicted:
            if actual == 0 and predicted == 1:
                false_positives.append(filename)
            elif actual == 1 and predicted == 0:
                false_negatives.append(filename)

    print(f"\nFALSE POSITIVES (Dangerous): {len(false_positives)}")
    print("   (Actual: FAIL, AI said: PASS)")
    for f in false_positives:
        print(f"   - {f}")

    print(f"\nFALSE NEGATIVES (Frustrating): {len(false_negatives)}")
    print("   (Actual: PASS, AI said: FAIL)")
    for f in false_negatives:
        print(f"   - {f}")

    print("=" * 40 + "\n")

    acc = results[1]
    if hasattr(acc, 'numpy'):
        acc = float(acc.numpy())

    return acc

if __name__ == "__main__":
    learning_rate = 0.001

    optimizer_instance = RMSprop(learning_rate=learning_rate)

    current_data, current_labels, current_ids = load_data()

    if len(current_data) > 0:
        trainX, trainY, testX, testY, test_ids, Hist, Model, modelName, mean_params, std_params = train_model(
            data=current_data,
            labels=current_labels,
            ids=current_ids,
            testOptimizer=optimizer_instance
        )

        if Model is not None:
            accu = evaluate_model(testX, testY, Hist, Model, modelName, test_ids, mean_params, std_params)

            print(f"\nModel accuracy: {accu:.3f}")
            print(f"Model saved to: {os.path.join(models_folder_path, modelName)}")
            print("Preprocessing used: Resampling (Interpolation) to 120 frames")
    else:
        print("Nie załadowano żadnych danych. Sprawdź ścieżkę data_source_path.")