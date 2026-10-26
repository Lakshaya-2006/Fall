
from pathlib import Path
import time

import joblib
import numpy as np
import serial


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "live_radar_svm_temporal.pkl"

PORT = "COM3"
BAUD_RATE = 115200
WINDOW_SIZE = 20

FEATURES = [
    "state",
    "moving_cm",
    "moving_energy",
    "stationary_cm",
    "stationary_energy",
    "detection_cm",
]


# ==========================================
# 2. LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

if getattr(model, "n_features_in_", None) != 42:
    raise ValueError("Expected a model trained with 42 features.")

print("=" * 50)
print("ELDERLY FALL DETECTION - LIVE RADAR")
print("=" * 50)
print("Model:", MODEL_PATH.name)
print("Serial port:", PORT)
print("Baud rate:", BAUD_RATE)
print("Window size:", WINDOW_SIZE)
print("Input: six comma-separated values")
print("Press Ctrl+C to stop.")
print("=" * 50)


# ==========================================
# 3. PARSE RADAR DATA
# ==========================================

def parse_radar_line(line):
    """
    Expected line:
    3,143,100,209,100,162

    Order:
    state, moving_cm, moving_energy,
    stationary_cm, stationary_energy, detection_cm
    """
    try:
        parts = [part.strip() for part in line.split(",")]

        if len(parts) != 6:
            return None

        values = np.asarray(parts, dtype=float)

        if not np.all(np.isfinite(values)):
            return None

        return values

    except ValueError:
        return None


# ==========================================
# 4. EXTRACT 42 TEMPORAL FEATURES
# ==========================================


def extract_features(values, times):
    if values.shape != (WINDOW_SIZE, 6):
        raise ValueError(
            f"Expected ({WINDOW_SIZE}, 6), got {values.shape}"
        )

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)

    mean_abs_change = np.abs(np.diff(values, axis=0)).mean(axis=0)
    start_to_end = values[-1] - values[0]

    # Must match the training script: slope per sample index.
    sample_times = np.arange(len(values), dtype=float)
    centered_times = sample_times - sample_times.mean()

    slopes = np.array([
        np.polyfit(centered_times, values[:, j], 1)[0]
        for j in range(values.shape[1])
    ])

    features = np.concatenate([
        means,
        stds,
        mins,
        maxs,
        mean_abs_change,
        start_to_end,
        slopes,
    ])

    if features.size != 42:
        raise ValueError(f"Expected 42 features, got {features.size}")

    return features.reshape(1, -1)

# ==========================================
# 5. LIVE PREDICTION
# ==========================================

rows = []
timestamps = []

try:
    with serial.Serial(PORT, BAUD_RATE, timeout=2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()

        print("\nSerial connection opened.")
        print("Reading radar data...\n")

        while True:
            raw_line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not raw_line:
                continue

            print("RAW:", raw_line)

            values = parse_radar_line(raw_line)

            if values is None:
                print("Skipping invalid message.")
                continue

            print(
                "MATCHED:",
                dict(zip(FEATURES, values.tolist()))
            )

            rows.append(values)
            timestamps.append(time.time())

            if len(rows) > WINDOW_SIZE:
                rows.pop(0)
                timestamps.pop(0)

            print(f"Window: {len(rows)}/{WINDOW_SIZE}")

            if len(rows) < WINDOW_SIZE:
                continue

            window = np.asarray(rows, dtype=float)
            times = np.asarray(timestamps, dtype=float)

            x = extract_features(window, times)

            # Get model prediction.
            prediction = int(model.predict(x)[0])

            # Show decision score when supported.
            if hasattr(model, "decision_function"):
                score_values = model.decision_function(x)
                score = float(np.asarray(score_values).ravel()[0])
                print("SVM decision score:", round(score, 4))

            if prediction == 1:
                label = "FALL"
            elif prediction == 0:
                label = "NORMAL"
            else:
                label = f"UNKNOWN ({prediction})"

            print("\n" + "=" * 45)
            print("RADAR PREDICTION:", label)
            print("State:", int(values[0]))
            print("Moving distance:", int(values[1]), "cm")
            print("Moving energy:", int(values[2]))
            print("Stationary distance:", int(values[3]), "cm")
            print("Stationary energy:", int(values[4]))
            print("Detection distance:", int(values[5]), "cm")
            print("=" * 45 + "\n")


except KeyboardInterrupt:
    print("\nPrediction stopped by user.")

except serial.SerialException as error:
    print("\nSERIAL CONNECTION ERROR:", error)
    print("Check COM3 and close Arduino Serial Monitor.")

except Exception as error:
    print("\nPREDICTION ERROR:", repr(error))
    raise
