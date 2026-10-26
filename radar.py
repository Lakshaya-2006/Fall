
from pathlib import Path
import time

import joblib
import numpy as np
import serial


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = Path(
    r"C:\Users\Laks\OneDrive\Desktop\New folder\Fall\live_radar_svm_temporal.pkl"
)

PORT = "COM3"
BAUD_RATE = 115200
WINDOW_SIZE = 20
EXPECTED_FEATURES = 42

FEATURES = [
    "state",
    "moving_cm",
    "moving_energy",
    "stationary_cm",
    "stationary_energy",
    "detection_cm",
]


# ==========================================
# 2. LOAD AND VALIDATE MODEL
# ==========================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}\n\n"
        "Make sure live_radar_svm_temporal.pkl exists "
        "at the specified location."
    )

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    raise RuntimeError(
        f"Could not load the model: {error}\n"
        "Check the model file and scikit-learn version."
    ) from error

if getattr(model, "n_features_in_", None) != EXPECTED_FEATURES:
    raise ValueError(
        "Model feature mismatch. This program requires a model "
        "trained with exactly 42 input features."
    )

print("=" * 50)
print("ELDERLY FALL DETECTION - LIVE RADAR")
print("=" * 50)
print(f"Model: {MODEL_PATH.name}")
print(f"Serial port: {PORT}")
print(f"Baud rate: {BAUD_RATE}")
print(f"Window size: {WINDOW_SIZE}")
print("Expected input: six comma-separated numbers")
print("Press Ctrl+C to stop.")
print("=" * 50)


# ==========================================
# 3. PARSE RADAR DATA
# ==========================================

def parse_radar_line(line):
    """
    Expected format:
    state,moving_cm,moving_energy,
    stationary_cm,stationary_energy,detection_cm
    """
    parts = [part.strip() for part in line.split(",")]

    if len(parts) != 6:
        return None

    try:
        values = np.asarray(parts, dtype=float)
    except ValueError:
        return None

    if not np.all(np.isfinite(values)):
        return None

    return values


# ==========================================
# 4. EXTRACT 42 TEMPORAL FEATURES
# ==========================================

def extract_features(values):
    if values.shape != (WINDOW_SIZE, 6):
        raise ValueError(
            f"Expected shape ({WINDOW_SIZE}, 6), "
            f"received {values.shape}"
        )

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)

    mean_abs_change = np.abs(
        np.diff(values, axis=0)
    ).mean(axis=0)

    start_to_end = values[-1] - values[0]

    # Slope per sample index, not per second.
    sample_indices = np.arange(WINDOW_SIZE, dtype=float)
    centered_indices = sample_indices - sample_indices.mean()

    slopes = np.array([
        np.polyfit(centered_indices, values[:, j], 1)[0]
        for j in range(6)
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

    if features.size != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected 42 features, received {features.size}"
        )

    return features.reshape(1, -1)


# ==========================================
# 5. LIVE RADAR PREDICTION
# ==========================================

rows = []

try:
    with serial.Serial(
        PORT,
        BAUD_RATE,
        timeout=2
    ) as ser:

        time.sleep(2)
        ser.reset_input_buffer()

        print("\nSerial connection opened successfully.")
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

            if len(rows) > WINDOW_SIZE:
                rows.pop(0)

            print(f"Window: {len(rows)}/{WINDOW_SIZE}")

            if len(rows) < WINDOW_SIZE:
                continue

            window = np.asarray(rows, dtype=float)
            x = extract_features(window)

            prediction = int(model.predict(x)[0])

            if hasattr(model, "decision_function"):
                score_values = model.decision_function(x)
                score = float(
                    np.asarray(score_values).ravel()[0]
                )
                print("SVM decision score:", round(score, 4))

            if prediction == 1:
                label = "FALL"
            elif prediction == 0:
                label = "NORMAL"
            else:
                label = f"UNKNOWN ({prediction})"

            latest = window[-1]

            print("\n" + "=" * 45)
            print("RADAR PREDICTION:", label)
            print("State:", int(latest[0]))
            print("Moving distance:", int(latest[1]), "cm")
            print("Moving energy:", int(latest[2]))
            print("Stationary distance:", int(latest[3]), "cm")
            print("Stationary energy:", int(latest[4]))
            print("Detection distance:", int(latest[5]), "cm")
            print("=" * 45 + "\n")


except KeyboardInterrupt:
    print("\nPrediction stopped by user.")

except serial.SerialException as error:
    print("\nSERIAL CONNECTION ERROR:", error)
    print("Check that:")
    print("- The ESP32 is connected.")
    print("- COM3 is the correct port.")
    print("- Arduino Serial Monitor is closed.")
    print("- No other program is using COM3.")

except Exception as error:
    print(f"\nPREDICTION ERROR: {error}")
    raise
