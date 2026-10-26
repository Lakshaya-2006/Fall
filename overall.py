
from pathlib import Path
from collections import deque
from datetime import datetime
import os
import subprocess
import sys
import threading
import time

import cv2
import joblib
import numpy as np
import serial
import torch
import torch.nn as nn
from torchvision import models, transforms
from dotenv import load_dotenv
from twilio.rest import Client


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Existing trained models
CAMERA_CHECKPOINT = BASE_DIR / "best.pth"
RADAR_MODEL_PATH = Path(
    r"C:\Users\Laks\OneDrive\Desktop\New folder\Fall"
    r"\live_radar_svm_temporal.pkl"
)

# Existing caregiver calling script
CALL_SCRIPT = BASE_DIR / "call_care.py"

# Existing .env file
load_dotenv(BASE_DIR / ".env")

# Camera: use 0 for live webcam, or a video path for testing.
VIDEO_SOURCE = 0

# Camera model settings
CAMERA_SEQ_LEN = 16
IMG_SIZE = 224
FALL_INDEX = 1
CAMERA_FALL_THRESHOLD = 0.7
CAMERA_CONFIRM_PREDS = 3

# Radar settings
RADAR_PORT = "COM3"
RADAR_BAUD_RATE = 115200
RADAR_WINDOW_SIZE = 20
RADAR_CONFIRM_PREDS = 3
RADAR_EXPECTED_FEATURES = 42

RADAR_FEATURE_NAMES = [
    "state",
    "moving_cm",
    "moving_energy",
    "stationary_cm",
    "stationary_energy",
    "detection_cm",
]

# Shared event settings
GET_UP_SECONDS = 5
ALERT_AFTER_SECONDS = 60

PATIENT_NAME = "Patient"
LOCATION = "Home"

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")
CAREGIVER_NUMBER = os.environ.get("CAREGIVER_NUMBER")

# Sensor states older than this are considered stale.
SENSOR_STALE_SECONDS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 2. CAMERA MODEL: PRESERVE EXISTING CNN-LSTM ARCHITECTURE
# =========================================================

class CNNLSTM(nn.Module):
    def __init__(self, hidden=128, num_classes=2):
        super().__init__()

        resnet = models.resnet18(weights=None)

        self.cnn = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]

        features = self.cnn(x.flatten(0, 1))
        features = self.pool(features).flatten(1)
        features = features.view(
            batch_size, seq_len, -1
        )

        output, _ = self.lstm(features)

        return self.fc(output[:, -1])


def load_camera_model():
    if not CAMERA_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Camera checkpoint not found:\n"
            f"{CAMERA_CHECKPOINT}\n"
            "Update CAMERA_CHECKPOINT to your existing best.pth."
        )

    model = CNNLSTM()

    checkpoint = torch.load(
        CAMERA_CHECKPOINT,
        map_location=DEVICE,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The camera checkpoint does not contain "
            "'model_state_dict'. Use your original checkpoint "
            "loading format."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


camera_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# =========================================================
# 3. LOAD EXISTING RADAR SVM MODEL
# =========================================================

def load_radar_model():
    if not RADAR_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Radar model not found:\n{RADAR_MODEL_PATH}"
        )

    model = joblib.load(RADAR_MODEL_PATH)

    if getattr(model, "n_features_in_", None) != (
        RADAR_EXPECTED_FEATURES
    ):
        raise ValueError(
            "Radar model must accept exactly 42 features."
        )

    return model


# =========================================================
# 4. RADAR DATA PARSING AND FEATURE EXTRACTION
# =========================================================

def parse_radar_line(line):
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


def extract_radar_features(values):
    if values.shape != (RADAR_WINDOW_SIZE, 6):
        raise ValueError(
            f"Expected ({RADAR_WINDOW_SIZE}, 6), "
            f"got {values.shape}"
        )

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)

    mean_abs_change = np.abs(
        np.diff(values, axis=0)
    ).mean(axis=0)

    start_to_end = values[-1] - values[0]

    # Match your existing training feature calculation.
    sample_indices = np.arange(
        RADAR_WINDOW_SIZE, dtype=float
    )
    centered_indices = (
        sample_indices - sample_indices.mean()
    )

    slopes = np.array([
        np.polyfit(
            centered_indices,
            values[:, j],
            1,
        )[0]
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

    if features.size != RADAR_EXPECTED_FEATURES:
        raise ValueError(
            f"Expected 42 features, got {features.size}"
        )

    return features.reshape(1, -1)


# =========================================================
# 5. EXISTING ALERT INTEGRATION
# =========================================================

def trigger_caregiver_call():
    if not CALL_SCRIPT.is_file():
        print("CALL SCRIPT NOT FOUND:", CALL_SCRIPT)
        return False

    try:
        subprocess.Popen(
            [sys.executable, str(CALL_SCRIPT)],
            cwd=str(BASE_DIR),
        )

        print("Caregiver calling process started.")
        return True

    except Exception as error:
        print("Could not start caregiver call:", error)
        return False


def send_sms(body):
    required = {
        "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
        "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN,
        "TWILIO_FROM": TWILIO_FROM,
        "CAREGIVER_NUMBER": CAREGIVER_NUMBER,
    }

    missing = [
        name for name, value in required.items()
        if not value
    ]

    if missing:
        print(
            "SMS NOT SENT. Missing environment values:",
            ", ".join(missing),
        )
        return False

    try:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN,
        )

        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM,
            to=CAREGIVER_NUMBER,
        )

        print("SMS submitted. SID:", message.sid)
        return True

    except Exception as error:
        print("SMS FAILED:", error)
        return False


# =========================================================
# 6. SHARED SENSOR-FUSION AND ALERT CONTROLLER
# =========================================================

lock = threading.Lock()
stop_event = threading.Event()

sensor_state = {
    "camera": None,
    "radar": None,
}

sensor_last_update = {
    "camera": None,
    "radar": None,
}

fall_hits = {
    "camera": 0,
    "radar": 0,
}

event = {
    "active": False,
    "start": None,
    "wallclock": None,
    "call_started": False,
    "sms_attempted": False,
    "both_normal_since": None,
}


def update_sensor(sensor_name, prediction):
    """
    prediction: 1 = FALL, 0 = NORMAL.

    Either sensor can start a shared fall event.
    The shared event starts only after consecutive
    fall predictions from that sensor.
    """

    should_call = False

    with lock:
        sensor_last_update[sensor_name] = time.monotonic()
        sensor_state[sensor_name] = prediction

        if prediction == 1:
            fall_hits[sensor_name] += 1
        else:
            fall_hits[sensor_name] = 0

        confirmed = (
            fall_hits[sensor_name]
            >= (
                CAMERA_CONFIRM_PREDS
                if sensor_name == "camera"
                else RADAR_CONFIRM_PREDS
            )
        )

        # Either sensor can confirm the fall.
        if confirmed and not event["active"]:
            event["active"] = True
            event["start"] = time.monotonic()
            event["wallclock"] = datetime.now()
            event["call_started"] = True
            event["sms_attempted"] = False
            event["both_normal_since"] = None

            print("\n" + "!" * 55)
            print(
                f"FALL CONFIRMED BY {sensor_name.upper()} "
                f"at {event['wallclock']:%H:%M:%S}"
            )
            print("!" * 55 + "\n")

            # Only one call-process launch per event.
            should_call = True

    if should_call:
        trigger_caregiver_call()


def alert_monitor():
    """
    Runs independently of the camera frame loop.
    Handles recovery and the delayed SMS alert.
    """

    while not stop_event.wait(0.25):
        sms_body = None

        with lock:
            if not event["active"]:
                continue

            now = time.monotonic()

            # Require both sensors to be recently reporting NORMAL.
            both_normal = all(
                sensor_state[name] == 0
                and sensor_last_update[name] is not None
                and now - sensor_last_update[name]
                    <= SENSOR_STALE_SECONDS
                for name in ("camera", "radar")
            )

            if both_normal:
                if event["both_normal_since"] is None:
                    event["both_normal_since"] = now

                elif (
                    now - event["both_normal_since"]
                    >= GET_UP_SECONDS
                ):
                    print(
                        "Both sensors reported NORMAL "
                        "for 5 seconds. Resetting event."
                    )

                    event["active"] = False
                    event["start"] = None
                    event["wallclock"] = None
                    event["call_started"] = False
                    event["sms_attempted"] = False
                    event["both_normal_since"] = None

                    fall_hits["camera"] = 0
                    fall_hits["radar"] = 0

                    continue
            else:
                event["both_normal_since"] = None

            duration = now - event["start"]

            if (
                not event["sms_attempted"]
                and duration >= ALERT_AFTER_SECONDS
            ):
                minutes = int(duration // 60)
                fall_time = event["wallclock"]

                sms_body = (
                    f"EMERGENCY ALERT: {PATIENT_NAME} "
                    f"may have fallen at {fall_time:%I:%M %p}. "
                    f"Location: {LOCATION}. "
                    f"Neither-sensor recovery confirmation "
                    f"has not completed. Fall event duration: "
                    f"{minutes} minute(s). "
                    "Please check immediately."
                )

                # Prevent repeated SMS attempts for this event.
                event["sms_attempted"] = True

        # Do not hold the shared lock while contacting Twilio.
        if sms_body is not None:
            send_sms(sms_body)


# =========================================================
# 7. RADAR WORKER THREAD
# =========================================================

def radar_worker(radar_model):
    rows = deque(maxlen=RADAR_WINDOW_SIZE)

    try:
        with serial.Serial(
            RADAR_PORT,
            RADAR_BAUD_RATE,
            timeout=1,
        ) as ser:

            time.sleep(2)
            ser.reset_input_buffer()

            print(
                f"[RADAR] Serial connected: {RADAR_PORT}"
            )

            while not stop_event.is_set():
                raw_line = ser.readline().decode(
                    "utf-8",
                    errors="ignore",
                ).strip()

                if not raw_line:
                    continue

                values = parse_radar_line(raw_line)

                if values is None:
                    continue

                rows.append(values)

                if len(rows) < RADAR_WINDOW_SIZE:
                    continue

                window = np.asarray(rows, dtype=float)
                features = extract_radar_features(window)

                prediction = int(
                    radar_model.predict(features)[0]
                )

                label = (
                    "FALL" if prediction == 1
                    else "NORMAL" if prediction == 0
                    else f"UNKNOWN ({prediction})"
                )

                print(f"[RADAR] {label}")

                if prediction in (0, 1):
                    update_sensor("radar", prediction)

    except serial.SerialException as error:
        print("[RADAR] SERIAL ERROR:", error)
        print(
            "[RADAR] Check COM3 and close Arduino Serial Monitor."
        )

    except Exception as error:
        print("[RADAR] WORKER ERROR:", repr(error))


# =========================================================
# 8. CAMERA WORKER / MAIN LOOP
# =========================================================

def run_camera(camera_model):
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera/video: {VIDEO_SOURCE}"
        )

    frame_buffer = deque(maxlen=CAMERA_SEQ_LEN)
    frame_index = 0

    try:
        print("[CAMERA] Camera/video opened.")

        while not stop_event.is_set():
            ok, frame = cap.read()

            if not ok:
                print("[CAMERA] Video ended or frame unavailable.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(camera_transform(rgb))

            label = "Collecting camera frames"
            color = (200, 200, 200)

            if len(frame_buffer) == CAMERA_SEQ_LEN:
                clip = torch.stack(
                    list(frame_buffer)
                ).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    probabilities = torch.softmax(
                        camera_model(clip),
                        dim=1,
                    )[0]

                    p_fall = probabilities[FALL_INDEX].item()

                prediction = int(
                    p_fall >= CAMERA_FALL_THRESHOLD
                )

                update_sensor("camera", prediction)

                label = (
                    f"CAMERA FALL {p_fall:.2f}"
                    if prediction == 1
                    else f"CAMERA NORMAL {1 - p_fall:.2f}"
                )

                color = (
                    (0, 0, 255)
                    if prediction == 1
                    else (0, 200, 0)
                )

            with lock:
                event_active = event["active"]

            if event_active:
                label = "ELDERGUARD: FALL EVENT ACTIVE"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                label,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

            cv2.imshow(
                "ElderGuard AI - Camera + Radar",
                frame,
            )

            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[CAMERA] Stop requested.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# 9. START THE COMPLETE APPLICATION
# =========================================================

def main():
    print("=" * 55)
    print("ELDERGUARD AI - CAMERA + RADAR INTEGRATION")
    print("=" * 55)
    print("Device:", DEVICE)
    print("Camera source:", VIDEO_SOURCE)
    print("Radar port:", RADAR_PORT)
    print("Fall rule: EITHER sensor can confirm a fall")
    print("SMS delay:", ALERT_AFTER_SECONDS, "seconds")
    print("=" * 55)

    camera_model = load_camera_model()
    radar_model = load_radar_model()

    print("Camera CNN-LSTM model loaded.")
    print("Radar SVM model loaded.")

    monitor_thread = threading.Thread(
        target=alert_monitor,
        daemon=True,
    )

    radar_thread = threading.Thread(
        target=radar_worker,
        args=(radar_model,),
        daemon=True,
    )

    monitor_thread.start()
    radar_thread.start()

    try:
        run_camera(camera_model)

    finally:
        stop_event.set()
        cv2.destroyAllWindows()

        radar_thread.join(timeout=2)
        monitor_thread.join(timeout=2)

        print("ElderGuard AI stopped.")


if __name__ == "__main__":
    main()
