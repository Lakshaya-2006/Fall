
import os
import time
import subprocess
import sys
from collections import deque
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from twilio.rest import Client

# ---------------- CONFIG ----------------
CHECKPOINT = "best.pth"
VIDEO_SOURCE = "t14.mp4"       # Use 0 for webcam

SEQ_LEN = 16
FRAME_STRIDE = 1
IMG_SIZE = 224

FALL_INDEX = 1                 # Output index meaning "fall"
FALL_THRESHOLD = 0.7
FALL_CONFIRM_PREDS = 3         # Consecutive fall predictions required

GET_UP_SECONDS = 5             # No-fall duration needed to reset event
ALERT_AFTER_SECONDS = 60       # Send SMS after 60 seconds

PATIENT_NAME = "Patient"
LOCATION = ""                  # Example: "Living room"

# Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")
CAREGIVER_NUMBER = os.environ.get("CAREGIVER_NUMBER")
# ----------------------------------------


def send_sms(body):
    """Send one SMS alert for the current fall event."""

    missing = [
        name
        for name, value in [
            ("TWILIO_ACCOUNT_SID", TWILIO_ACCOUNT_SID),
            ("TWILIO_AUTH_TOKEN", TWILIO_AUTH_TOKEN),
            ("TWILIO_FROM", TWILIO_FROM),
            ("CAREGIVER_NUMBER", CAREGIVER_NUMBER),
        ]
        if not value
    ]

    if missing:
        print("SMS NOT SENT. Missing variables:", ", ".join(missing))
        return False

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM,
            to=CAREGIVER_NUMBER,
        )

        print(f"SMS sent successfully. SID: {message.sid}")
        return True

    except Exception as error:
        print("SMS failed:", error)
        return False


def trigger_caregiver_call():
    """Launch call_care.py only when a new fall event is confirmed."""

    project_dir = os.path.dirname(os.path.abspath(__file__))
    call_script = os.path.join(project_dir, "call_care.py")

    if not os.path.isfile(call_script):
        print(f"ERROR: call_care.py not found: {call_script}")
        return False

    try:
        subprocess.Popen(
            [sys.executable, call_script],
            cwd=project_dir,
        )

        print("Caregiver calling process started.")
        return True

    except Exception as error:
        print("Failed to start caregiver calling process:", error)
        return False


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
        batch_size, sequence_length = x.shape[:2]

        features = self.cnn(x.flatten(0, 1))
        features = self.pool(features).flatten(1)
        features = features.view(batch_size, sequence_length, -1)

        output, _ = self.lstm(features)

        return self.fc(output[:, -1])


# ---------------- LOAD MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNNLSTM()

checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {VIDEO_SOURCE}")

is_file = isinstance(VIDEO_SOURCE, str)
fps = cap.get(cv2.CAP_PROP_FPS)

if not fps or fps <= 0:
    fps = 30.0


def clock(frame_idx):
    # Video input uses video time; webcam uses real time.
    return frame_idx / fps if is_file else time.time()


# ---------------- DETECTION STATE ----------------
buffer = deque(maxlen=SEQ_LEN)
frame_idx = 0

text = "Collecting frames..."
color = (200, 200, 200)

fallen = False
fall_hits = 0

fall_start = None
fall_wallclock = None
no_fall_since = None

alert_sent = False
call_triggered = False

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Video ended or camera frame unavailable.")
            break

        current_time = clock(frame_idx)

        if frame_idx % FRAME_STRIDE == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(transform(rgb_frame))

            if len(buffer) == SEQ_LEN:
                clip = torch.stack(list(buffer)).unsqueeze(0).to(device)

                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(clip), dim=1
                    )[0]

                    p_fall = probabilities[FALL_INDEX].item()

                is_fall = p_fall >= FALL_THRESHOLD

                # 1. Confirm a new fall only if no fall is active.
                if not fallen:
                    fall_hits = fall_hits + 1 if is_fall else 0

                    if fall_hits >= FALL_CONFIRM_PREDS:
                        fallen = True
                        fall_start = current_time
                        fall_wallclock = datetime.now()
                        no_fall_since = None
                        alert_sent = False

                        print(
                            f"[{fall_wallclock:%H:%M:%S}] "
                            "FALL DETECTED!"
                        )

                        # Launch once for this confirmed fall event.
                        if not call_triggered:
                            call_triggered = True
                            trigger_caregiver_call()

                # 2. Check whether the person appears to have recovered.
                if fallen:
                    if is_fall:
                        no_fall_since = None

                    elif no_fall_since is None:
                        no_fall_since = current_time

                    elif current_time - no_fall_since >= GET_UP_SECONDS:
                        print(
                            "Person appears to have gotten up. "
                            "Resetting fall event."
                        )

                        fallen = False
                        fall_hits = 0
                        fall_start = None
                        fall_wallclock = None
                        no_fall_since = None
                        alert_sent = False
                        call_triggered = False

                # 3. Send SMS after the configured delay.
                if fallen and fall_start is not None:
                    down_for = current_time - fall_start

                    text = f"FALLEN {int(down_for)}s"
                    color = (0, 0, 255)

                    if (
                        not alert_sent
                        and down_for >= ALERT_AFTER_SECONDS
                    ):
                        minutes = int(down_for // 60)

                        body = (
                            f"ALERT: {PATIENT_NAME} fell down at "
                            f"{fall_wallclock:%I:%M %p}"
                            f"{' (' + LOCATION + ')' if LOCATION else ''} "
                            f"and has not gotten up for over "
                            f"{minutes} minutes. "
                            "Please check on them immediately."
                        )

                        # Mark attempted to avoid repeated SMS attempts
                        # for the same fall event if sending fails.
                        alert_sent = True
                        send_sms(body)

                else:
                    text = f"No fall ({1 - p_fall:.2f})"
                    color = (0, 200, 0)

        frame_idx += 1

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        cv2.imshow("ElderGuard AI - Fall Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopping fall detection.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
