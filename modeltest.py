import os
import time
from collections import deque
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from twilio.rest import Client

# ---------------- CONFIG ----------------
CHECKPOINT = "best.pth"
VIDEO_SOURCE = "t14.mp4"          # use 0 for webcam
SEQ_LEN = 16
FRAME_STRIDE = 1
IMG_SIZE = 224
FALL_INDEX = 1                   # output index meaning "fall"
FALL_THRESHOLD = 0.7

FALL_CONFIRM_PREDS = 3           # consecutive "fall" predictions needed to start a fall event
GET_UP_SECONDS = 5         # continuous "no fall" for this long = person got up
ALERT_AFTER_SECONDS = 1 * 60    # send SMS if still down after this long (set to 15 to test)

PATIENT_NAME = "Patient"         # name shown in the SMS
LOCATION = ""                    # optional, e.g. "Living room"

# Credentials come from environment variables (never hardcode them)
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")            # your Twilio number, e.g. +1415XXXXXXX
CAREGIVER_NUMBER = os.environ.get("CAREGIVER_NUMBER")  # e.g. +91XXXXXXXXXX (E.164 format)
# ----------------------------------------


def send_sms(body):
    missing = [n for n, v in [("TWILIO_ACCOUNT_SID", TWILIO_ACCOUNT_SID),
                              ("TWILIO_AUTH_TOKEN", TWILIO_AUTH_TOKEN),
                              ("TWILIO_FROM", TWILIO_FROM),
                              ("CAREGIVER_NUMBER", CAREGIVER_NUMBER)] if not v]
    if missing:
        print("SMS NOT SENT, missing environment variables:", ", ".join(missing))
        return False
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(body=body, from_=TWILIO_FROM, to=CAREGIVER_NUMBER)
        print(f"SMS sent to {CAREGIVER_NUMBER} (sid {msg.sid})")
        return True
    except Exception as e:
        print("SMS failed:", e)
        return False


class CNNLSTM(nn.Module):
    def __init__(self, hidden=128, num_classes=2):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(512, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        B, T = x.shape[:2]
        f = self.cnn(x.flatten(0, 1))
        f = self.pool(f).flatten(1).view(B, T, -1)
        out, _ = self.lstm(f)
        return self.fc(out[:, -1])


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNLSTM()
ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {VIDEO_SOURCE}")

is_file = isinstance(VIDEO_SOURCE, str)
fps = cap.get(cv2.CAP_PROP_FPS) or 30


def clock(frame_idx):
    # Video files use video time (so a 2-minute clip behaves the same however fast it runs);
    # webcam uses real time.
    return frame_idx / fps if is_file else time.time()


buf = deque(maxlen=SEQ_LEN)
frame_idx = 0
text, color = "Collecting frames...", (200, 200, 200)

# Fall event state
fallen = False
fall_hits = 0
fall_start = None
fall_wallclock = None
no_fall_since = None
alert_sent = False

while True:
    ok, frame = cap.read()
    if not ok:
        break

    t = clock(frame_idx)

    if frame_idx % FRAME_STRIDE == 0:
        buf.append(tf(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if len(buf) == SEQ_LEN:
            clip = torch.stack(list(buf)).unsqueeze(0).to(device)
            with torch.no_grad():
                p_fall = torch.softmax(model(clip), dim=1)[0][FALL_INDEX].item()
            is_fall = p_fall >= FALL_THRESHOLD

            if not fallen:
                fall_hits = fall_hits + 1 if is_fall else 0
                if fall_hits >= FALL_CONFIRM_PREDS:
                    fallen, fall_start, no_fall_since, alert_sent = True, t, None, False
                    fall_wallclock = datetime.now()
                    print(f"[{fall_wallclock:%H:%M:%S}] Fall detected, timer started")
                text, color = ("Fall suspected...", (0, 165, 255)) if fall_hits else (
                    f"No fall ({1 - p_fall:.2f})", (0, 200, 0))
            else:
                if is_fall:
                    no_fall_since = None
                else:
                    if no_fall_since is None:
                        no_fall_since = t
                    elif t - no_fall_since >= GET_UP_SECONDS:
                        print("Person appears to have gotten up, timer reset")
                        fallen, fall_hits = False, 0

                if fallen:
                    down_for = t - fall_start
                    text, color = f"FALLEN {int(down_for)}s", (0, 0, 255)
                    if not alert_sent and down_for >= ALERT_AFTER_SECONDS:
                        minutes = int(down_for // 60)
                        body = (f"ALERT: {PATIENT_NAME} fell down at {fall_wallclock:%I:%M %p}"
                                f"{' (' + LOCATION + ')' if LOCATION else ''} "
                                f"and has not gotten up for over {minutes} minutes. "
                                f"Please check on them immediately.")
                        alert_sent = send_sms(body)

    frame_idx += 1
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fall detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()