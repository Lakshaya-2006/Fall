import os
import time

import requests
from dotenv import load_dotenv
from twilio.base.exceptions import TwilioRestException
from twilio.http.http_client import TwilioHttpClient
from twilio.rest import Client

load_dotenv()

# ============================================================
# SETTINGS
# ============================================================

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

CAREGIVERS = [
    os.getenv("CAREGIVER_1"),
    os.getenv("CAREGIVER_2"),
    os.getenv("CAREGIVER_3"),
]

PUBLIC_URL = os.getenv("PUBLIC_URL", "https://tinkling-slapstick-scarce.ngrok-free.dev")  # ngrok (Twilio -> your server)
LOCAL_URL = "http://127.0.0.1:5000"                                                        # this script -> your server

RING_LIMIT = 40                # seconds we wait for an answer, then hang up and call the next caregiver
EXTRA_WAIT = 30                # extra safety time on top of RING_LIMIT
USE_MACHINE_DETECTION = False  # PAID Twilio accounts only (trial accounts reject it). True = ignore voicemail

TERMINAL = ("completed", "busy", "failed", "canceled", "no-answer")

# timeout=15: if the internet to Twilio is slow or blocked, fail after 15 s instead of hanging forever
client = Client(ACCOUNT_SID, AUTH_TOKEN, http_client=TwilioHttpClient(timeout=15))


# ============================================================
# HELPERS
# ============================================================

def reset_alert():
    try:
        requests.get(f"{LOCAL_URL}/reset", timeout=5)
        print("🔄 Alert status reset")
    except Exception as e:
        print("⚠️ Reset error (is server.py running?):", e)


def check_acknowledgement():
    """True once server.py has received the /voice request (= the call was answered)."""
    try:
        r = requests.get(f"{LOCAL_URL}/status", timeout=5)
        if r.status_code == 200:
            return r.json().get("acknowledged", False)
    except Exception:
        pass
    return False


def call_caregiver(phone_number, caregiver_number):
    print("\n" + "=" * 60)
    print(f"📞 CALLING CAREGIVER {caregiver_number}: {phone_number}")
    print("=" * 60)

    # Only basic parameters: Twilio TRIAL accounts reject extras such as timeout or machine_detection
    params = dict(to=phone_number, from_=TWILIO_NUMBER, url=f"{PUBLIC_URL}/voice")
    if USE_MACHINE_DETECTION:
        params["machine_detection"] = "Enable"

    for attempt in range(1, 4):   # retry up to 3 times on network problems
        try:
            call = client.calls.create(**params)
            print("✅ CALL CREATED, SID:", call.sid)
            return call.sid
        except TwilioRestException as e:      # Twilio refused the request: retrying will not help
            print("❌ CALL CREATION FAILED (Twilio rejected it):", e)
            return None
        except Exception as e:                # network problem: retry
            print(f"❌ CALL CREATION FAILED (network, attempt {attempt}/3): {e}")
            time.sleep(2)
    return None


def wait_for_call(call_sid):
    """
    Blocks until this call has a final outcome. Returns 'answered' or 'no_response'.
    It does NOT return while the phone is still ringing.
    """
    print(f"\n⏳ Waiting for the caregiver to attend (waiting up to {RING_LIMIT}s)...")
    started = time.time()
    last_status = None
    errors = 0

    while True:
        # 1) Signal from the server: the call was answered and Twilio requested /voice
        if check_acknowledgement():
            print("✅ Call attended (server received the answer)")
            return "answered"

        # 2) Ask Twilio what the call is doing
        try:
            call = client.calls(call_sid).fetch()
            errors = 0
        except Exception as e:
            errors += 1
            print(f"⚠️ Could not read call status ({errors}/5):", e)
            if errors >= 5:
                return "no_response"
            time.sleep(2)
            continue

        status = call.status
        if status != last_status:
            print(f"📞 Call status: {status}  ({int(time.time() - started)}s)")
            last_status = status

        # Caregiver picked up
        if status == "in-progress" and not USE_MACHINE_DETECTION:
            print("✅ Call attended (caregiver picked up)")
            return "answered"

        # The call has ended: decide whether it was ever attended
        if status in TERMINAL:
            time.sleep(2)  # give the server a moment to register /voice
            if check_acknowledgement():
                print("✅ Call attended")
                return "answered"
            if status == "completed" and int(call.duration or 0) > 0:
                print("✅ Call was answered (but server.py did not receive /voice, check server.py and the ngrok URL)")
                return "answered"
            reason = {"no-answer": "nobody picked up", "busy": "line was busy",
                      "failed": "call failed", "canceled": "call was canceled"}.get(status, "call ended without being answered")
            print(f"❌ Not attended: {reason}")
            return "no_response"

        # Not answered within our limit: hang up so the next caregiver can be called
        elapsed = time.time() - started
        ringing = status in ("queued", "ringing")
        if (ringing and elapsed > RING_LIMIT) or elapsed > RING_LIMIT + EXTRA_WAIT:
            print(f"⏱️ No answer after {int(elapsed)}s, hanging up")
            try:
                client.calls(call_sid).update(status="canceled" if ringing else "completed")
            except Exception as e:
                print("⚠️ Could not hang up the call (Twilio will stop it by itself):", e)
            time.sleep(2)
            return "answered" if check_acknowledgement() else "no_response"

        time.sleep(1)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚨 FALL DETECTED")
    print("=" * 60)

    reset_alert()

    for index, phone_number in enumerate(CAREGIVERS):
        caregiver_number = index + 1
        print(f"\n========== CAREGIVER {caregiver_number} ==========")

        if not phone_number:
            print(f"⚠️ Caregiver {caregiver_number} number not configured, skipping")
            continue

        call_sid = call_caregiver(phone_number, caregiver_number)
        if not call_sid:
            print(f"⚠️ Could not call caregiver {caregiver_number}, trying next")
            continue

        # The next caregiver is called only AFTER this returns
        result = wait_for_call(call_sid)

        if result == "answered":
            print("\n" + "=" * 60)
            print(f"✅ CAREGIVER {caregiver_number} ATTENDED — RESPONSE RECEIVED")
            print("🛑 No more caregivers will be called.")
            print("=" * 60)
            return

        print(f"\n➡️ Caregiver {caregiver_number} did not attend, calling the next caregiver...")
        time.sleep(1)

    print("\n" + "=" * 60)
    print("🚨 CRITICAL ALERT: no caregiver attended the fall alert.")
    print("=" * 60)


if __name__ == "__main__":
    main()