from datetime import datetime

from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

# ============================================================
# ALERT STATUS
# ============================================================
# Rule:  caregiver ANSWERS the call  -> response (acknowledged)
#        caregiver does NOT answer   -> no response (call_care.py moves to the next caregiver)

alert_acknowledged = False
answered_at = None

# Only used when machine detection is switched on in call_care.py
MACHINE_VALUES = ("machine_start", "machine_end_beep", "machine_end_silence", "machine_end_other", "fax")


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return "<h1>Fall Detection Voice Alert Server</h1><p>Server is running.</p>"


# ============================================================
# VOICE CALL  (Twilio requests this URL only when the call is answered)
# ============================================================

@app.route("/voice", methods=["POST", "GET"])
def voice():
    global alert_acknowledged, answered_at

    to_number = request.values.get("To", "")
    answered_by = request.values.get("AnsweredBy", "")

    response = VoiceResponse()

    # Voicemail / answering machine does not count as a person attending the call
    if answered_by in MACHINE_VALUES:
        print(f"\n📠 Answering machine detected ({answered_by}) -> NOT counted as a response")
        response.hangup()
        return str(response)

    alert_acknowledged = True
    answered_at = datetime.now().isoformat(timespec="seconds")

    print("\n========================================")
    print(f"📞 CAREGIVER ANSWERED THE CALL ({to_number})")
    print("✅ COUNTED AS A RESPONSE — ESCALATION STOPPED")
    print("========================================")

    message = ("Emergency alert. A fall has been detected, and the person has not gotten up. "
               "Please check on them immediately.")
    response.say(message, voice="alice")
    response.pause(length=1)
    response.say(message, voice="alice")
    response.hangup()
    return str(response)


# ============================================================
# STATUS / RESET
# ============================================================

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"acknowledged": alert_acknowledged, "answered_at": answered_at})


@app.route("/reset", methods=["GET"])
def reset():
    global alert_acknowledged, answered_at
    alert_acknowledged = False
    answered_at = None
    print("\n🔄 Alert status reset")
    return jsonify({"acknowledged": False})


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    print("\n🚨 FALL DETECTION VOICE SERVER  ->  http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)