from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather

app = Flask(__name__)

# ============================================================
# GLOBAL ALERT STATUS
# ============================================================

alert_acknowledged = False


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return """
    <h1>🚨 Fall Detection Voice Alert Server</h1>
    <p>Server is running.</p>
    """


# ============================================================
# VOICE CALL
# ============================================================

@app.route("/voice", methods=["POST", "GET"])
def voice():

    print("\n========================================")
    print("📞 CAREGIVER ANSWERED THE CALL")
    print("========================================")

    response = VoiceResponse()

    # --------------------------------------------------------
    # Speech recognition
    # --------------------------------------------------------

    gather = Gather(
        input="speech",
        action="/speech",
        method="POST",
        speech_timeout="auto",
        timeout=10
    )

    gather.say(
        "Emergency alert. "
        "A fall has been detected. "
        "Please say yes to acknowledge this alert.",
        voice="alice"
    )

    response.append(gather)

    # --------------------------------------------------------
    # If no speech was detected
    # --------------------------------------------------------

    response.say(
        "No response was received. Goodbye.",
        voice="alice"
    )

    response.hangup()

    return str(response)


# ============================================================
# RECEIVE SPEECH
# ============================================================

@app.route("/speech", methods=["POST"])
def speech():

    global alert_acknowledged

    # Twilio sends the recognized speech here
    speech_result = request.form.get("SpeechResult", "")

    confidence = request.form.get("Confidence", "")

    speech_text = speech_result.lower().strip()

    print("\n========================================")
    print("🎤 CAREGIVER SPEECH RECEIVED")
    print("Speech:", speech_result)
    print("Confidence:", confidence)
    print("========================================")

    # --------------------------------------------------------
    # Words/phrases that mean ACKNOWLEDGED
    # --------------------------------------------------------

    acknowledgement_words = [
        "yes",
        "yeah",
        "yep",
        "okay",
        "ok",
        "acknowledge",
        "acknowledged",
        "i am coming",
        "i'm coming",
        "coming",
        "on my way"
    ]

    acknowledged = False

    for word in acknowledgement_words:

        if word in speech_text:

            acknowledged = True
            break

    response = VoiceResponse()

    # --------------------------------------------------------
    # CAREGIVER ACKNOWLEDGED
    # --------------------------------------------------------

    if acknowledged:

        alert_acknowledged = True

        print("\n========================================")
        print("✅ CAREGIVER ACKNOWLEDGED THE ALERT")
        print("🛑 ESCALATION STOPPED")
        print("========================================")

        response.say(
            "Thank you. "
            "Your response has been received. "
            "The emergency alert has been acknowledged.",
            voice="alice"
        )

        response.hangup()

        return str(response)

    # --------------------------------------------------------
    # CAREGIVER DID NOT SAY YES
    # --------------------------------------------------------

    print("\n❌ Response was not recognized as acknowledgement")

    response.say(
        "Your response was not recognized. "
        "Goodbye.",
        voice="alice"
    )

    response.hangup()

    return str(response)


# ============================================================
# CHECK ACKNOWLEDGEMENT
# ============================================================

@app.route("/status", methods=["GET"])
def status():

    return jsonify({
        "acknowledged": alert_acknowledged
    })


# ============================================================
# RESET ALERT
# ============================================================

@app.route("/reset", methods=["GET"])
def reset():

    global alert_acknowledged

    alert_acknowledged = False

    print("\n🔄 Alert status reset")

    return jsonify({
        "acknowledged": False
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("🚨 FALL DETECTION VOICE SERVER")
    print("========================================")

    print("🌐 Local server:")
    print("http://127.0.0.1:5000")

    print("🎤 Speech endpoint:")
    print("/speech")

    print("========================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )