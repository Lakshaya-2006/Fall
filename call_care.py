import os
import time
import requests

from dotenv import load_dotenv
from twilio.rest import Client


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# TWILIO
# ============================================================

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")


# ============================================================
# CAREGIVERS
# ============================================================

CAREGIVERS = [
    os.getenv("CAREGIVER_1"),
    os.getenv("CAREGIVER_2"),
    os.getenv("CAREGIVER_3")
]


# ============================================================
# NGROK
# ============================================================

PUBLIC_URL = "https://tinkling-slapstick-scarce.ngrok-free.dev"


# ============================================================
# TWILIO CLIENT
# ============================================================

client = Client(
    ACCOUNT_SID,
    AUTH_TOKEN
)


# ============================================================
# RESET ALERT
# ============================================================

def reset_alert():

    try:

        response = requests.get(
            f"{PUBLIC_URL}/reset",
            timeout=10
        )

        if response.status_code == 200:

            print("🔄 Alert status reset")

    except Exception as e:

        print("⚠️ Reset error:", e)


# ============================================================
# CHECK ACKNOWLEDGEMENT
# ============================================================

def check_acknowledgement():

    try:

        response = requests.get(
            f"{PUBLIC_URL}/status",
            timeout=5
        )

        if response.status_code == 200:

            data = response.json()

            return data.get("acknowledged", False)

    except Exception as e:

        print("⚠️ Status error:", e)

    return False


# ============================================================
# CALL CAREGIVER
# ============================================================

def call_caregiver(phone_number, caregiver_number):

    print("\n")
    print("=" * 60)
    print(f"📞 CALLING CAREGIVER {caregiver_number}")
    print(f"Number: {phone_number}")
    print("=" * 60)

    try:

        # IMPORTANT:
        # Only basic parameters are used because
        # your Twilio account is a trial account.

        call = client.calls.create(
            to=phone_number,
            from_=TWILIO_NUMBER,
            url=f"{PUBLIC_URL}/voice"
        )

        print("✅ CALL CREATED")
        print("Call SID:", call.sid)

        return call.sid

    except Exception as e:

        print("❌ CALL CREATION FAILED")
        print(e)

        return None


# ============================================================
# WAIT FOR CALL RESULT
# ============================================================

def wait_for_call(call_sid):

    print("\n⏳ Waiting for caregiver response...")

    while True:

        # ----------------------------------------------------
        # Check acknowledgement
        # ----------------------------------------------------

        if check_acknowledgement():

            print("\n")
            print("=" * 60)
            print("✅ CAREGIVER ACKNOWLEDGED")
            print("🛑 STOPPING ESCALATION")
            print("=" * 60)

            return "acknowledged"


        # ----------------------------------------------------
        # Check Twilio call status
        # ----------------------------------------------------

        try:

            call = client.calls(call_sid).fetch()

            status = call.status

            print("📞 Call status:", status)

            # ------------------------------------------------
            # Call finished
            # ------------------------------------------------

            if status in [
                "completed",
                "busy",
                "failed",
                "canceled",
                "no-answer"
            ]:

                # Give Flask a little time to receive
                # the speech response.

                time.sleep(3)

                if check_acknowledgement():

                    print("✅ ACKNOWLEDGEMENT RECEIVED")

                    return "acknowledged"

                print("❌ CAREGIVER DID NOT ACKNOWLEDGE")

                return "no_response"

        except Exception as e:

            print("⚠️ Call status error:", e)

            return "no_response"

        time.sleep(2)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("🚨 FALL DETECTED")
    print("=" * 60)

    # Reset previous alert
    reset_alert()


    # ========================================================
    # SEQUENTIAL CAREGIVER ESCALATION
    # ========================================================

    for index, phone_number in enumerate(CAREGIVERS):

        caregiver_number = index + 1

        print("\n")
        print("=" * 60)
        print(f"========== CAREGIVER {caregiver_number} ==========")
        print("=" * 60)


        # ----------------------------------------------------
        # Check if already acknowledged
        # ----------------------------------------------------

        if check_acknowledgement():

            print("✅ Alert already acknowledged")

            return


        # ----------------------------------------------------
        # Check number
        # ----------------------------------------------------

        if not phone_number:

            print(
                f"⚠️ Caregiver {caregiver_number} "
                "number not configured"
            )

            continue


        # ----------------------------------------------------
        # Make call
        # ----------------------------------------------------

        call_sid = call_caregiver(
            phone_number,
            caregiver_number
        )


        if not call_sid:

            print(
                f"⚠️ Could not call Caregiver "
                f"{caregiver_number}"
            )

            continue


        # ----------------------------------------------------
        # Wait
        # ----------------------------------------------------

        result = wait_for_call(call_sid)


        # ----------------------------------------------------
        # ACKNOWLEDGED
        # ----------------------------------------------------

        if result == "acknowledged":

            print("\n")
            print("=" * 60)
            print("🚨 ALERT ACKNOWLEDGED")
            print("=" * 60)
            print(
                f"Caregiver {caregiver_number} "
                "confirmed the alert."
            )
            print("🛑 No more caregivers will be called.")
            print("=" * 60)

            return


        # ----------------------------------------------------
        # NO RESPONSE
        # ----------------------------------------------------

        print("\n")
        print("=" * 60)
        print(
            f"❌ CAREGIVER {caregiver_number} "
            "DID NOT RESPOND"
        )
        print("➡️ Moving to next caregiver...")
        print("=" * 60)


    # ========================================================
    # NOBODY RESPONDED
    # ========================================================

    print("\n")
    print("=" * 60)
    print("🚨 CRITICAL ALERT")
    print("=" * 60)
    print("No caregiver responded to the fall alert.")
    print("=" * 60)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()