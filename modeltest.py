from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best.pt")

# Open test video
cap = cv2.VideoCapture("t7.mp4")

if not cap.isOpened():
    print("❌ Could not open test.mp4")
    exit()

print("✅ Video opened successfully")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("✅ Video finished")
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)

    # Draw detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Fall Detection Test", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()