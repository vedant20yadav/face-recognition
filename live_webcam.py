import cv2
from roboflow import Roboflow
import supervision as sv
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key="riMiboxeLKguBWpBlc7J")
project = rf.workspace().project("real-time-facial-recognition")
model = project.version(1).model

# Start video capture (0 = default webcam, change to 1 if you have an external camera)
cap = cv2.VideoCapture(0)

# Create BYTETracker instance for tracking
byte_tracker = sv.ByteTrack(lost_track_buffer=30)
box_annotator = sv.BoxAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no video frame is captured

    # Resize frame for faster processing (optional)
    resized_frame = cv2.resize(frame, (640, 480))

    # Run face recognition on the frame
    results = model.predict(resized_frame).json()

    # Handle empty predictions safely
    if not results["predictions"]:
        detections = sv.Detections.empty()
    else:
        detections = sv.Detections(
            xyxy=np.array([
                [p["x"] - p["width"] / 2, p["y"] - p["height"] / 2, 
                 p["x"] + p["width"] / 2, p["y"] + p["height"] / 2] 
                for p in results["predictions"]
            ]),
            confidence=np.array([p["confidence"] for p in results["predictions"]]),
            class_id=np.array([p["class_id"] if "class_id" in p else 0 for p in results["predictions"]]),  
        )

    # Track detections
    detections = byte_tracker.update_with_detections(detections)

    # Prepare labels for detected faces
    labels = [
        f"ID: {tracker_id}, Conf: {confidence:.2f}"
        for (_, _, confidence, _, tracker_id) in detections
    ]

    # Annotate frame with bounding boxes
    # Draw bounding boxes without labels
annotated_frame = box_annotator.annotate(scene=resized_frame, detections=detections)

# Draw labels manually on top of the bounding boxes
for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
    label = labels[i] if i < len(labels) else "Unknown"
    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Ensure OpenCV window appears correctly
    cv2.namedWindow("Live Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Face Recognition", 800, 600)
    cv2.imshow("Live Face Recognition", annotated_frame)

    # Press 'q' to exit the live video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Properly release the camera and close windows
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # Ensures OpenCV processes the exit properly
cap = None  # Reset camera object
