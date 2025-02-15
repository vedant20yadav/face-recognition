import cv2
from roboflow import Roboflow
import supervision as sv
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key="LhTB2xgn5tVxptXfmyYD")
project = rf.workspace().project("face-recognition-tw2ab")
model = project.version(4).model

# Start video capture (0 = default webcam, change to 1 if you have an external camera)
cap = cv2.VideoCapture(0)

# Create BYTETracker instance for tracking
byte_tracker = sv.ByteTrack(lost_track_buffer=30, match_threshold=0.8)


box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no video frame is captured

    # Resize frame for faster processing (optional)
    resized_frame = cv2.resize(frame, (640, 480))

    # Run face recognition on the frame
    results = model.predict(resized_frame).json()
    detections = sv.Detections.from_roboflow(results)

    # Track detections
    detections = byte_tracker.update_with_detections(detections)

    # Prepare labels for detected faces
    labels = [
        f"ID: {tracker_id}, Conf: {confidence:.2f}"
        for _, _, confidence, _, tracker_id in detections
    ]

    # Annotate frame with bounding boxes
    annotated_frame = box_annotator.annotate(scene=resized_frame, detections=detections, labels=labels)

    # Display live video feed with detections
    cv2.imshow("Live Face Recognition", annotated_frame)

    # Press 'q' to exit the live video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
