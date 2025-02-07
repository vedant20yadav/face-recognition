import cv2
import numpy as np
import supervision as sv
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="riMiboxeLKguBWpBlc7J")
project = rf.workspace().project("real-time-facial-recognition")
model = project.version(1).model

# Input and output video files
SOURCE_VIDEO_PATH = "sample 1.mp4"  # Change to your video file name
TARGET_VIDEO_PATH = "detect.mp4"  # Processed output video

# Create BYTETracker instance for tracking
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Open video file
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
video_writer = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is read

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
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
    annotated_frame = box_annotator.annotate(scene=resized_frame, detections=detections, labels=labels)

    # Initialize video writer if not set
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 output
        video_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, 30, (640, 480))

    # Write processed frame to output video
    video_writer.write(annotated_frame)

    # Display processed video (optional)
    cv2.imshow("Face Recognition in Video", annotated_frame)
    
    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processing complete! Saved as {TARGET_VIDEO_PATH}")
