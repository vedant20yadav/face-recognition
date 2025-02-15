import os
import cv2
from roboflow import Roboflow

# Get API Key
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found! Make sure it's set in environment variables.")

# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("face-recognition-tw2ab")
model = project.version(4).model  # Update if needed

# Define video source
video_path = "Face Recognition/sample.mp4"  # Change this if using another video
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("Face Recognition/detect.mp4", fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame for detection
    cv2.imwrite("frame.jpg", frame)
    results = model.predict("frame.jpg", confidence=40, overlap=30).json()

    # Draw bounding boxes
    for pred in results["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        start = (x - w // 2, y - h // 2)
        end = (x + w // 2, y + h // 2)
        cv2.rectangle(frame, start, end, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Video Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
