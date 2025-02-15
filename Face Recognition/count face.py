import os
import requests
from roboflow import Roboflow
import cv2

# Get API Key
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found! Make sure it's set in environment variables.")

# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("face-recognition-tw2ab")
model = project.version(4).model  # Update if needed

# Define input image
input_image = "Face Recognition/f.jpg"

# Run prediction
results = model.predict(input_image, confidence=40, overlap=30).json();
print(f"Total Faces Detected: {len(results['predictions'])}");

# Display output
image = cv2.imread(input_image);
for pred in results["predictions"]:
    x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"]);
    start = (x - w // 2, y - h // 2);
    end = (x + w // 2, y + h // 2);
    cv2.rectangle(image, start, end, (255, 0, 0), 2);

cv2.imshow("Face Count", image);
cv2.waitKey(0);
cv2.destroyAllWindows();