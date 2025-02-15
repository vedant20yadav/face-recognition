import os
import requests
from roboflow import Roboflow
import cv2

# Get API Key from environment variable
API_KEY = "LhTB2xgn5tVxptXfmyYD"
if not API_KEY:
    raise ValueError("API Key not found! Make sure it's set in environment variables.")

# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("face-recognition-tw2ab")
model = project.version(4).model  # Update version if needed

# Define input/output images
input_image = "Face Recognition/b.jpg"
output_image = "Face Recognition/output.jpg"

# Run prediction
results = model.predict(input_image, confidence=40, overlap=30).json()
print(results)

# Draw bounding boxes on detected faces
image = cv2.imread(input_image)
for pred in results["predictions"]:
    x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
    start = (x - w // 2, y - h // 2)
    end = (x + w // 2, y + h // 2)
    cv2.rectangle(image, start, end, (0, 255, 0), 2)

# Save & show the output
cv2.imwrite(output_image, image)
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
