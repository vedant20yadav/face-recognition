import base64
import os
import requests
import json
import cv2
import numpy as np

# Replace with your details
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found! Set ROBOFLOW_API_KEY as an environment variable.")

MODEL_ID = "face-recognition-tw2ab/4"
ENDPOINT = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}"
IMAGE_PATH = "E:/face-recognition-zip/a.jpg"

# Read and encode the image in base64
with open(IMAGE_PATH, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Send the POST request with the base64-encoded image as data
headers = {"Content-Type": "application/x-www-form-urlencoded"}
response = requests.post(ENDPOINT, data=encoded_image, headers=headers)

# Parse JSON response
result = response.json()
print(result)

# Load the image using OpenCV
image = cv2.imread(IMAGE_PATH)

# Check if the response contains predictions
if "predictions" in result:
    for pred in result["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        confidence = pred["confidence"]
        label = pred["class"]

        # Calculate bounding box coordinates (convert from center format)
        top_left = (x - w // 2, y - h // 2)
        bottom_right = (x + w // 2, y + h // 2)

        # Draw the bounding box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the annotated image
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No faces detected.")
