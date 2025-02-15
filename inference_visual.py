import os
import cv2
import requests

# Retrieve API key from environment variable
API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Fallback for debugging
print("Using API Key:", API_KEY)  # Debug print

MODEL_ID = "face-recognition-tw2ab/4"  # Verify this is correct on your dashboard
ENDPOINT = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}"
IMAGE_PATH = "E:/face-recognition-zip/c.jpg"  # Replace with your image file

def detect_faces(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(ENDPOINT, files={"file": image_file})
    return response.json()

def visualize_predictions(image_path, predictions):
    image = cv2.imread(image_path)
    for pred in predictions:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])
        left = x - w // 2
        top = y - h // 2
        right = x + w // 2
        bottom = y + h // 2
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{pred['class']} {round(pred['confidence'] * 100, 1)}%"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    result = detect_faces(IMAGE_PATH)
    print("Inference result:", result)
    predictions = result.get("predictions", [])
    annotated_image = visualize_predictions(IMAGE_PATH, predictions)
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
