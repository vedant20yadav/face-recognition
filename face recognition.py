import cv2
import matplotlib.pyplot as plt
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="riMiboxeLKguBWpBlc7J")
project = rf.workspace().project("real-time-facial-recognition")
model = project.version(1).model

# Input and output image names
input_image = "a.jpg"
output_image = "output.jpg"

# Perform face recognition on a.jpg
print(model.predict(input_image, confidence=40, overlap=30).json())

# Save the processed image
model.predict(input_image, confidence=40, overlap=30).save(output_image)
print("Face recognition completed. Results saved in output.jpg")

# Show the output image
output = cv2.imread(output_image)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display
plt.imshow(output)
plt.axis("off")  # Hide axis
plt.title("Detected Faces in a.jpg")
plt.show()
