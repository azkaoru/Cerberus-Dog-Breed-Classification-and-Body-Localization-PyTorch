import requests
import sys
import os

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python detect_objects_example.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# API endpoint URL
url = "http://localhost:5000/detect_objects"

# Open the image file and send it to the API
files = {'image': open(image_path, 'rb')}
print(f"Sending image to detection API: {image_path}")
response = requests.post(url, files=files)

if response.status_code == 200:
    # Save the detected image
    output_path = "detected_objects.jpg"
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Detection successful! Image saved to {os.path.abspath(output_path)}")
else:
    print(f"Error detecting objects: {response.text}")