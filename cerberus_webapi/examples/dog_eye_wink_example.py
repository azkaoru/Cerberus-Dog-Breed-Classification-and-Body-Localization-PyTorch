import requests
import sys
import os

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python dog_eye_wink_example.py <image_path> [eye_index]")
    sys.exit(1)

image_path = sys.argv[1]
eye_index = "0"  # Default to the first eye
if len(sys.argv) > 2:
    eye_index = sys.argv[2]

# API endpoint URL
url = "http://localhost:5000/make_eye_wink"

# First, detect the eyes to show which ones are found
detect_url = "http://localhost:5000/detect_eyes"
files = {'image': open(image_path, 'rb')}
response = requests.post(detect_url, files=files)

if response.status_code == 200:
    data = response.json()
    print(f"Eyes detected: {len(data['eyes_detected'])}")
    for i, eye in enumerate(data['eyes_detected']):
        print(f"Eye {i}: ({eye['x1']}, {eye['y1']}) - ({eye['x2']}, {eye['y2']}) with confidence {eye['confidence']:.2f}")
else:
    print(f"Error detecting eyes: {response.text}")

# Now make the dog wink
print(f"\nMaking dog wink with eye index {eye_index}...")
files = {'image': open(image_path, 'rb')}
data = {'eye_index': eye_index}
response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    # Save the winking image
    output_path = f"winking_dog_eye{eye_index}.jpg"
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Winking image saved to {os.path.abspath(output_path)}")
    
    # Simple message about viewing the result
    print(f"\nResult saved! Compare the original and winking images:")
    print(f"Original: {os.path.abspath(image_path)}")
    print(f"Winking: {os.path.abspath(output_path)}")
else:
    print(f"Error making dog wink: {response.text}")