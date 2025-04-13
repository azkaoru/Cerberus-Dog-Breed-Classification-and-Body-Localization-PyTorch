# Dog Body Part Detection Web API

This Web API extends the Cerberus Dog Breed Classification project by providing HTTP endpoints for the dog body part detection functionality.

## Setup and Installation

1. Make sure you have all the dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Run the web API:
   ```
   python dog_eye_webapi.py
   ```

   The server will start on `http://0.0.0.0:5000`.

## API Endpoints

### 1. Detect Eyes

**Endpoint:** `/detect_eyes`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `image`: The image file containing a dog (JPG, PNG)

**Response:**
```json
{
  "status": "success",
  "eyes_detected": [
    {
      "x1": 100,
      "y1": 120,
      "x2": 150,
      "y2": 170,
      "confidence": 0.92
    },
    {
      "x1": 200,
      "y1": 120,
      "x2": 250,
      "y2": 170,
      "confidence": 0.89
    }
  ]
}
```

### 2. Detect Objects (All Body Parts)

**Endpoint:** `/detect_objects`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `image`: The image file containing a dog (JPG, PNG)

**Response:**
- A JPEG image with detected body parts outlined and labeled (exactly the same output as the GUI)

### 3. Make Eye Wink

**Endpoint:** `/make_eye_wink`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `image`: The image file containing a dog (JPG, PNG)
- `eye_index`: (Optional) The index of the eye to wink (default: 0, which is typically the first detected eye)

**Response:**
- A JPEG image with the selected eye filled with white to create a winking effect

## Example Usage

### Using the Provided Example Scripts

1. Detect all objects in an image:
   ```bash
   python detect_objects_example.py path/to/your/dog_image.jpg
   ```

2. Make a dog wink:
   ```bash
   python dog_eye_wink_example.py path/to/your/dog_image.jpg [eye_index]
   ```

### Using cURL

1. Detect eyes in an image:
   ```bash
   curl -X POST -F "image=@path/to/your/dog_image.jpg" http://localhost:5000/detect_eyes
   ```

2. Detect all body parts (eyes, nose, tail):
   ```bash
   curl -X POST -F "image=@path/to/your/dog_image.jpg" http://localhost:5000/detect_objects --output detected_dog.jpg
   ```

3. Make a dog wink:
   ```bash
   curl -X POST -F "image=@path/to/your/dog_image.jpg" -F "eye_index=0" http://localhost:5000/make_eye_wink --output winking_dog.jpg
   ```

### Using Python with requests

```python
import requests

# Detect all body parts
files = {'image': open('path/to/your/dog_image.jpg', 'rb')}
response = requests.post('http://localhost:5000/detect_objects', files=files)

# Save the detected image
with open('detected_dog.jpg', 'wb') as f:
    f.write(response.content)
```

## Note on Model Accuracy

The detection accuracy depends on the quality of the input image and how well the dog is visible. For best results:

1. Use images with the dog clearly visible
2. Ensure good lighting conditions
3. Provide unobstructed views of the dog

The model is trained to detect dog body parts (eyes, nose, tail) and matches the performance of the original GUI application.