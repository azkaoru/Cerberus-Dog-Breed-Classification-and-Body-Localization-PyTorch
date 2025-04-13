# Cerberus Dog Body Part Detection API

This project provides a web API for the Cerberus Dog Breed Classification and Body Localization system. It allows detection of dog body parts (eyes, nose, tail) via HTTP endpoints.

## Features

- Detect dog eyes and return their coordinates
- Detect all dog body parts and return an annotated image
- Make a dog "wink" by filling one eye with white

## Setup and Installation

### Prerequisites

- Python 3.8+
- Main Cerberus project with trained models

### Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. This API requires the main Cerberus project with its trained models. The API expects to be in a subdirectory of the main project:
   ```
   Cerberus-project/
   ├── Object-detection/
   │   ├── Cerberus/
   │   ├── yolov5/
   │   └── ...
   ├── cerberus_webapi/
   │   ├── app.py
   │   ├── requirements.txt
   │   └── ...
   └── ...
   ```

### Running the API

Start the web API:
```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`.

## API Endpoints

### 1. API Info

**Endpoint:** `/`  
**Method:** `GET`

Returns basic information about the API and available endpoints.

### 2. Detect Eyes

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

### 3. Detect Objects (All Body Parts)

**Endpoint:** `/detect_objects`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `image`: The image file containing a dog (JPG, PNG)

**Response:**
- A JPEG image with detected body parts outlined and labeled (same as the GUI output)

### 4. Make Eye Wink

**Endpoint:** `/make_eye_wink`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `image`: The image file containing a dog (JPG, PNG)
- `eye_index`: (Optional) The index of the eye to wink (default: 0, which is typically the first detected eye)

**Response:**
- A JPEG image with the selected eye filled with white to create a winking effect

## Example Usage

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

## Docker Support

A Dockerfile is provided to containerize the API:

```bash
# Build the Docker image
docker build -t cerberus-api .

# Run the container
docker run -p 5000:5000 -v /path/to/main/cerberus/project:/app/cerberus cerberus-api
```

## Note on Model Accuracy

The detection accuracy depends on the quality of the input image and how well the dog is visible. For best results:

1. Use images with the dog clearly visible
2. Ensure good lighting conditions
3. Provide unobstructed views of the dog

The model is trained to detect dog body parts (eyes, nose, tail) and matches the performance of the original GUI application.