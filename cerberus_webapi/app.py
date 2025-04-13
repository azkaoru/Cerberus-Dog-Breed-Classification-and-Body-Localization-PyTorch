import os
import sys
import io
import cv2
import numpy as np
import torch
import tempfile
import subprocess
import uuid
import shutil
from flask import Flask, request, jsonify, send_file
from pathlib import Path
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Get the absolute path to the project root
FILE = Path(__file__).resolve()
WEBAPI_ROOT = FILE.parents[0]  # cerberus_webapi directory
PROJECT_ROOT = WEBAPI_ROOT.parents[0]  # Main project directory

# Set up paths to YOLOv5 and model
DETECTION_ROOT = PROJECT_ROOT / 'Object-detection'
YOLOV5_ROOT = DETECTION_ROOT / 'yolov5'

# Add YOLOv5 to Python path
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

# Fix Python module resolution by modifying sys.path
# Make sure utils from YOLOv5 is imported, not any other utils
original_path = list(sys.path)
sys.path.insert(0, str(YOLOV5_ROOT))

# Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Restore original path
sys.path = original_path
sys.path.append(str(YOLOV5_ROOT))

# Define paths for detection (same as in Object-detection/main.py)
weights_path = os.path.join(DETECTION_ROOT, 'Cerberus', 'training_2', 'weights', 'best.pt')
device = select_device('')  # Use empty string for auto-selection (CPU if no GPU)
model = DetectMultiBackend(weights_path, device=device)
stride, names = model.stride, model.names
imgsz = check_img_size((640, 640), s=stride)  # check image size
conf_thres = 0.5  # Same confidence as in Object-detection/main.py
data_yaml = os.path.join(YOLOV5_ROOT, 'Cerberus-12', 'data.yaml')

@app.route('/')
def index():
    """API home page"""
    return jsonify({
        "name": "Cerberus Dog Body Part Detection API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/detect_eyes", "method": "POST", "description": "Detect dog eyes and return coordinates"},
            {"path": "/detect_objects", "method": "POST", "description": "Detect all dog body parts and return annotated image"},
            {"path": "/make_eye_wink", "method": "POST", "description": "Make a dog wink by filling one eye with white"}
        ]
    })

@app.route('/detect_eyes', methods=['POST'])
def detect_eyes():
    """Detect dog eyes and return their coordinates"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process image for detection
    img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    
    # Inference
    pred = model(img)
    
    # NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[1])  # Only detect eyes (class index 1)
    
    eyes_detected = []
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
            
            # Extract eye detections
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                eyes_detected.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': float(conf),
                })
    
    return jsonify({
        'status': 'success',
        'eyes_detected': eyes_detected
    })

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    """Detect dog body parts using the exact same method as the GUI"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    # Get uploaded file
    file = request.files['image']
    
    # Create a temporary file to save the uploaded image
    temp_dir = tempfile.mkdtemp()
    
    # Save the uploaded image to a temporary file
    temp_image_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
    file.save(temp_image_path)
    
    # Set up paths exactly like the GUI does
    detect_script = os.path.join(YOLOV5_ROOT, 'detect.py')
    
    # Run the detection script with the same parameters as the GUI
    command = [
        'python', detect_script,
        '--weights', weights_path,
        '--source', temp_image_path,
        '--data', data_yaml,
        '--conf', str(conf_thres),  # Use the same confidence threshold as in main.py
    ]
    
    try:
        # Run the detection
        subprocess.run(command, check=True)
        
        # Find the detection results
        output_directory = os.path.join(YOLOV5_ROOT, 'runs', 'detect')
        latest_directory = max(
            [os.path.join(output_directory, d) for d in os.listdir(output_directory)],
            key=os.path.getmtime
        )
        
        # Get the path to the output image within the latest directory
        output_images = [f for f in os.listdir(latest_directory)]
        if len(output_images) == 0:
            return jsonify({'error': 'No detection output found'}), 500
            
        detected_image_path = os.path.join(latest_directory, output_images[0])
        
        # Read the detected image with bounding boxes
        detected_image = cv2.imread(detected_image_path)
        if detected_image is None:
            return jsonify({'error': 'Failed to read detected image'}), 500
        
        # Convert the image to bytes for response
        _, buffer = cv2.imencode(".jpg", detected_image)
        img_bytes = io.BytesIO(buffer)
        img_bytes.seek(0)
        
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {e}")
        
        # Return the image with detections
        return send_file(img_bytes, mimetype='image/jpeg')
        
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/make_eye_wink', methods=['POST'])
def make_eye_wink():
    """Make a dog wink by filling one eye with white"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    # Get params
    file = request.files['image']
    eye_index = request.form.get('eye_index', '0')  # Default to first eye if not specified
    try:
        eye_index = int(eye_index)
    except ValueError:
        return jsonify({'error': 'Invalid eye_index parameter'}), 400
    
    # Create a temporary file to save the uploaded image
    temp_dir = tempfile.mkdtemp()
    
    # Save the uploaded image to a temporary file
    temp_image_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
    file.save(temp_image_path)
    
    # Set up paths exactly like the GUI does
    detect_script = os.path.join(YOLOV5_ROOT, 'detect.py')
    
    # Run the detection script with the same parameters as the GUI
    command = [
        'python', detect_script,
        '--weights', weights_path,
        '--source', temp_image_path,
        '--data', data_yaml,
        '--conf', str(conf_thres),  # Use the same confidence threshold as in main.py
    ]
    
    try:
        # Run the detection
        subprocess.run(command, check=True)
        
        # Find the detection results
        output_directory = os.path.join(YOLOV5_ROOT, 'runs', 'detect')
        latest_directory = max(
            [os.path.join(output_directory, d) for d in os.listdir(output_directory)],
            key=os.path.getmtime
        )
        
        # Get the detected image path
        output_images = [f for f in os.listdir(latest_directory)]
        if len(output_images) == 0:
            return jsonify({'error': 'No detection output found'}), 500
            
        detected_image_path = os.path.join(latest_directory, output_images[0])
        
        # Read the detected image with bounding boxes
        detected_image = cv2.imread(detected_image_path)
        if detected_image is None:
            return jsonify({'error': 'Failed to read detected image'}), 500
        
        # Read the original image in its original dimensions
        original_image = cv2.imread(temp_image_path)
        
        # Create a copy for processing the wink
        wink_image = original_image.copy()
        
        # Process image for detection to get eye coordinates
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # Use the same model and confidence as the GUI
        pred = model(img)
        pred_eyes = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45, classes=[1])  # Only eyes (class 1)
        
        eyes_detected = []
        for i, det in enumerate(pred_eyes):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], original_image.shape).round()
                
                # Extract eye detections
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    eyes_detected.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': float(conf),
                    })
        
        # If no eyes detected, check if any eyes are visible in the detection output
        if not eyes_detected:
            # Try with all classes (just in case eye was detected but not classified correctly)
            pred_all = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45)
            
            for i, det in enumerate(pred_all):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], original_image.shape).round()
                    
                    # Extract all detections
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        
                        # Special case for dog eyes - they're usually in upper half of the image
                        # and relatively small
                        width = x2 - x1
                        height = y2 - y1
                        center_y = (y1 + y2) // 2
                        
                        if (center_y < original_image.shape[0] // 2 and  # In upper half
                            width < original_image.shape[1] // 4 and    # Not too wide
                            height < original_image.shape[0] // 4):     # Not too tall
                            
                            eyes_detected.append({
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'confidence': float(conf),
                            })
        
        # If still no eyes detected, use the fallback with default positions
        if not eyes_detected:
            h, w = original_image.shape[:2]
            # Position eyes in the upper third of the image
            eye_y = h // 4
            eye_size = w // 10
            
            left_eye = {
                'x1': w // 4 - eye_size // 2,
                'y1': eye_y - eye_size // 2,
                'x2': w // 4 + eye_size // 2,
                'y2': eye_y + eye_size // 2
            }
            
            right_eye = {
                'x1': 3 * w // 4 - eye_size // 2,
                'y1': eye_y - eye_size // 2,
                'x2': 3 * w // 4 + eye_size // 2,
                'y2': eye_y + eye_size // 2
            }
            
            eyes_detected = [left_eye, right_eye]
            print("Using fallback eye positions - no eyes detected")
        
        if eye_index >= len(eyes_detected):
            eye_index = 0  # Default to first eye if index is out of range
        
        # Fill the selected eye with white
        eye = eyes_detected[eye_index]
        
        # Use an elliptical mask for more natural eye shape
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        center = ((eye['x1'] + eye['x2']) // 2, (eye['y1'] + eye['y2']) // 2)
        axes = ((eye['x2'] - eye['x1']) // 2, (eye['y2'] - eye['y1']) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Apply the mask to make the eye white
        wink_image[mask > 0] = [255, 255, 255]
        
        # Convert processed image to bytes
        is_success, buffer = cv2.imencode(".jpg", wink_image)
        if not is_success:
            return jsonify({'error': 'Failed to encode the processed image'}), 500
        
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {e}")
        
        # Convert to bytes and serve
        img_bytes = io.BytesIO(buffer)
        img_bytes.seek(0)
        
        return send_file(img_bytes, mimetype='image/jpeg')
        
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)