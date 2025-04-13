#!/usr/bin/env python3
"""
Test script for Cerberus Dog Body Part Detection API

This script tests all API endpoints using a test image and saves the results
to view and confirm the API is working correctly.
"""

import requests
import os
import time
import subprocess
import sys
import webbrowser
from pathlib import Path
import shutil

# Constants
API_URL = "http://localhost:5000"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(SCRIPT_DIR, "hanzo_test.jpg")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_results")

# Check if the test image exists
if not os.path.exists(TEST_IMAGE):
    print(f"Error: Test image not found at {TEST_IMAGE}")
    sys.exit(1)

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to check if the API is running
def check_api_running():
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# Function to start the API if it's not running
def start_api():
    if not check_api_running():
        print("API is not running. Starting it now...")
        api_script = os.path.join(SCRIPT_DIR, "app.py")
        
        # Start API in a new process
        process = subprocess.Popen([sys.executable, api_script], 
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        
        # Give it a moment to start
        time.sleep(5)
        
        # Check if it's running
        if check_api_running():
            print("API started successfully.")
            return process
        else:
            print("Failed to start API.")
            return None
    else:
        print("API is already running.")
        return None

# Test the /detect_eyes endpoint
def test_detect_eyes():
    print("\n[Testing /detect_eyes endpoint]")
    url = f"{API_URL}/detect_eyes"
    files = {'image': open(TEST_IMAGE, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Detected {len(data['eyes_detected'])} eyes")
        
        for i, eye in enumerate(data['eyes_detected']):
            print(f"Eye {i}: ({eye['x1']}, {eye['y1']}) - ({eye['x2']}, {eye['y2']}) with confidence {eye['confidence']:.2f}")
        
        # Save the response to a file
        result_path = os.path.join(OUTPUT_DIR, "eyes_detected.json")
        with open(result_path, 'w') as f:
            import json
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {result_path}")
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Test the /detect_objects endpoint
def test_detect_objects():
    print("\n[Testing /detect_objects endpoint]")
    url = f"{API_URL}/detect_objects"
    files = {'image': open(TEST_IMAGE, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result_path = os.path.join(OUTPUT_DIR, "objects_detected.jpg")
        with open(result_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Success! Results saved to {result_path}")
        return result_path
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Test the /make_eye_wink endpoint
def test_make_eye_wink(eye_index=0):
    print(f"\n[Testing /make_eye_wink endpoint with eye_index={eye_index}]")
    url = f"{API_URL}/make_eye_wink"
    files = {'image': open(TEST_IMAGE, 'rb')}
    data = {'eye_index': str(eye_index)}
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result_path = os.path.join(OUTPUT_DIR, f"winking_eye_{eye_index}.jpg")
        with open(result_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Success! Results saved to {result_path}")
        return result_path
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Create an HTML file to display the results
def create_results_html(eye_data, objects_path, wink_paths):
    html_path = os.path.join(OUTPUT_DIR, "results.html")
    
    # Copy the test image to the results directory for display
    test_image_copy = os.path.join(OUTPUT_DIR, "original_test_image.jpg")
    shutil.copy(TEST_IMAGE, test_image_copy)
    
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cerberus API Test Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .section { margin-bottom: 30px; }
                .image-container { display: flex; flex-wrap: wrap; gap: 20px; }
                .image-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
                img { max-width: 400px; max-height: 400px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>Cerberus Dog Body Part Detection API Test Results</h1>
            
            <div class="section">
                <h2>Original Test Image</h2>
                <div class="image-container">
                    <div class="image-box">
                        <img src="original_test_image.jpg" alt="Test Image">
                        <p>Original test image</p>
                    </div>
                </div>
            </div>
        """)
        
        # Eye detection results
        if eye_data:
            f.write("""
            <div class="section">
                <h2>Eye Detection Results</h2>
                <table>
                    <tr>
                        <th>Eye Index</th>
                        <th>Coordinates (x1, y1) - (x2, y2)</th>
                        <th>Confidence</th>
                    </tr>
            """)
            
            for i, eye in enumerate(eye_data['eyes_detected']):
                f.write(f"""
                    <tr>
                        <td>{i}</td>
                        <td>({eye['x1']}, {eye['y1']}) - ({eye['x2']}, {eye['y2']})</td>
                        <td>{eye['confidence']:.2f}</td>
                    </tr>
                """)
            
            f.write("""
                </table>
            </div>
            """)
        
        # Object detection results
        if objects_path:
            objects_filename = os.path.basename(objects_path)
            f.write(f"""
            <div class="section">
                <h2>Object Detection Results</h2>
                <div class="image-container">
                    <div class="image-box">
                        <img src="{objects_filename}" alt="Detected Objects">
                        <p>All detected dog body parts</p>
                    </div>
                </div>
            </div>
            """)
        
        # Winking results
        if wink_paths and any(wink_paths):
            f.write("""
            <div class="section">
                <h2>Eye Winking Results</h2>
                <div class="image-container">
            """)
            
            for i, path in enumerate(wink_paths):
                if path:
                    filename = os.path.basename(path)
                    f.write(f"""
                    <div class="image-box">
                        <img src="{filename}" alt="Winking Eye {i}">
                        <p>Dog with eye {i} winking</p>
                    </div>
                    """)
            
            f.write("""
                </div>
            </div>
            """)
        
        f.write("""
        </body>
        </html>
        """)
    
    return html_path

# Main execution
if __name__ == "__main__":
    api_process = start_api()
    
    try:
        if check_api_running():
            print(f"Using test image: {TEST_IMAGE}")
            print(f"Results will be saved to: {OUTPUT_DIR}")
            
            # Run all tests
            eye_data = test_detect_eyes()
            objects_path = test_detect_objects()
            
            # Test winking with each detected eye
            wink_paths = []
            if eye_data and len(eye_data['eyes_detected']) > 0:
                for i in range(len(eye_data['eyes_detected'])):
                    wink_path = test_make_eye_wink(i)
                    wink_paths.append(wink_path)
            else:
                # If no eyes detected, just test with index 0
                wink_path = test_make_eye_wink(0)
                wink_paths.append(wink_path)
            
            # Create HTML to view results
            html_path = create_results_html(eye_data, objects_path, wink_paths)
            print(f"\nTest completed! View the results at: {html_path}")
            
            # Open the results in a web browser
            webbrowser.open(f"file://{html_path}")
        else:
            print("API is not running. Please start it manually with: python cerberus_webapi/app.py")
    finally:
        # If we started the API, clean it up
        if api_process:
            print("Stopping the API...")
            api_process.terminate()