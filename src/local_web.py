import cv2
import requests
import base64
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Server URL
INFERENCE_URL = "http://localhost:9001"

# Your model details
API_KEY = os.getenv('ROBOFLOW_API_KEY')
MODEL_ID = f"{os.getenv('ROBOFLOW_WORKSPACE')}/{os.getenv('ROBOFLOW_PROJECT')}/{os.getenv('ROBOFLOW_VERSION')}"

def detect_frame(frame):
    """Send frame to local inference server"""
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Make request to local server
    url = f"{INFERENCE_URL}/infer/workflows/{MODEL_ID}"
    
    payload = {
        "api_key": API_KEY,
        "image": {
            "type": "base64",
            "value": img_base64
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def draw_predictions(frame, predictions):
    """Draw bounding boxes on frame"""
    if not predictions:
        return frame
    
    for pred in predictions.get('predictions', []):
        x = int(pred['x'])
        y = int(pred['y'])
        w = int(pred['width'])
        h = int(pred['height'])
        
        # Convert to corners
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{pred['class']}: {pred['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Main loop
cap = cv2.VideoCapture(0)
print("Starting real-time detection... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect every frame (local server is fast!)
    results = detect_frame(frame)
    
    if results:
        frame = draw_predictions(frame, results)
    
    cv2.imshow('Real-time Trash Detection (Local)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()