from roboflow import Roboflow
from dotenv import load_dotenv
import os
import cv2

load_dotenv()

# Initialize Roboflow
rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
project = rf.workspace(os.getenv('ROBOFLOW_WORKSPACE')).project(os.getenv('ROBOFLOW_PROJECT'))
model = project.version(int(os.getenv('ROBOFLOW_VERSION'))).model

def detect_trash(image_path):
    """
    Detect trash using Roboflow API
    
    Args:
        image_path: Path to image file or numpy array
    
    Returns:
        dict with detection results
    """
    # If it's a numpy array (from webcam), save temporarily
    if isinstance(image_path, str):
        result = model.predict(image_path, confidence=40, overlap=30).json()
    else:
        # Save numpy array temporarily
        cv2.imwrite('temp.jpg', image_path)
        result = model.predict('temp.jpg', confidence=40, overlap=30).json()
        os.remove('temp.jpg')
    
    # Parse results
    detections = []
    for prediction in result['predictions']:
        detections.append({
            'class': prediction['class'],
            'confidence': prediction['confidence'],
            'bbox': [
                prediction['x'],
                prediction['y'],
                prediction['width'],
                prediction['height']
            ]
        })
    
    return {
        'classes': [d['class'] for d in detections],
        'count': len(detections),
        'detections': detections,
        'image_width': result['image']['width'],
        'image_height': result['image']['height']
    }

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    for det in detections:
        x, y, w, h = det['bbox']
        
        # Convert center coordinates to corner coordinates
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{det['class']}: {det['confidence']:.2f}"
        cv2.putText(image, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image