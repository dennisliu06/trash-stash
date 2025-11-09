from inference import get_model
from dotenv import load_dotenv
import cv2
import os

load_dotenv()

model_id = "wastenet-vvsjj/2"  # Update this!
api_key = os.getenv('ROBOFLOW_API_KEY')

print(f"Loading model: {model_id}")

try:
    model = get_model(model_id=model_id, api_key=api_key)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n🔍 To find your correct model ID:")
    print("1. Go to your Roboflow project")
    print("2. Click 'Deploy' tab")
    print("3. Select 'Python' → Look for the model ID")
    exit(1)

cap = cv2.VideoCapture(0)
print("Starting real-time detection... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference on every frame (much faster!)
    results = model.infer(frame)
    
    # Draw predictions
    for prediction in results[0].predictions:
        x = int(prediction.x)
        y = int(prediction.y)
        w = int(prediction.width)
        h = int(prediction.height)
        
        # Convert center coords to corners
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{prediction.class_name}: {prediction.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Real-time Trash Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()