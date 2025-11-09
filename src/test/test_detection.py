import sys

import os
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import detect_trash, draw_detections

# Test with an image
image_path = "./././images/crushsoda2.jpg"
results = detect_trash(image_path)

print(f"Detected {results['count']} items:")
for cls in results['classes']:
    print(f"  - {cls}")

# Draw and show results
img = cv2.imread(image_path)
img_with_boxes = draw_detections(img, results['detections'])
cv2.imshow('Detection Results', img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()