import cv2
import numpy as np


def detect_signboard(output_dict, min_threshold_value):
    keys = np.flipud(output_dict['detection_scores'].argsort)
    detection_boxes = output_dict['detection_boxes'][keys]
    detection_scores = detection_scores[keys]
    detection_scores = detection_scores[detection_scores > min_threshold_value]
    detection_boxes = detection_boxes[0:len(detection_scores)]
    # tree type structure for searching
    # template matching - slide the picture over entire bounding box to eliminate false positives and match particular objects
    for x, y, w, h in detection_boxes:
        
    
     
    
    
      

    
