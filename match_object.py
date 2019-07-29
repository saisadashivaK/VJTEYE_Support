import cv2
import numpy as np


def match(frame, ref_img, box_img, threshold):
    box_img_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(box_img_gray, ref_img_gray, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if min_val < threshold:
        cv2.rectangle(frame, (box_img[0], box_img[1]), (box_img[0] + w, box_img[1] + h), (0, 255, 255))
    return frame
        


def detect_signboard(output_dict, min_threshold_value):
    keys = np.flipud(output_dict['detection_scores'].argsort)
    detection_boxes = output_dict['detection_boxes'][keys]
    detection_scores = detection_scores[keys]
    detection_scores = detection_scores[detection_scores > min_threshold_value]
    detection_boxes = detection_boxes[0:len(detection_scores)]
    # tree type structure for searching
    # template matching - slide the picture over entire bounding box to eliminate false positives and match particular objects
    for x, y, w, h in detection_boxes:
        box_img = frame[x : x + w, y : y + h]
        ref_img = 
        
        
    
     
    
    
      

    
