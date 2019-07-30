import cv2
import numpy as np


def match(frame, ref_img, box_img, threshold):
    flag = False
    box_img_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(box_img_gray, ref_img_gray, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if min_val < threshold:
        cv2.rectangle(frame, (box_img[0], box_img[1]), (box_img[0] + w, box_img[1] + h), (0, 255, 255))
        flag = True
    return frame, flag
        


def detect_signboard(output_dict, min_threshold_value, frame, dir_name):
    keys = np.flipud(output_dict['detection_scores'].argsort)
    detection_boxes = output_dict['detection_boxes'][keys]
    detection_classes = output_dict['detection_classes'][keys]
    detection_scores = detection_scores[keys]
    detection_scores = detection_scores[detection_scores > min_threshold_value]
    detection_boxes = detection_boxes[0:len(detection_scores)]
    detection_classes = detection_classes[0:len(detection_scores)]
    
    # tree type structure for searching
    # template matching - slide the picture over entire bounding box to eliminate false positives and match particular objects
    index = 0
    for x, y, w, h in detection_boxes:
        isDetected = False
        for ref_img_filename in os.listdir(dir_name + '/' + detection_classes[index]):
             box_img = frame[x : x + w, y : y + h]
             ref_img = cv2.imread(dir_name + '/' + detection_classes[index] + '/' + ref_img_filename, 1)
             final_frame, flag = match(frame)
             index += 1
             if flag is True:
                    isDetected = True
                    break
        if isDetected is True:
            break
        else:
            detected = True
            for class in os.listdir(dir_name):
                for ref_img_filename in os.listdir(dir_name + '/' + class):
                    box_img = frame[x : x + w, y : y + h]
                    ref_img = cv2.imread(dir_name + '/' + class + '/' + ref_img_filename, 1)
                    final_frame, flag = match(frame)
                    index += 1
                    if flag is True:
                        isDetected = True
                        break
                if isDetected is True:
                    break
                else:
                    detected = False
            if detected is False:
                print("Not found anything")
            return final_frame
                    
        
           
         
        
            
            
                
                
           
     
    
            
            
        
        
    
     
    
    
      

    
