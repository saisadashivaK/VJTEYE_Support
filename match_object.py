import cv2
import numpy as np
import os


def match(frame, ref_img, box_img, threshold):
    flag = False
    
    box_img_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.resize(ref_img_gray, (box_img.shape[1], box_img.shape[0]))
    res = cv2.matchTemplate(box_img_gray, ref_img_gray, cv2.TM_SQDIFF_NORMED)
    ##cv2.imshow("EXP", res)
    ##cv2.waitKey(0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if min_val < threshold:
        cv2.rectangle(frame, (box_img[0], box_img[1]), (box_img[0] + w, box_img[1] + h), (0, 255, 255))
        flag = True
    return frame, flag
        
def denormalize_coordinates(x1, x2, y1, y2, img):
    x1 = x1*img.shape[0]
    x2 = x2*img.shape[0]
    y1 = y1*img.shape[1]
    y2 = y2*img.shape[1]
    return x1, x2, y1, y2

def detect_signboard(output_dict, min_threshold_value, frame, dir_name):
    keys = np.flipud(output_dict['detection_scores'].argsort())
    print(keys)
    detection_boxes = output_dict['detection_boxes'][keys]
    
    detection_classes = output_dict['detection_classes'][keys]
    detection_scores = output_dict['detection_scores'][keys]
    ##print(detection_scores)
    detection_scores = detection_scores[detection_scores > min_threshold_value]
    detection_boxes = detection_boxes[0:len(detection_scores)]
    detection_classes = detection_classes[0:len(detection_scores)]
    if frame is not None:
        print("frame exists")
    
    
    
    # tree type structure for searching
    # template matching - slide the picture over entire bounding box to eliminate false positives and match particular objects
    index = 0
    print(detection_boxes.shape)
    for y1, x1, y2, x2 in detection_boxes:
        x1, x2, y1, y2 = denormalize_coordinates(x1, x2, y1, y2, frame)
        print([x1, y1, x2, y2])
        isDetected = False
        for ref_img_filename in os.listdir(dir_name + '/' + str(detection_classes[index])):
             box_img = frame[int(x1) : int(x2), int(y1) : int(y2)]
             
             ref_img = cv2.imread(dir_name + '/' + str(detection_classes[index]) + '/' + ref_img_filename, 1)
             print(dir_name + '/' + str(detection_classes[index]) + '/' + ref_img_filename)
             cv2.imshow("Test", box_img)
             cv2.imshow("Test1", ref_img)
             cv2.waitKey(0)
             final_frame, flag = match(frame, ref_img, box_img, min_threshold_value)

             index += 1
             
             if flag is True:
                    isDetected = True
                    break
        if isDetected is True:
             break
        else:
            detected = True
            for detection_class in os.listdir(dir_name):
                for ref_img_filename in os.listdir(dir_name + '/' + detection_class):
                    box_img = frame[int(x1) : int(x2), int(y1) : int(y2)]
                    ref_img = cv2.imread(dir_name + '/' + detection_class + '/' + ref_img_filename, 1)
                    final_frame, flag = match(frame, ref_img, box_img, min_threshold_value)
                    
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
                    
        
           
         
        
            
            
                
                
           
     
    
            
            
        
        
    
     
    
    
      

    
