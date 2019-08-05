import cv2
import numpy as np
import os


# decision_tree: first parameter is aspect_ratio : error in aspect ratio should be below 2%
# second level will contain match template aspect ratio matches
# if aspect ratio does not match:
# find homography matrix using features, transform the image and then do template matching
# compare histograms after template matching


def match(frame, ref_img, box_img, threshold, coordinates):
    flag = False
    
    box_img_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.resize(ref_img_gray, (box_img.shape[1], box_img.shape[0]))
    ref_img_resized = cv2.resize(ref_img, (box_img.shape[1], box_img.shape[0]))

    res = cv2.matchTemplate(box_img_gray, ref_img_gray, cv2.TM_SQDIFF_NORMED)
    ##cv2.imshow("EXP", res)
    ##cv2.waitKey(0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(box_img_gray, mask=None)
    kp2, des2 = orb.detectAndCompute(ref_img_gray, mask=None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
#   matches = sorted(matches, lambda x: x.distance)
    
    print("Matching value  ", min_val)
    if len(matches) > 20: ##and min_val < threshold:
        print(*coordinates)
        pt1 = (int(coordinates[0]), int(coordinates[1]))
        pt2 = (int(coordinates[2]), int(coordinates[3]))
#       cv2.rectangle(frame, pt1, pt2, (0, 255, 255), 9)
        img3 = cv2.drawMatches(box_img_gray,kp1,ref_img_gray,kp2,matches[:20], flags=2, outImg=None)
        cv2.imshow("matching", img3)
        flag = True
    return frame, flag


def denormalize_coordinates(x1, x2, y1, y2, img):
    x1 = x1*img.shape[1]
    x2 = x2*img.shape[1]
    y1 = y1*img.shape[0]
    y2 = y2*img.shape[0]
    return x1, x2, y1, y2


def detect_signboard(output_dict, min_threshold_value, frame, dir_name):

    detection_boxes = output_dict['detection_boxes']
  
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']

    detection_scores = detection_scores[detection_scores > min_threshold_value]
    detection_boxes = detection_boxes[0:len(detection_scores)]
    detection_classes = detection_classes[0:len(detection_scores)]
    if frame is not None:
        print("frame exists")
    
    class_dict = {'1': 'Brownboard', '2': 'Layoutboard1', '3': 'Non-blueboard', '4': 'Rectangle_blueboard1', '5': 'Square_blueboard', '6': 'Rectangle_blueboard2', '7': 'Bigboard', \
                  '8': 'Layoutboard2'}
    
    # tree type structure for searching
    # template matching - slide the picture over entire bounding box to eliminate false positives and match particular objects
##    print(detection_boxes.shape)
    for y1, x1, y2, x2 in detection_boxes:
        x1, x2, y1, y2 = denormalize_coordinates(x1, x2, y1, y2, frame)
        coordinates = [x1, y1, x2, y2]
        isDetected = False
        index = 0

        
        for ref_img_filename in os.listdir(dir_name + '/' + str(detection_classes[index])):
             box_img = frame[int(y1) : int(y2), int(x1) : int(x2)]
             
             ref_img = cv2.imread(dir_name + '/' + str(detection_classes[index]) + '/' + ref_img_filename, 1)
             print(dir_name + '/' + str(detection_classes[index]) + '/' + ref_img_filename)
##             cv2.imshow("Test", box_img)
##             cv2.imshow("Test1", ref_img)
##             cv2.waitKey(0)
             final_frame, flag = match(frame, ref_img, box_img, min_threshold_value, coordinates)

             if flag is True:
                    isDetected = True
                    break
        index += 1

        if isDetected is True:
             print("detected")
        else:
            detected = True
            for detection_class in os.listdir(dir_name):
                for ref_img_filename in os.listdir(dir_name + '/' + detection_class):
                    box_img = frame[int(x1) : int(x2), int(y1) : int(y2)]
                    ref_img = cv2.imread(dir_name + '/' + detection_class + '/' + ref_img_filename, 1)
                    final_frame, flag = match(frame, ref_img, box_img, min_threshold_value, coordinates)
                    
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
                    
        
           
         
        
            
            
                
                
           
     
    
            
            
        
        
    
     
    
    
      

    
