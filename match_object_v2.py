import cv2
import numpy as np
import os
import math


label_dict = {1 : "brownboard",\
              2 : "layout_board1",\
              3 : "non-blueboard", \
              4 : "rectangle_blueboard1", \
              5 : "square_blueboard", \
              6 : "rectangle_blueboard2", \
              7 : "bigboard", \
              8 : "layout_board2"}
weight_dict = {"aspect_ratio" : 0.7, "template_match" : 0.85, "histogram_match" : 0.6}
# error function to be used is
def swap(a, b):
    a = a + b
    b = a - b
    a = a - b
    return a, b
    
def err_fn(v1, v2):
    return math.sqrt((v1 - v2)**2)
    
        
def aspect_ratio(coordinates):
    x1, y1, x2, y2 = coordinates
    try:
        return abs((x2 - x1)/(y2 - y1))
    except ZeroDivisionError:
        print("ERROR : In aspect_ratio() :- Bounding box coordinates are incorrect")
        return False


def isContained(x, y, box):
    y1, x1, y2, x2 = box
    if x1 > x2:
        x1, x2 = swap(x1, x2)
    if y1 > y2:
        y1, y2 = swap(y1, y2)
    try:
        if x in range(x1, x2) and y in range(y1, y2):
            return True
        else:
            return False
        
    except Exception as e:
        print("In isContained() :- ", e)
        return -1


def zero_check(num):
    if num == 0.0:
        return 0.1**3

def match(frame, ref_img, box_img, threshold, box, matchHistogram=False, histThresh=0.5):
    flag = false
    scores_dict = {}
    try:
        aspect_ratio_error = err_fn(aspect_ratio([0, 0, ref_img.shape[1], ref_img.shape[0]]), aspect_ratio(box))
        if aspect_ratio_error < 0.6:
            scores_dict.update({"aspect_ratio" : (1/zero_check(aspect_ratio_error))*weight_dict["aspect_ratio"]})
        else:
            scores_dict.update({"aspect_ratio" : (1/aspect_ratio_error)*(1 - weight_dict["aspect_ratio"])})
        def feature_and_template_match():
            orb = cv2.ORB_Create()
            kp1, des1 = orb.detectAndCompute(ref_img, None)
            kp2, des2 = orb.detectAndCompute(frame, None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            def filter_matches(kp1, kp2, matches, box, ratio=0.75):
                mkp1, mkp2 = [], []
                for m in matches:
                    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                        m = m[0]
                        mkp1.append(kp1[m.queryIdx])
                        mkp2.append(kp2[m.trainIdx])
                p1 = np.float32([kp.pt for kp in mkp1 if isContains(kp.pt[0], kp.pt[1], box) is True])
                p2 = np.float32([kp.pt for kp in mkp2 if isContains(kp.pt[0], kp.pt[1], box) is True])
                kp_pairs = zip(mkp1, mkp2)
                return p1, p2, list(kp_pairs)
            p_ref, p_test, kp_pairs = filter_matches(kp1, kp2, matches)
            M = cv.getPerspectiveTransform(p_test,p_ref)
            final_template = cv.warpPerspective(box_img, M, tuple(box_img.shape[:2]))
            final_template_grey = cv2.cvtColor(final_template, cv2.COLOR_BGR2GRAY)
            box_img_grey = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(box_img_grey, final_template_gray, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < threshold:
                scores_dict.update({"template_match" : (1/zero_check(min_val))*weight_dict["template_match"]})
            else:
                scores_dict.update({"template_match" : (1/min_val)*(1 - weight_dict["template_match"])})
        feature_and_template_match()
        if matchHistogram is true:
          hist1 = cv2.calcHist([final_template_grey],[0],None,[256],[0,256])
          hist2 = cv2.calcHist([box_img_grey],[0],None,[256],[0,256])
          res = cv2.compareHist(hist1, hist2, 3)
          if res < histThresh
            scores_dict.update({"histogram_match : (1/zero_check(res))"})
        
                
    

def main():
    box = [25, 26, 24, 19]
    print("Is the point contained in the given region? ", isContained(25.8, 26.9, box))
##    print("Test error function for 5.5, 5.7 ", err_fn(5.5, 5.7))
##    print("Test error function for 0.5, 0.7 ", err_fn(0.5, 0.7))
    print("Test for aspect_ratio x1 = 200, y1 = 100, x2 = 150, y2 = 100 ", aspect_ratio([200, 100, 150, 100]))
main()
