import cv2 
import imutils
import numpy as np 

def edges(image) : 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral_filtered_image = cv2.GaussianBlur(gray,(3,3),0)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 90, 170)

    return edge_detected_image

#----------------------------------------------------------------------------------------------------------

def cont(image) : 

    contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    return contours

def HSV(image) : 

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_sat = np.array([50, 50, 50]) # Minimum saturation and value thresholds
    upper_sat = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_sat, upper_sat)

    return mask 

def parameters(contour) : 

    length = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*length, True)
    # Check for regularity of sides
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / perimeter ** 2

    return length, approx, area, perimeter, circularity