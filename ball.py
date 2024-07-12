import numpy as np
import cv2

# Read the images
img1 = cv2.imread('./img.jpg')
img2 = cv2.imread('./image.jpeg')

cap = cv2.VideoCapture(0)
while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # range of yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    # range of blue color
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # create masks
    mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    # Contour calculation
    for contour in contours:
        if len(contour) > 0:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 3)
    
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.add(mask1,mask2)
    res = cv2.bitwise_not(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()