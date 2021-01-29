import cv2
import numpy as np
from classshape import ShapeDetector
from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial import distance as dist


class Color:

    def color_detection(new):
        #img = new.copy()
        hsv = cv2.cvtColor(new,cv2.COLOR_BGR2HSV)
        sd = ShapeDetector()
        kernal = np.ones((5, 5), "uint8")

        #.................red.................
        red_lower = np.array([136,87,111],np.uint8)
        red_upper = np.array([180,255,255],np.uint8)
        frame_threshed_red = cv2.inRange(hsv, red_lower, red_upper)
        red = cv2.dilate(frame_threshed_red, kernal)
        res_red = cv2.bitwise_and(img, img, mask=frame_threshed_red)


        cnts = cv2.findContours(red.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
            shape = sd.detect(c)
            area1 = cv2.contourArea(c)

            if area1>=1000:
                c = c.astype("int")
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                #cv2.imshow('red', img)
                color = "red"
                return color
    
    
        #.................blue.................
        blue_lower = np.array([99,115,150],np.uint8)
        blue_upper = np.array([110,255,255],np.uint8)
        frame_threshed_blue = cv2.inRange(hsv, blue_lower, blue_upper)
        blue = cv2.dilate(frame_threshed_blue, kernal)
        res_blue = cv2.bitwise_and(img, img, mask=frame_threshed_blue)

        cnts = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        for c in cnts:
            shape = sd.detect(c)
            area1 = cv2.contourArea(c)

            if area1>=1000:
                c = c.astype("int")
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                #cv2.imshow('blue', img)
                color = "blue"
                return color
    
        #.................yellow.................
        yellow_lower = np.array([22,60,200],np.uint8)
        yellow_upper = np.array([60,255,255],np.uint8)
        frame_threshed_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow = cv2.dilate(frame_threshed_yellow, kernal)
        res_yellow = cv2.bitwise_and(img, img, mask=frame_threshed_yellow)

        cnts = cv2.findContours(yellow.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        for c in cnts:
            shape = sd.detect(c)
            area1 = cv2.contourArea(c)

            if area1>=1000:
                c = c.astype("int")
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                #cv2.imshow('yellow', img)
                color = yellow
                return color
    
        #.................white.................
        white_lower = np.array([0,0,200],np.uint8)
        white_upper = np.array([180,20,255],np.uint8)
        frame_threshed_white = cv2.inRange(hsv, white_lower, white_upper)
        white = cv2.dilate(frame_threshed_white, kernal)
        res_white = cv2.bitwise_and(img, img, mask=frame_threshed_white)

        cnts = cv2.findContours(white.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        for c in cnts:
            shape = sd.detect(c)
            area1 = cv2.contourArea(c)

            if area1>=1000:
                c = c.astype("int")
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                #cv2.imshow('white', img)
                color = "white"
                return color
    
        #.................black.................
        black_lower = np.array([0,0,0],np.uint8)
        black_upper = np.array([180,255,30],np.uint8)
        frame_threshed_black = cv2.inRange(hsv, black_lower, black_upper)
        black = cv2.dilate(frame_threshed_black, kernal)
        res_black = cv2.bitwise_and(img, img, mask=frame_threshed_black)

        cnts = cv2.findContours(black.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        for c in cnts:
            shape = sd.detect(c)
            area1 = cv2.contourArea(c)

            if area1>=1000:
                c = c.astype("int")
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                #cv2.imshow('black', img)
                color = "black"
                return color
    
        #.................orange.................
        orange_lower = np.array([5, 50, 50],np.uint8)
        orange_upper = np.array([15, 255, 255],np.uint8)
        frame_threshed_orange = cv2.inRange(hsv, orange_lower, orange_upper )
        orange = cv2.dilate(frame_threshed_orange, kernal)
        res_orange = cv2.bitwise_and(img, img, mask=frame_threshed_orange)

        cnts = cv2.findContours(orange.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        for c in cnts:
            shape = sd.detect(c)
            area1 = cv2.contourArea(c)

            if area1>=1000:

                c = c.astype("int")
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                #cv2.imshow('orange', img)
                color = "orange"
                return color


        cv2.waitKey()
        cv2.destroyAllWindows()
'''
#cap = cv2.VideoCapture(0)
img = cv2.imread('color.png')
#_, img = cap.read()
cv2.imshow('image1', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=5)
edged = cv2.erode(edged, None, iterations=5)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
    shape = sd.detect(c)
    area = cv2.contourArea(c)

    if area>1000:
        print(area)
        x,y,w,h = cv2.boundingRect(c)
        new_img=img[y-1:y+h+1,x-1:x+w+1]
        cv2.imshow('new image',new_img)
        color = color_detection(new_img,area)
        print(color,shape)
        cv2.waitKey(0)
'''


