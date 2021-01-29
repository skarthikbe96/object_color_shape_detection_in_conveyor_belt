import datetime
import imutils
import time
import cv2
import numpy as np
from imutils import contours
from classshape import ShapeDetector
from imutils import perspective
from collections import deque

vs = cv2.VideoCapture(0)
firstFrame = None
tracker = cv2.TrackerCSRT_create()

while True:
        _, frame = vs.read()
        #height_image,width_image = frame.shape[:2]
        h,w = frame.shape[:2]
        #print(h,w)

        center = {}

        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if firstFrame is None:
                firstFrame = gray
                continue
        
        frameDelta = cv2.absdiff(firstFrame, gray)
        
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        for (i,c) in enumerate(cnts):
                #if cv2.contourArea(c) < 500:
                        #continue

                (x, y, w1, h1) = cv2.boundingRect(c)
                
                M = cv2.moments(c)
                
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX,cY = 0,0
                        
                #print("center",cX,cY)
                        
                
                cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
                cv2.rectangle(frame, (x, y), (x + w1, y + h1), (0, 255, 0), 2)
                
                c = c.astype("int")
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

                
                bbox = (x, y, w1, h1)
                ok = tracker.init(frame, bbox)

                ok, bbox = tracker.update(frame)

                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,255,255), 2, 1)
                else :
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord("q"):
                break
 
vs.release()
cv2.destroyAllWindows()
        
