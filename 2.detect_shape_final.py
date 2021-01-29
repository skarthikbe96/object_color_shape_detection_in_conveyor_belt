from scipy.spatial import distance as dist
from classshape import ShapeDetector
from colorlabeler import ColorLabeler
from imutils import perspective
from imutils import contours
from imutils.video import VideoStream
from imutils.video import FPS
from time import gmtime, strftime
from time import sleep
import numpy as np
import argparse  
import imutils
import time
import serial
import cv2
import math


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def order_points_old(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

'''......Angle calcualtion.............................'''
def angle(matrix):
        z1,z2,z3,z4=matrix
        x1,y1=z1
        x2,y2=z2
        if y2>y1:
                slope=(y1-y2)/(x1-x2)
                angle=int(math.degrees(math.atan(slope)))
                tilted = "CW"
        else:
                slope=(y1-y2)/(x1-x2)
                angle=int(math.degrees(math.atan(slope)))
                tilted = "CCW"
        if angle == 0:
                tilted = "ND"
                return angle,tilted
        else:
                return angle,tilted
'''.......................................................'''
        
'''.......IMAGE OR VIDEO CAPTURING......'''
#image = cv2.imread('color.png')
cap = cv2.VideoCapture(0) 
width = 14  #10cm = 183.09 matrics so 1cm = 18 as a cutoff
ser = serial.Serial('/dev/ttyAMA0',baudrate=9600)
'''....................................'''

while(1):
        '''.......GETTING THE FRAME...........'''
        #frame = cap.get(cv2.CAP_PROP_FPS)
        _, image = cap.read()
        
        height_image,width_image = image.shape[:2]

        blurred = cv2.medianBlur(image,5)
        blurred = cv2.bilateralFilter(blurred,9,75,75)

        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        thresh = cv2.Canny(gray, 50, 100)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh = cv2.erode(thresh, None, iterations=5)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


        '''..FINDING THE OBJECT CORDINATES IN THE IMAGE...'''
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        '''...............................................'''
        '''...CLASS FOR DETECTING THE SHAPE...............'''
        sd = ShapeDetector()
        cl = ColorLabeler()
        '''................................................'''

        '''...SHORTING THE COUNTER FOR EASILER CALCULATION...'''
        (cnts, _) = contours.sort_contours(cnts)
        '''......................................'''
        for (i, c) in enumerate(cnts):
                '''..FINDING THE AREA OF THE EACH OBJECT IN THE IMAGE..'''
                area = cv2.contourArea(c)
                if area > 1000 :                       
                        shape = sd.detect(c)
                        color = cl.label(lab, c)
                        '''..FINDING THE CENTER OF THE OBJECT..'''
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                        else:
                                cX,cY = 0,0
                        '''......................................'''
                        
                        cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)

                        c = c.astype("int")
                        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                        if cv2.contourArea(c) < 100:
                                continue

                        '''.....FIND THE FOUR COORDINATE OF THE RECTANGLE...'''
                        
                        box = cv2.minAreaRect(c)
                        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                        box = np.array(box, dtype="int")

                        box = perspective.order_points(box)

                        rect = order_points_old(box)
                        rect = perspective.order_points(box)
                        rec_array=rect.astype("int")
                        ang,tilt=angle(rect.astype("int"))
                        
                        '''.................................................'''

                        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

                        for (x, y) in box:
                                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

                        '''..FINDING THE DISTANCE..................'''
                        
                        (tl, tr, br, bl) = box
                        
                        (tltrX, tltrY) = midpoint(tl, tr)
                        (blbrX, blbrY) = midpoint(bl, br)

                        (tlblX, tlblY) = midpoint(tl, bl)
                        (trbrX, trbrY) = midpoint(tr, br)

                        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                        dimA = "{:.1f}cm".format(dA / width)
                        dimB = "{:.1f}cm".format(dB / width)
                
                        '''...............................................'''
                        
                        cv2.putText(image, dimB,
                                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (255, 255, 255), 2)
                        cv2.putText(image, dimA,
                                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (255, 255, 255), 2)
                        cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
                        #cv2.line(image,(0,int(height_image/2)),(width_image,int(height_image/2)),(255,255,0),1)
                        #cv2.line(image,(int(width_image/2),0),(int(width_image/2),height_image),(255,255,0),1)
                        '''.................................................'''
                        d1 = dist.euclidean((cX,cY),(0,cY))
                        d2 = dist.euclidean((cX,cY),(cX,0))
                        dimA1 = "{:.1f}cm".format(d1 / width)
                        dimB1 = "{:.1f}cm".format(d2 / width)  
                        t = time.strftime("%H:%M:%S")

                        '''..................................................'''
                        if cX!=320:
                                data=[shape,color,t,[dimA,dimB],[cX,cY],ang,tilt,rect.astype("int"),'#']
                                print(data)
                                #print("frame per second",frame)
                                print("....................")
                                '''..................................................'''
                                '''..........Serial communication....................'''

           
                        '''...................................................'''
                        cv2.imshow("Image2", image)
                        #cv2.waitKey(1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break

cv2.destroyAllWindows()
cap.release()

'''
                                ser.write(str.encode('*'))
                                ser.write(str.encode(shape))
                                ser.write(str.encode(','))
                                ser.write(str.encode(color))
                                ser.write(str.encode(','))
                                ser.write(str.encode(t))
                                ser.write(str.encode(','))
                                ser.write(str.encode(dimA1))
                                ser.write(str.encode(','))
                                ser.write(str.encode(dimB1))
                                ser.write(str.encode(','))
                                ser.write(str.encode(str(cX)))
                                ser.write(str.encode(','))
                                ser.write(str.encode(str(cY)))
                                ser.write(str.encode(','))
                                ser.write(str.encode(str(ang)))
                                ser.write(str.encode(','))
                                ser.write(str.encode(tilt))
                                ser.write(str.encode(','))
                                ser.write(str.encode(str(rec_array[0])))
                                ser.write(str.encode(','))
                                ser.write(str.encode(str(rec_array[1])))
                                ser.write(str.encode(','))
                                ser.write(str.encode(str(rec_array[2])))
                                ser.write(str.encode(','))
                                ser.write(str.encode(str(rec_array[3])))
                                ser.write(str.encode('#'))'''

