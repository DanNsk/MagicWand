#!/usr/bin/python


import sys
import os
import numpy as np
import cv2
import math
import imutils
import datetime
import platform

from imutils import contours
from skimage import measure
from io import BytesIO
from time import sleep
from numpy import linalg
from collections import deque



from picamera.array import PiRGBArray
from picamera import PiCamera

cam =  PiCamera(resolution=(640, 480), framerate=24, sensor_mode=6)
cam.exposure_mode = 'fixedfps'
rawcam = PiRGBArray(cam)
sleep(0.1)

mindist = 5.0
mindistactive = 15.0
directions = 8
commandkeys = ['4321076', '0123456', '1360']
commands = ['Woot!' , 'Woot1!', 'W00t1']


counter = 0
counter0 = 0

points = deque([])

def ClosestPointOnLine(a, b, p):
    ap = p-a
    ab = b-a
    result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result



def CaptureFrame():
    ds = datetime.datetime.now()

    try:
        rawcam.truncate(0)
        cam.capture(rawcam, "bgr")
        frame = rawcam.array

        if (not(frame is None)):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("1.jpg", frame)
    except KeyboardInterrupt:
        raise
    except:
        return None

    #print ((datetime.datetime.now() - ds).microseconds)


    frame = cv2.flip(frame,0)

    return frame


def Scan():
    try:
        cntrsp = np.array([])
        while(1):
            cntrs = FindNewPoints()


            if (not(cntrs is None) and cntrs.any()):
                cntrsp = ProcessNewPoints(cntrsp, cntrs)

                ProcessPointsToGestures()

                CleanupPointsToGestures(cntrs[np.where(~np.isnan(cntrs[:, 5]))][:, 5].astype(int))
            else:
                cntrsp = np.array([])
                CleanupPointsToGestures(cntrsp)


    except KeyboardInterrupt:
        pass
#    except:
#        e = sys.exc_info()[1]
#        print("Error: %s" % e )
    End()
    exit

def RecognizeCommand(cmd):
    for i,k in enumerate(commandkeys):
        if (k in cmd):
            return commands[i]

    return None

def RunCommand(recognizedcomand):
    print (recognizedcomand)

def ProcessPointsToGestures():
    global points, mindist, directions
    for i, pts in enumerate(points):
        if (len(pts) < 2 or linalg.norm(pts[0]) < mindist):
            continue

        cmd = ""
        ccmd = None

        pts = np.array(pts)
        (mag, ang) = cv2.cartToPolar(-pts[:,1], pts[:,0], angleInDegrees=True)
        portion = (360.0/directions)

        ang = (((ang + portion / 2) / portion).astype(int) % directions).astype(str)

        for j, c in enumerate(ang):
            if (ccmd == c):
                continue
            ccmd = c
            cmd = cmd + c[0]

        print (i, cmd)

        recognizedcmd = RecognizeCommand(cmd)

        if (not(recognizedcmd is None)):
            points[i] = [[0.0, 0.0]] #so if we found some command we clean the array
            RunCommand(recognizedcmd)



def CleanupPointsToGestures(cntrs):
    global points, counter0

    tmp = counter0

    for i in range(0, 0 if not(cntrs.any()) else np.amin(cntrs) - tmp - 1):
        counter0 = counter0 + 1
        points.popleft()



def ProcessNewPoints(cntrs0, cntrs1):
    global points, counter0, counter, mindistactive


    if (cntrs0.any()):
        dists = []

        for j0, (cX0, cY0, r0, u0, v0) in enumerate(cntrs0[:,:5]):
            for j1, (cX1, cY1, r1, u1, v1) in enumerate(cntrs1[:,:5]):
                p0 = np.array([cX0, cY0, r0, u0, v0])
                p1 = np.array([cX1, cY1, r1, cX1 - cX0, cY1 - cY0])

                dists.append([j0, j1, int(linalg.norm(p0 - p1) * 1000)]);


        dists = np.array(dists)
        dists = dists[np.argsort(dists[:,2], 0)]
        dists = np.delete(dists, np.where(dists[:,2] > 250000), 0)


        distsres = []

        while(dists.any()):
            d0 = dists[0]

            distsres.append(d0)

            dists = np.delete(dists, 0, 0)
            dists = np.delete(dists, np.where(dists[:,0] == d0[0]), 0)
            dists = np.delete(dists, np.where(dists[:,1] == d0[1]), 0)

        for i, c in enumerate(distsres):
            old = cntrs0[c[0]]
            new = cntrs1[c[1]]

            new[3] = new[0] - old[0]
            new[4] = new[1] - old[1]

            if (math.isnan(old[5]) and c[2] > 1000): #at least one pixel to consider it moved
                counter = counter + 1
                points.append([[0.0, 0.0]])

                old[5] = counter

            if (not(math.isnan(old[5]))):
                new[5] = old[5]

                pointarr = points[int(new[5]) - counter0 - 1]

                if ((linalg.norm(pointarr[-1]) < mindistactive)):
                    pointarr[-1][0] = pointarr[-1][0] + new[3]
                    pointarr[-1][1] = pointarr[-1][1] + new[4]
                else:
                    pointarr.append([new[3], new[4]])

    return cntrs1

def FindNewPoints():
    frame = CaptureFrame()

    if (frame is None):
        return None

    frame = cv2.threshold(frame, 180, 255, cv2.THRESH_BINARY)[1]

    labels = measure.label(frame, connectivity=2, background=0)

    contrs = []

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(frame.shape, dtype="uint8")

        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if (numPixels < 150):
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            c = contours.sort_contours(cnts)[0][0]

            ((cX, cY), radius) = cv2.minEnclosingCircle(c)

            if (radius > 1.1):
                contrs.append([cX, cY, radius, 0, 0, None])

    return  np.array(contrs, np.float32)



def End():
    cam.close()

Scan()
