#!/usr/bin/python3


import sys
import os
import re
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



from pivideostream import PiVideoStream

def ConvFrame(frame):
    if (not(frame is None)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)[1]
    return frame

vs = PiVideoStream(resolution=(640, 480), framerate=12, vflip=True, func = ConvFrame)
vs.camera.exposure_mode = 'fixedfps'
vs = vs.start()

mindist = 7.0 #min distance for initial move
mindistactive = 15.0 #min distance for the move 

directions = 8 #there will be only 8 possible move directions with 0 - straigt up, 4 straight down 2 - right, 6 - left

commands = {
   '[701]{1,2}[345]{1,2}[701]{1,2}[345]{1,2}':'Woot!', #m - like symbol
   '[01]{1,2}[23]{1,2}[45]{1,2}[67]{1,2}':'Woot2!', #circle cw
   '[67]{1,2}[45]{1,2}[23]{1,2}[01]{1,2}':'Woot4!' #circle ccw
}
commandscompiled = dict(zip(commands.keys(), [re.compile(k) for k in commands.keys()]))


counter = 0
counter0 = 0

points = deque([])

frCount = 0

def CaptureFrame():

    try:
        frame = vs.read()

    except KeyboardInterrupt:
        raise
    except:
        #return None
        raise

    return frame


def Scan():
    try:
        cntrsp = np.array([])
        while(1):
            ds = datetime.datetime.now()

            cntrs = FindNewPoints()

            #print ((datetime.datetime.now() - ds).microseconds / 1000) #debug - frame processing time

            if (not(cntrs is None) and cntrs.any()):
                cntrsp = ProcessNewPoints(cntrsp, cntrs)

                ProcessPointsToGestures()

                CleanupPointsToGestures(cntrs[np.where(~np.isnan(cntrs[:, 5]))][:, 5].astype(int))
            else:
                cntrsp = np.array([])
                CleanupPointsToGestures(cntrsp)


    except KeyboardInterrupt:
        pass
    End()
    exit

def RecognizeCommand(cmd):
    for k in commands.keys():
        if (commandscompiled[k].search(cmd)):
            return commands[k]

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

        #print (i, cmd) #print accumulated commands

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


    labels = measure.label(frame, connectivity=2, background=0)

    contrs = []
    uniques = zip(*np.unique(labels, return_counts=True))

    for (label, counts) in uniques:
        if (label == 0 or counts < 5 or counts > 2000):
            continue

        labelMask = np.zeros(frame.shape, dtype="uint8")

        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]

        c = contours.sort_contours(cnts)[0][0]

        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        contrs.append([cX, cY, radius, 0, 0, None])

    return  np.array(contrs, np.float32)



def End():
    vs.stop()

Scan()
