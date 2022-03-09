#!/usr/bin/env python

from argparse import ArgumentParser
from typing import Optional, Tuple
from FreenectPlaybackWrapper.PlaybackWrapper import FreenectPlaybackWrapper

import cv2
import numpy as np
import os
import mahotas
import matplotlib.pyplot as plt

def trainRun():
    s1 = "Set1"
    parser = ArgumentParser(description="OpenCV Demo for Kinect Coursework")
    parser.add_argument("videofolder", help="Folder containing Kinect video. Folder must contain INDEX.txt.",
                        default=s1, nargs="?")
    parser.add_argument("--no-realtime", action="store_true", default=False)

    args = parser.parse_args()

    # Labels for the objects to be detected in the order given.
    labels = ["Baby", "Dog", "Dinosaur", "Coffee Tin", "Mug", "Car", "Camera", "Keyboard", "Koala", 
    "Blackberry", "Diet Coke Bottle", "Duck", "Dragon", "Android"]

    # Keeping a record of the active label and the number of frames passed.
    # c = count, f = frames, tElap = amount of frames that has occured
    c = 0
    f = 1
    tElap = 0

    # Checks if image was seen
    seen = False 

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):

        # Depth data transformed to align with RGB image.
        v, h, a = depth.shape
        N = np.float32([[1, 0, -40], 
                        [0, 1, 30],
                        [0, 0, 1]])
        
        # Depth image transformed.
        tnsl_image = cv2.warpPerspective(depth, N, (v, h))

        # Key information retrieved through adding thresholding to depth data.
        r, depthTest = cv2.threshold(tnsl_image, 80, 255, cv2.THRESH_TOZERO_INV)

        # Searches for edges and contours of depth data given at time.
        borders = cv2.Canny(depthTest, 10, 100)
        lines, ranking = cv2.findContours(borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(lines) > 1:
            cv2.drawContours(rgb, lines, -1, (0, 255, 0), 6)
            cv2.putText(rgb, labels[c], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # saving frames in png file format. 
            if f < 10:
                cv2.imwrite("training\\{}\\{}{}{}training.png".format(labels[c], 0,0,f), depthTest, (50, 50))
            elif f < 100:
                cv2.imwrite("training\\{}\\{}{}training.png".format(labels[c], 0,f), depthTest, (50, 50))
            else:
                cv2.imwrite("training\\{}\\{}training.png".format(labels[c], f), depthTest, (50, 50))
            f += 1
            seen = True
            tElap = 0

        # if the image was seen and 30 frames have passed, go to the next label
        elif seen == True and tElap > 30:
            c += 1
            seen = False
            f = 0

        # count frames passed
            tElap += 1

        # If we have an updated Depth image, then start processing
        if status.updated_depth:

            # Show depth/thresholded depth images
            cv2.imshow("Depth", depth)

        # If we have an updated rgb image, then draw
        if status.updated_rgb:

            # Show RGB image
            cv2.imshow("RGB", rgb)

        # Check for Keyboard input.
        key = cv2.waitKey(5)

        # Break out of the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    return 0

def testRun():
    s2 = "Set2"
    parser = ArgumentParser(description="OpenCV Demo for Kinect Coursework")
    parser.add_argument("videofolder", help="Folder containing Kinect video. Folder must contain INDEX.txt.",
                        default=s2, nargs="?")
    parser.add_argument("--no-realtime", action="store_true", default=False)
    
    args = parser.parse_args()

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):

        # If we have an updated Depth image, then start processing
        if status.updated_depth:

            # Show depth/thresholded depth images
            cv2.imshow("Depth", depth)

        # If we have an updated rgb image, then draw
        if status.updated_rgb:

            # Show RGB image
            cv2.imshow("RGB", rgb)

        # Check for Keyboard input.
        key = cv2.waitKey(5)

        # Break out of the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    return 0

# defining the function HU Moments
def hummts(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feat = cv2.HuMoments(cv2.moments(img)).flatten()
    return feat 

# defining method Haralick Texture
def hrlic(image):
    # Image is grayscaled.
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Haralick feature calculated.
    hrlic = mahotas.features.haralick(g).mean(axis=0)
    return hrlic

# defining method Histogram
def hist(image, mask=None):
    # Image is changed to HSV color-space.
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Color histogram calculated.
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    # Histogram normalised to get even distribution of points.
    cv2.normalize(hist, hist)
    # Histogram returned.
    return hist.flatten()


def classifier():
    dtry = os.path.join("training")
    # os.mkdir(dtry)
    gfeats = []
    # Obtain the current label used in training.
    print("tst {}".format(dtry))
    x = 1
    l = os.listdir(dtry)
    l.sort()

    for fe in l:
        fe = dtry + "/" + os.fsdecode(fe)
        # Image read from the folder and scaled to specified size.
        print("2 {}".format(fe))
        img = cv2.imread(fe)
        if img is not None:
            hummtsV = hummts(img)
            hrlicV = hrlic(img)
            histV = hist(img)

        # global feature 
        gfeat = np.hstack([hummtsV, hrlicV, histV])
        pr = clf.predict(gfeat.reshape(1, -1))[0]
        gfeats.append(gfeat)
        cv2.putText(img, "dog", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # Result image displayed.
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        
def main():
    s1 = "Set1"
    s2 = "Set2"

    # trainRun()
    # testRun()
    classifier()
    
if __name__ == "__main__":
    exit(main())
