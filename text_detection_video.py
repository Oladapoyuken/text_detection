# USAGE
# python text_detection_video.py --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.9:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


cons = 0
tempStartX = 0
tempEndX = 0
tempStartY = 0
tempEndY = 0
number = 0
expand = 0



# extract plate
def select_largestBox(xmin, xmax, ymin, ymax):
    print('Normal Boxes :\t')
    # lstmp = [xmin, xmax, ymin, ymax]
    # print(lstmp)
    # print('\n')
    global tempStartY
    global tempEndY
    global tempStartX
    global tempEndX
    global expand


    if (ymax - ymin) > (tempEndY - tempStartY):
        if (xmax - xmin) > (tempEndX - tempStartX):
            expand = xmax - xmin
            tempStartX = xmin
            tempEndX = xmax
            tempStartY = ymin
            tempEndY = ymax

            print('Largest Box :\t')
            lstmp = [tempStartX, tempEndX, tempStartY, tempEndY]
            print(lstmp)
            print('\n')

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
# (newW, newH) = (args["width"], args["height"])
(newW, newH) = (320, 320)
(rW, rH) = (None, None)


layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('C:\\Users\\Yuken4real\\PycharmProjects\\opencv-text-detection\\frozen_east_text_detection.pb')

vs = cv2.VideoCapture('C:\\Users\\Yuken4real\\PycharmProjects\\opencv-text-detection\\flipthis.mp4')

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while vs.isOpened():
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ret, frame = vs.read()
    # frame = frame[1] #if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame, maintaining the aspect ratio
    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()
    extractedImage = orig.copy()

    # if our frame dimensions are None, we still need to compute the
    # ratio of old frame dimensions to new frame dimensions
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

    # resize the frame, this time ignoring aspect ratio
    frame = cv2.resize(frame, (newW, newH))

    # construct a blob from the frame and then perform a forward pass
    # of the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        if (endY - startY) > (tempEndY - tempStartY):
            if (endX - startX) > (tempEndX - tempStartX):
                expand = endX - startX
                tempStartX = startX
                tempEndX = endX
                tempStartY = startY
                tempEndY = endY
                upgradeL = tempStartX - expand
                upgradeR = tempEndX + expand
                upgradeU = tempStartY - 10
                upgradeD = tempEndY + 10

                if upgradeL < 0:
                    upgradeL = 0

                roi = extractedImage[upgradeU:upgradeD, upgradeL:upgradeR]


        # Select the most probable box for plate number extraction
        # select_largestBox(startX, endX, startY, endY)

        # draw the bounding box on the frame
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        (h2, w2) = orig.shape[:2]



        # print('Extracted box')
        # print(tempStartX)
        # print(tempStartY)
        # print(tempEndX)
        # print(tempEndY)

        cv2.imshow('CROP', roi)
        # vs.release()


    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Text Detection", orig)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
# if not args.get("video", False):
# vs.stop()
#
# # otherwise, release the file pointer
# else:
vs.release()
cv2.destroyAllWindows()

cv2.imshow("Plate number", roi)
cv2.waitKey(0)

# close all windows
cv2.destroyAllWindows()
