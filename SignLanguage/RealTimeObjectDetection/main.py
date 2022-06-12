import cv2
import os
import os.path
import fnmatch
import time
from cvzone.HandTrackingModule import HandDetector


bool = os.path.isdir("Output Images")
if bool == True:
    pass
if bool == False:
    os.mkdir("Output Images")
label = input("Specify the sign name ")
path = "/Users/ashmika/Desktop/hand_detector/Output Images"


cv2.namedWindow("test")
img_counter = 0
handImg_counter =0

# SET THE COUNTDOWN TIMER
# for simplicity we set it to 3
# We can also take this as input
TIMER = int(3)
def cropImage(bboxes, img):
    xMin = bboxes[0][0]
    xMax = xMin + bboxes[0][2]
    yMin = bboxes[0][1]
    yMax = yMin + bboxes[0][3]

    #if there are 2 hands
    if len(bboxes) == 2:
        xMin1 = bboxes[1][0]
        xMax1 = xMin1 + bboxes[1][2]
        yMin1 = bboxes[1][1]
        yMax1 = yMin1 + bboxes[1][3]

        if xMin > xMin1:
            xMin = xMin1
        if xMax < xMax1:
            xMax = xMax1
        if yMin > yMin1:
            yMin = yMin1
        if yMax < yMax1:
            yMax = yMax1

    #crop the image
    croppedImg = img[(yMin-20):(yMax+20), (xMin-20):(xMax+20)]
    return croppedImg

def saveImage(img, fileName):
    # check if data directory exist
    path = os.getcwd()
    path = path + '\data'
    if not os.path.isdir(path):
        os.mkdir(path)

    # find and count the number files whose names are similar to fileName
    numOfFile = len(fnmatch.filter(os.listdir(path), (fileName+'*.png')))

    # save image
    # cv2.imwrite(fileName, img)
    fileName = (fileName + "{}.png").format(numOfFile)
    cv2.imwrite(os.path.join(path, fileName), img)


def handDetect(img):
    list_bbox = []
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    hands, img = detector.findHands(img)

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)
        list_bbox.append(bbox1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)
            list_bbox.append(bbox2)

    return list_bbox

# Open the camera
cam = cv2.VideoCapture(0)

while True:

    # Read and display each frame
    ret, img = cam.read()
    wholeImage = img

    cv2.imshow('test', img)

    # check for the key pressed
    k = cv2.waitKey(125)

    # set the key for the countdown
    # to begin. Here we set q
    # if key pressed is q
    if k % 256 == 32:
        prev = time.time()

        while TIMER >= 0:
            ret, img = cam.read()
            handDetect(img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(TIMER),
                        (200, 250), font,
                        7, (0, 0, 255),
                        4, cv2.LINE_AA)
            cv2.imshow('test', img)
            cv2.waitKey(125)

            # current time
            cur = time.time()

            if cur - prev >= 1:
                prev = cur
                TIMER = TIMER - 1

        else:
            ret, img = cam.read()
            l = handDetect(img)

            cv2.imshow('test', img)
            #img_name = "opencvframe{}.png".format(img_counter)
            # time for which image displayed
            cv2.waitKey(2000)
            cropImage(l, img)
            #cv2.imwrite(img_name, img)

            #crop and save image
            croppedImg = cropImage(l, img)
            saveImage(croppedImg, label)

            # Save the frame
            # cv2.imwrite('test.jpg', img)
            #print("{} written!".format(img_name))
            img_counter += 1
            handImg_counter += 1

            # HERE we can reset the Countdown timer
            # if we want more Capture without closing
            # the camera
            TIMER = int(3)

    # Press Esc to exit
    elif k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

# close the camera
cam.release()

# close all the opened windows
cv2.destroyAllWindows()

'''

import cv2  # Import opencv
import uuid  # Import uuid
import os  # Import Operating System
import time  # Import time

# using a folder to store our dataset
IMAGES_PATH = 'Tensorflow/workspace/images/collectedimages'

# creating different folders which are used in categorizing the language
labels = ['Hi', 'Thanks', 'iloveyou', 'yes', 'no']

# taking input images
number_imgs = 10

for label in labels:

    os.mkdir('Tensorflow\workspace\images\collectedimages\\' + label)

    #using external webcam otherwise using 0 in parameter
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("The Camera is not Opened....Exiting")
        exit()
    print('Collecting images for {}'.format(label))
    print("*****************************************")
    time.sleep(2)

    for imgnum in range(10):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        # uuid is used to write up the images, so that it can prevent overlapping
        imgname = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)

        # gives a break of 2 sec after an image capture
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
'''