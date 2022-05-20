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
