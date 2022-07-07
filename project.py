#importing some essential libraries
import time
import cv2
import numpy as np
import pickle

class mpHands:
    import mediapipe as mp
    def __init__(self,maxHands=2,tol1=.5,tol2=.5):
        #this is creating a mediapipe object
        self.hands=self.mp.solutions.hands.Hands(False,maxHands,tol1,tol2)
    def Marks(self,frame):
        #showcasing landmark of 1st hand to the other hand as a 2D array
        myHands=[]

        #converting into RGB frame
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)

        #when hand is detected (with landmarks)
        if results.multi_hand_landmarks != None:
            #this has all the data of all the hands
            for handLandMarks in results.multi_hand_landmarks:
                #analysing one hand at a time (meaning to read and understand one landmark of 1 hand at a time)
                myHand=[]
                for landMark in handLandMarks.landmark:
                    #trying to modify that tuple
                    myHand.append((int(landMark.x*width),int(landMark.y*height)))

                #after finishing analysing the 1st hand we are going to store that data
                myHands.append(myHand)
        return myHands

def findDistances(handData): 
    #passing landmarks in this function

    #creating an empty array of 0's
    distMatrix = np.zeros([len(handData), len(handData[0]), len(handData[0])], dtype='float')
    for i in range(0, len(handData)):
        #used for calculating distance

        #using the one point as the POINT OF REFERERENCE..making it more accurate
        palmSize=((handData[i][0][0]-handData[i][9][0])**2+(handData[i][0][1]-handData[i][9][1])**2)**(1./2.)
        for row in range(0,len(handData[i])):
            for column in range(0,len(handData[i])):
                distMatrix[i][row][column]=(((handData[i][row][0]-handData[i][column][0])**2+(handData[i][row][1]-handData[i][column][1])**2)**(1./2.))/palmSize

    return distMatrix

#used for determing absolute error 
def findError(gestureMatrix,unknownMatrix,keyPoints):
    error=0
    handLen = len(gestureMatrix)
    for i in range(0,handLen):
        for row in keyPoints:
            for column in keyPoints:
                #calculating error from a definite and unknown source image
                error = error + abs(gestureMatrix[i][row][column] - unknownMatrix[i][row][column])
    if handLen == 2:
        error = error/2
    return error

def findGesture(unknownGesture,knownGestures,keyPoints,gestNames,tol):

    errorArray=[]
    #stepping thru each gesture
    for i in range(0,len(gestNames),1):
        #checking the preset data if it is similar to the unknown gesture
        if len(knownGestures[i]) == len(unknownGesture):
            #calculating error
            error=findError(knownGestures[i],unknownGesture,keyPoints)
        else:
            error = -1
        errorArray.append(error)
    j = 0
    errorMin = -1
    while j < len(errorArray):
        if errorArray[j]!=-1:
            errorMin = errorArray[j]
            minIndex = j

            break;
        j += 1
    if (errorMin == -1):
        gesture = 'unknown'
    else:
        #analyzing error and trying to get lowest
        for i in range(j,len(errorArray),1):
            if error != -1 and errorArray[i]<errorMin:
                errorMin=errorArray[i]
                minIndex=i
        if errorMin<tol:
            gesture=gestNames[minIndex]
        if errorMin>=tol:
            gesture='Unknown'
    return gesture

#size of window
width=1280
height=720

#opening camera
cam=cv2.VideoCapture(0)

#resizing tactics
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
findHands=mpHands(2)
time.sleep(5)
#defining specfic landmarks
keyPoints=[0,4,5,9,13,17,8,12,16,20]

#using serialization method called "pickling"

#process of capturing gesture
train=int(input('Enter 1 to Train, Enter 0 to Recognize '))
if train==1:
    trainCnt=0
    #storing gesture names and checking if we have a preset data
    knownGestures=[]
    numGest=int(input('How Many Gestures Do You Want?: '))
    gestNames=[]
    for i in range(0,numGest,1):
        prompt='Name of Gesture #'+str(i+1)+' '
        name=input(prompt)
        #saving gesture labels in array
        gestNames.append(name)
    print(gestNames)
    trainName=input('Filename for training data? (Press Enter for Default): ')
    if trainName=='':
        trainName='default'
    trainName=trainName+'.pkl'
if train==0:
    trainName=input('What Training Data Do You Want to Use? (Press Enter for Default): ')
    if trainName=='':
        trainName='default'
    trainName=trainName+'.pkl'
    #here the gesture landmarks are stored in a pickle file which is coded and helps in data integrity
    with open(trainName,'rb') as f:
        gestNames=pickle.load(f)
        knownGestures=pickle.load(f)
#this is a secured file and cannot be changed manually making it useful for data confidentiality

#here we are using timer to make the user ready to capture desire pattern(like 2 hands)
tol=10
TIMER = int(3)
while True:

    #reading camera image
    ignore,  frame = cam.read()

    #re-sizing camera
    frame=cv2.resize(frame,(width,height))
    handData=findHands.Marks(frame)
    if train==1:
        if handData!=[]:
            ##if there is a new gesture
            print('Please Show Gesture ',gestNames[trainCnt],': Press t when Ready')
            k = cv2.waitKey(125)
            if k == ord('t'):
                prev = time.time()
                while TIMER >=0:
                    ignore,frame = cam.read()
                    frame = cv2.resize(frame, (width, height))
                    handData = findHands.Marks(frame)
                    for hand in handData:
                        for ind in keyPoints:
                            cv2.circle(frame, hand[ind], 25, (255, 0, 255), 3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(TIMER),
                                (200, 250), font,
                                7, (0, 0, 255),
                                4, cv2.LINE_AA)
                    cv2.imshow('my WEBcam', frame)
                    cv2.waitKey(125)
                    cur = time.time()
                    if cur - prev >= 1:
                        prev = cur
                        TIMER = TIMER - 1
                else:
                    ignore, frame = cam.read()
                    frame = cv2.resize(frame, (width, height))
                    knownGesture=findDistances(handData)
                    knownGestures.append(knownGesture)
                    trainCnt=trainCnt+1
                    if trainCnt==numGest:
                        train=0
                        with open(trainName,'wb') as f:
                            pickle.dump(gestNames,f)
                            pickle.dump(knownGestures,f)
                TIMER = int(3)
    if train == 0:
        if handData!=[]:
            #so now doing a train after capturing
            unknownGesture=findDistances(handData)
            myGesture=findGesture(unknownGesture,knownGestures,keyPoints,gestNames,tol)
            
            #displaying title
            cv2.putText(frame,myGesture,(100,175),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),8)
    for hand in handData:
        for ind in keyPoints:
            #drawing circle around landmarks
            cv2.circle(frame,hand[ind],25,(255,0,255),3)
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    
    #secret quit key
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()
