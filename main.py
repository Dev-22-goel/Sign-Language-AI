import cv2
from cvzone.HandTrackingModule import HandDetector


cap= cv2.VideoCapture(0)
    
#maxhands as 2 to get atmost 2 hands and not only 2 hands

detector=HandDetector(detectionCon=0.8, maxHands=2)


while True:
    success,img=cap.read()
    hands,img=detector.findHands(img)

    if hands:
        #getting first hand

        hand1=hands[0]
        lmList1= hand1["lmList"] #list of 21 landmarks

        # centerPoint1=hand1["center"]

        #getting boundary box info of 1st hand..

        bbox1=hand1["bbox"]  #need to x, y, w, h

        #type hand L or R
        handType1= hand1["type"]

        # print(bbox1) this will give the box parameters
        # print()


        if len(hands)==2:
            hand2=hands[1]
            lmList2= hand2["lmList"] #list of 21 landmarks

            # centerPoint2=hand2["center"]

            #getting boundary box info of other hand
            bbox2=hand2["bbox"]  #need to x, y, w, h

            #type hand L or R
            handType2= hand2["type"]
            
            #box 1 upper coordinates coordinates
            left_ux=bbox1[0]
            left_uy=bbox1[1]

            #box 2 upper coordinates 
            right_ux=bbox2[0]+bbox2[2]
            right_uy=bbox2[1]

            #box 1 lower coordinates coordinates
            left_lx=left_ux
            left_ly=bbox1[1]-bbox1[3]

            #box 2 upper coordinates 
            right_lx=right_ux
            right_ly=bbox2[1]-bbox2[3]

            #prints the boundary boxes of the hands i.e. Left and right one
            print(bbox1, bbox2)
            print("Left: ", left_lx, left_ly,left_ux, left_uy)
            print("Right: ", right_lx, right_ly,right_ux, right_uy)

    cv2.imshow("Image", img)
    
    cv2.waitKey(1)