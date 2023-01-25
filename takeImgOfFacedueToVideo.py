from simple_facerec import SimpleFacerec
import face_recognition
import numpy as np
import time
import math
import cv2
import os

def logLastImgNumber(number):
    f=open('log\log.txt','w')
    f.write(str(number))
    f.close()

def getLastImgNumber():
    f=open('log\log.txt','r')
    data=f.readline()
    f.close()
    return data

def encodeTesters():
    testerArr=[]
    for names in os.listdir('tester'):
        tester = cv2.cvtColor(cv2.imread("tester/{}".format(names)), cv2.COLOR_BGR2RGB)
        cv2.imwrite("log\Testers\Tester{}".format(names),tester)
        tester_encoding = face_recognition.face_encodings(tester)[0]
        testerArr.append(tester_encoding)
    return testerArr

def simpleTest(ımgTemp,testerArr):    
    encodedTemp = face_recognition.face_encodings(ımgTemp)
    accuracyCnt=0
    
    for tester in testerArr:
        try:
            if face_recognition.compare_faces([encodedTemp], tester):
                accuracyCnt+=1
        except:
            return 0
            pass
    
    if accuracyCnt>=(len(testerArr)/2):
        return 1
    else:
        return 0

def offsetCrop(locations,offsetPx):
    newLocations=[]     
    for index,loc in enumerate(locations): #locations should be x1 y1 x2 y2
        loc = loc-offsetPx if index <= 1 else loc+offsetPx
        loc = 0 if loc<0 else loc
        if index%2==0: #x
            loc= int(videdoWidth) if loc>videdoWidth else loc
        else: #y
            loc= int(videoHeight) if loc>videoHeight else loc
        newLocations.append(loc)
    return newLocations

encodedTesters = encodeTesters()
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

print("Started to capture ..!")

cap = cv2.VideoCapture('input.mp4')

videdoWidth  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # float `width`
videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

frameCounter=0
previous=0
startTime=time.time()

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #cv2.imshow('frame',frame)
    frameCounter+=1
    progressPercentage = math.floor((frameCounter/frame_count)*100)
    if ((frameCounter/frame_count)*100)-progressPercentage < 0.05:
        os.system("cls")
        print("[{}{}]  progressPercentage: %{}".format(math.floor(progressPercentage/2)*"-",math.ceil((100-progressPercentage)/2)*" ",progressPercentage))
    timepassed=time.time()-startTime
    if timepassed > 3:
        startTime=time.time()
    
        face_locations, face_names =sfr.detect_known_faces(frame) # screenshot in known_faces()
        for face_loc, face_names in zip(face_locations, face_names):
            if face_names!="Unknown":
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                [x1,y1,x2,y2]=offsetCrop([x1,y1,x2,y2],math.floor((x2-x1)/3))
                croppedImage= frame[y1:y2,x1:x2]

                fileNum=getLastImgNumber()

                if simpleTest(ımgTemp=croppedImage,testerArr=encodedTesters):
                    cv2.imwrite('output/target{}.jpg'.format(fileNum),croppedImage)
                else:
                    cv2.imwrite('noise/noise{}.jpg'.format(fileNum),croppedImage)
                logLastImgNumber(int(fileNum)+1)
        
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  else: 
    print("Completed")
    break
