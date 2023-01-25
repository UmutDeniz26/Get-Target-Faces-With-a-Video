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

frameCounter,savedImg,noiseImg,cantDetectCnt=0,0,0,0
globalTımer=time.time()
cantDetectFlag=True
frameScaler=1000
videoCnt=0

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
print("Started to capture ..!")

for videoName in os.listdir('inputVideos'):
    videoCnt+=1
    frameCounter=0
    cap = cv2.VideoCapture('inputVideos/{}'.format(videoName))
    videdoWidth  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # float `width`
    videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  


    while(cap.isOpened()):
        
        frameCounter+=1
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameCounter*frameScaler)
        ret, frame = cap.read()
        
        if ret == True:
            progressPercentage =(cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT))*100
            passedTıme=math.floor(time.time()-globalTımer)
            Eta = math.floor(passedTıme*(100-progressPercentage)/progressPercentage) if progressPercentage!=0 else 1
            
            os.system("cls")
            print("Video Number: {} ProgressPercentage: %{}    Time passed: {}min {}sec     ETA: {}min {}sec".format(videoCnt,round(progressPercentage,2),
            math.floor(passedTıme/60),math.floor(passedTıme%60),math.floor(Eta/60),math.floor(Eta%60)))


            face_locations, face_names =sfr.detect_known_faces(frame) # screenshot in known_faces()
            for face_loc, face_names in zip(face_locations, face_names):
                if face_names!="Unknown":
                    cantDetectFlag=False
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                    [x1,y1,x2,y2]=offsetCrop([x1,y1,x2,y2],math.floor((x2-x1)/3))
                    croppedImage= frame[y1:y2,x1:x2]

                    if True:
                        cv2.imwrite('output/target{}.jpg'.format(getLastImgNumber()),croppedImage);savedImg+=1
                    else:
                        cv2.imwrite('noise/noise{}.jpg'.format(getLastImgNumber()),croppedImage);noiseImg+=1
                    logLastImgNumber(int(getLastImgNumber())+1)      
            if cantDetectFlag:
                cantDetectCnt+=1
            cantDetectFlag=True

        else: 
            os.system("cls")
            print("%100 Completed     Time passed: {}min {}sec    # of Saved Images: {}     # of Noise Images: {}    # of CantDetectImg: {}    # of input video: {}"
                .format(int(int(time.time()-globalTımer)/60),int(int(time.time()-globalTımer)%60),savedImg,noiseImg,cantDetectCnt,videoCnt))
            break
