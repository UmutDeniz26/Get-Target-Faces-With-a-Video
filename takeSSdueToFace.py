from simple_facerec import SimpleFacerec
import face_recognition
import pyautogui as pg
import numpy as np
import keyboard
import asyncio
import time
import math
import cv2
import os

def logLastImgNumber(number):
    f=open('log.txt','w')
    f.write(str(number))
    f.close()

def getLastImgNumber():
    f=open('log.txt','r')
    data=f.readline()
    f.close()
    return data

#face_recognition.face_encodings(cv2.cvtColor(np.array(screenshotTemp),cv2.COLOR_RGB2BGR))
def encodeTesters():
    testerArr=[]
    for names in os.listdir('tester'):
        tester = cv2.cvtColor(cv2.imread("tester/{}".format(names)), cv2.COLOR_BGR2RGB)
        #cv2.imwrite("Tester{}".format(names),tester)
        tester_encoding = face_recognition.face_encodings(tester)[0]
        testerArr.append(tester_encoding)
    return testerArr

def simpleTest(encodedScreenshot,testerArr):    
    accuracyCnt=0
    errorCnt=0
    print("Test Begins :)")
    for index, tester in enumerate(testerArr):
        try:
            if face_recognition.compare_faces([encodedScreenshot], tester):
                print("Test {} passed !".format(index))
                accuracyCnt+=1
        except:
            errorCnt+=1

    print("AccuracyCnt: {}".format(accuracyCnt),"    ErrorCnt: {}".format(errorCnt))
    return len(testerArr)-accuracyCnt # if all test accurate, returns 0


encodedTesters = encodeTesters()

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

start = time.time()
print("Started to capture ..!")

while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        
        if time.time()-start > 0.2:
            start=time.time()
            face_locations, face_names =sfr.detect_known_faces(cv2.cvtColor(np.array(pg.screenshot()),cv2.COLOR_RGB2BGR)) # screenshot in known_faces()
            for face_loc, face_names in zip(face_locations, face_names):
                if face_names!="Unknown":
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    offset=math.floor((x2-x1)/2)
                    screenshotTemp = pg.screenshot(region=(x1-offset,y1-offset,x2-x1+2*offset,y2-y1+2*offset))
                    
                    fileNum=getLastImgNumber()
                    encodedTemp = face_recognition.face_encodings(cv2.cvtColor(np.array(screenshotTemp),cv2.COLOR_RGB2BGR))
                    
                    if simpleTest(encodedTemp,encodedTesters) == 0:
                        screenshotTemp.save('output/target{}.jpg'.format(fileNum))
                        print("  [x2-x1]: ",x2-x1,"  [y2-y1]: ",y2-y1,"   Name:",face_names," FileNum:",fileNum," base:{",x1,y1,x2,y2,"}")
                    else:
                        screenshotTemp.save('noise/noise{}.jpg'.format(fileNum))             
                    logLastImgNumber(int(fileNum)+1)

        
        if keyboard.is_pressed('q'):  # if key 'q' is pressed 
            print('You Pressed q Key!')
            break  # finishing the loop
    except:
        break  # if user pressed a key other than the given key the loop will break