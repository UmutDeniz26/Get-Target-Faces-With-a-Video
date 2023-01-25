import cv2
import pyautogui as pg
import numpy as np
from simple_facerec import SimpleFacerec
import face_recognition

import asyncio
import keyboard
import math

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
Cnt=400
print("Lezzz Gooo")

tester = cv2.cvtColor(cv2.imread("tester/tester.jpg"), cv2.COLOR_BGR2RGB)
tester_encoding = face_recognition.face_encodings(tester)[0]

async def execute_loop():
    while True:
        await asyncio.sleep(0.1)
        global Cnt
        
        #finds faces and its positions, also compares with images folder
        face_locations, face_names =sfr.detect_known_faces(cv2.cvtColor(np.array(pg.screenshot()),cv2.COLOR_RGB2BGR)) # screenshot in known_faces()

        for face_loc, face_names in zip(face_locations, face_names):
            if face_names!="Unknown":
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                
                offset=math.floor((x2-x1)/2)
                screenshotTemp = pg.screenshot(region=(x1-offset,y1-offset,x2-x1+2*offset,y2-y1+2*offset))
                
                print("  [x2-x1]: ",x2-x1,"  [y2-y1]: ",y2-y1,"   Name:",face_names," FileNum:",Cnt," base:{",x1,y1,x2,y2,"}",)
                try:
                    if face_recognition.compare_faces([face_recognition.face_encodings(cv2.cvtColor(np.array(screenshotTemp),cv2.COLOR_RGB2BGR))], tester_encoding):
                        screenshotTemp.save('output/target{}.jpg'.format(Cnt))
                    else:
                        screenshotTemp.save('noise/noise{}.jpg'.format(Cnt))                
                except:
                    screenshotTemp.save('noise/noise{}.jpg'.format(Cnt))             

                Cnt+=1

        if keyboard.is_pressed('esc'):
            cv2.destroyAllWindows()
            break

asyncio.run(execute_loop())
