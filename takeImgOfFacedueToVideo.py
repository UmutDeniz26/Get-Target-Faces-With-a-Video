from simple_facerec import SimpleFacerec
import face_recognition
import numpy as np
import time
import math
import cv2
import os


globalTımer=time.time()

#default
wantedFrameNum=500
accuracyLimit=97

#wantedFrameNum=int(input("Type the number of frame that you want examine: "))
#accuracyLimit=int(input("Type the number of accuracy limit that you want (max->100, min->0) : "))
accuracyLimit=0 if accuracyLimit<0 else 100 if accuracyLimit>100 else accuracyLimit  

#print(face_recognition.compare_faces([encodedTemp], tester))
#cv2.imshow("tester-{}".format(getLastImgNumber()),cv2.imread("tester/{}".format(os.listdir('tester')[index])))
#cv2.imshow("obj-{}".format(getLastImgNumber()),ımgTemp)
#cv2.waitKey(0)
#cv2.imshow("s",cv2.imread("output/target105-100%.jpg"))
#cv2.waitKey(0)

def appendErrorLog(write):
    f=open('log\errorLog.txt','a')
    f.write(str(write)+"\n")
    f.close()

def logWrite(write,location):
    f=open('log\{}'.format(location),'w')
    f.write(str(write))
    f.close()

def getLastImgNumber():
    f=open('log/fileNumberLog.txt','r')
    data=f.readline()
    f.close()
    return data

def encodeTesters():
    testerArr=[]
    for names in os.listdir('tester'):
        tester = cv2.cvtColor(cv2.imread("tester/{}".format(names)), cv2.COLOR_BGR2RGB)
        tester_encoding = face_recognition.face_encodings(tester)[0]
        testerArr.append(tester_encoding)
    return testerArr

def simpleTest(ımgTemp,testerArr):     
    try:
        encodedTemp = face_recognition.face_encodings(cv2.cvtColor(ımgTemp, cv2.COLOR_BGR2RGB))[0]
    except:
        appendErrorLog("indexErr")
        return "indexErr"
    accuracyCnt=0
    for index,tester in enumerate(testerArr):
        if face_recognition.compare_faces([encodedTemp], tester)==[True]:    
            accuracyCnt+=1
    
    accuracyPercentage=int((accuracyCnt/len(os.listdir('tester')))*100)
    if accuracyPercentage>accuracyLimit:
        appendErrorLog("Done, Accur: {}".format(accuracyPercentage))
    else:
        appendErrorLog("Error, Accur: {}".format(accuracyPercentage))
    
    return accuracyPercentage

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

def formatCustomDigit(number,digitNumberOfReturn):
    if type(number)!=type(5):
        return number
    counter=0
    holdNumber=number
    while(number>=1):
        number/=10
        counter+=1
    
    return "0"*(digitNumberOfReturn-counter)+str(holdNumber)

def getVideoLengths():  
    videoLengths=[]  
    for videoName in os.listdir('inputVideos'):
        cap = cv2.VideoCapture('inputVideos/{}'.format(videoName))
        videoLengths.append(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS))
    
    return videoLengths


frameCounter,savedImg,noiseImg,videoCnt,examinedFramescnt=0,0,0,0,0

print("Tester encoding started!")
encodedTesters = encodeTesters()
print("Tester encoding completed!")
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
print("Started to capture ..!")
logWrite("","errorLog.txt")



weightedDistributionOfWantedLengths=[]

videoLengths=getVideoLengths()
for lengths in videoLengths:
    weightedDistributionOfWantedLengths.append(int(lengths*wantedFrameNum/sum(videoLengths)))
    
print(weightedDistributionOfWantedLengths)



for index,videoName in enumerate(os.listdir('inputVideos')):
    videoCnt+=1
    frameCounter=1
    cap = cv2.VideoCapture('inputVideos/{}'.format(videoName))

    videdoWidth  = cap.get(cv2.CAP_PROP_FRAME_WIDTH);videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    
    wantedFrameNum=weightedDistributionOfWantedLengths[index]
    videoDuration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
    frameScaler=math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT)/(wantedFrameNum)) #-> it looks frames time: frame/constant
    examinedFramescnt+=(wantedFrameNum)

    localTimer=time.time()
    while(1):
        frameCounter+=1
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameCounter*frameScaler)
        ret, frame = cap.read()
        if ret == True:
            progressPercentage =(cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT))*100
            passedTıme=math.floor(time.time()-localTimer)
            #Eta = math.floor(passedTıme*(100-progressPercentage)/progressPercentage) if progressPercentage!=0 else 1
            os.system("cls")
            print("{} - {}  ProgressPercentage: %{}   Video Length: {}h:{}m:{}s    {} frames will be examined for this video".format(
                videoCnt,videoName,round(progressPercentage,2),formatCustomDigit(int(videoDuration/(60*60)),2)
                ,formatCustomDigit(int((videoDuration/(60))%60),2),
                formatCustomDigit(int(videoDuration%60),2),
                wantedFrameNum))

            face_locations, face_names =sfr.detect_known_faces(frame) # screenshot in known_faces()
            for face_loc, face_names in zip(face_locations, face_names): 
                if face_names!="Unknown":
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                    [x1,y1,x2,y2]=offsetCrop([x1,y1,x2,y2],math.floor((x2-x1)/3))
                    croppedImage= frame[y1:y2,x1:x2]

                    if type(simpleTest(croppedImage,encodedTesters))!=type("^_^") and simpleTest(croppedImage,encodedTesters)>accuracyLimit:
                        cv2.imwrite('output/target{}-{}%.jpg'.format(getLastImgNumber(),
                            str(formatCustomDigit(simpleTest(croppedImage,encodedTesters),2))),croppedImage)
                        savedImg+=1
                    else:
                        cv2.imwrite('noise/noise{}-{}%.jpg'.format(getLastImgNumber(),
                            str(formatCustomDigit(simpleTest(croppedImage,encodedTesters),2))),croppedImage)
                        noiseImg+=1
                    logWrite((int(getLastImgNumber())+1),'fileNumberLog.txt')   
                
        else: 
            if videoCnt>= len(os.listdir('inputVideos')):
                os.system("cls")
                print("%100 Completed , Time passed: {}min {}sec  ,  # of Saved Images: {}  ,  # of Noise Images: {}  ,  # of CantDetectFace: {}  ,  # of examined frames: {}  ,  # of input video: {}"
                    .format(int(int(time.time()-globalTımer)/60),int(int(time.time()-globalTımer)%60),savedImg,noiseImg,
                    int(examinedFramescnt)-savedImg-noiseImg,int(examinedFramescnt),videoCnt))
            break
