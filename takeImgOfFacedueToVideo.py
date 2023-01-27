from simple_facerec import SimpleFacerec
import face_recognition
import time
import math
import cv2
import os


frameCursor,savedImg,noiseImg,videoCnt,examinedFramescnt,cantDetectCnt,DetectFaceFlag,otherFacesCnt=0,0,0,0,0,0,0,True

#default
wantedFrameNum=300
accuracyLimit=97
distributionChoice=0
clearFolderContentChoice=1
cropOffsetDivider=3

#wantedFrameNum=int(input("Type the number of frame that you want examine: "))
#accuracyLimit=int(input("Type the number of accuracy limit that you want (max->100, min->0) : "))
#distributionChoice=int(input("Type the distribution choice (1 -> Same weights, else -> Weighted distribution due to video lengths) : "))
#clearFolderContentChoice=int(input("Type the clear content choice (1 -> Clear, else -> Hold) : "))

#accuracy correction
accuracyLimit=0 if accuracyLimit<0 else 100 if accuracyLimit>100 else accuracyLimit  

def appendErrorLog(write):
    f=open('log/faceDetectAccuracyLog.txt','a')
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
    for names in os.listdir('TargetImagesToTest'):
        tester = cv2.cvtColor(cv2.imread("TargetImagesToTest/{}".format(names)), cv2.COLOR_BGR2RGB)
        tester_encoding = face_recognition.face_encodings(tester)[0]
        testerArr.append(tester_encoding)
    return testerArr

def simpleTest(ımgTemp,testerArr):     
    try:
        encodedTemp = face_recognition.face_encodings(cv2.cvtColor(ımgTemp, cv2.COLOR_BGR2RGB))[0]
    except:
        return "indexErr"
    accuracyCnt=0
    for index,tester in enumerate(testerArr):
        if face_recognition.compare_faces([encodedTemp], tester)==[True]:    
            accuracyCnt+=1
    
    accuracyPercentage=round((accuracyCnt/len(os.listdir('TargetImagesToTest')))*100,2)
    return int(accuracyPercentage)

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

def clearFolderContent(folderName):
    for f in os.listdir('{}'.format(folderName)):
        os.remove('{}/{}'.format(folderName,f))

def updateLogAndSaveImg(path,img,title="Untitled",accuracyPercentage="??"):
    logWrite((int(getLastImgNumber())+1),'fileNumberLog.txt')
    cv2.imwrite(path,img)
    appendErrorLog("{}            Img Number: {0:4}          Accuracy: %{1:4}".format(title,getLastImgNumber(),accuracyPercentage))

globalTımer=time.time()

#Encode testers
print("Tester encoding started!")
encodedTesters = encodeTesters()
print("Tester encoding completed!")

#Load Searcher
sfr = SimpleFacerec()
sfr.load_encoding_images("TargetImagesToSearch/")
print("Started to capture ..!")

#initial logs
logWrite("","faceDetectAccuracyLog.txt");logWrite("1","fileNumberLog.txt")

#Clears selected folders
if clearFolderContentChoice:
    clearFolderContent('output');clearFolderContent('noise');clearFolderContent('cantDetect');clearFolderContent('otherFaces')

weightedDistributionOfWantedLengths=[]
for lengths in getVideoLengths():
    #calculate Distribution weights
    weightedDistributionOfWantedLengths.append(int(lengths*wantedFrameNum/sum(getVideoLengths())))

#same weights
wantedFrameNum/=len(os.listdir('inputVideos')) if wantedFrameNum > len(os.listdir('inputVideos')) else wantedFrameNum


if distributionChoice == 1:

    #overwrite weights due to set same weights to all videos
    weightedDistributionOfWantedLengths=[math.ceil(wantedFrameNum)]*len(os.listdir('inputVideos'))


for index,videoName in enumerate(os.listdir('inputVideos')):
    
    #Setups to search faces in video
    videoCnt+=1
    frameCursor=0

    #load the video
    cap = cv2.VideoCapture('inputVideos/{}'.format(videoName))
    
    #distrube the weights to set scale factor
    wantedFrameNum=weightedDistributionOfWantedLengths[index]
    
    #get video attributes
    videdoWidth  = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    videoDuration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)

    wantedFrameNum = cap.get(cv2.CAP_PROP_FRAME_COUNT)  if wantedFrameNum>cap.get(cv2.CAP_PROP_FRAME_COUNT) else wantedFrameNum
    frameScaler=math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)/(wantedFrameNum)) #-> it looks frames time: frame/constant
    examinedFramescnt+=(wantedFrameNum)
    localTimer=time.time()

    #Starts searching
    while(True):
        frameCursor+=1

        #cursor * scaler -> if there is 18000frame in video -> for example, scaler is 500 ==> frameCursor would be 500,1000,1500 ... 18000
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameCursor*frameScaler)
        ret, frame = cap.read()
        DetectFaceFlag=False
        if ret == True:

            progressPercentage =(cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT))*100
            passedTıme=math.floor(time.time()-localTimer)
            
            #Information about process
            os.system("cls")
            print("{} - {}  ProgressPercentage: %{}   Video Length: {}h:{}m:{}s  frameScaler: {}    {} frames will be examined for this video".format(
                videoCnt,videoName,round(progressPercentage,2),formatCustomDigit(int(videoDuration/(60*60)),2)
                ,formatCustomDigit(int((videoDuration/(60))%60),2),formatCustomDigit(int(videoDuration%60),2),
                round(frameScaler),wantedFrameNum))
            
            #searchs faces in the image
            face_locations, face_names =sfr.detect_known_faces(frame) 
            for face_loc, face_names in zip(face_locations, face_names): 
                #crop image due to face
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                [x1,y1,x2,y2]=offsetCrop([x1,y1,x2,y2],math.floor((x2-x1)/cropOffsetDivider))
                croppedImage= frame[y1:y2,x1:x2]
                
                #if there is this face in the TargetImagesToSearch file
                if face_names!="Unknown":
                    DetectFaceFlag=True
                    #simple test due to TargetImagesToTest folder
                    testResult=simpleTest(croppedImage,encodedTesters) # result is in type of %
                    

                    if type(testResult)!=type("String") and testResult>accuracyLimit:
                        updateLogAndSaveImg(
                            path='output/target{}-%{}.jpg'.format(getLastImgNumber(),testResult),
                            img=croppedImage,
                            title="Face Detected Successfully!",
                            accuracyPercentage=testResult)
                        savedImg+=1
                        break
                    else:
                        updateLogAndSaveImg(
                            path='noise/noise{}-%{}.jpg'.format(getLastImgNumber(),testResult),
                            img=croppedImage,
                            title="Noise has been detected!",
                            accuracyPercentage=testResult)
                        noiseImg+=1   
                else:
                    updateLogAndSaveImg(
                        path='otherFaces/otherFaces{}.jpg'.format(getLastImgNumber()),
                        img=croppedImage,
                        title="Unknown face detected")
                    otherFacesCnt+=1


            #neither targetface nor noiseface
            if DetectFaceFlag==False:
                cantDetectCnt+=1
                updateLogAndSaveImg(
                    path='cantDetect/cantDetect{}.jpg'.format(getLastImgNumber()),
                    img=frame,
                    title="No face could be identified."
                )
                
        else: 
            #searching is over
            if videoCnt>= len(os.listdir('inputVideos')):
                os.system("cls")
                print("%100 Completed , Time passed: {}min {}sec  ,  # of Saved Images: {}  ,  # of Noise Images: {}  ,  # of Undetected face: {}  ,  # of other faces: {} ,  # of examined frames: {}  ,  # of input video: {}"
                    .format(int(int(time.time()-globalTımer)/60),int(int(time.time()-globalTımer)%60),savedImg,noiseImg,
                    int(examinedFramescnt)-savedImg-noiseImg,otherFacesCnt,int(examinedFramescnt),videoCnt))
            break
        #To create folder names
