from simple_facerec import SimpleFacerec
import face_recognition
import random
import time
import math
import cv2
import os
globalTımer=time.time()


frameCursor,targetImgCnt,noiseImgCnt,frameCnt,zeroFaceCnt,unknownFaceCnt,imgsWithAnyFace=0,0,0,0,0,0,0
infoTexts=[]

#default
wantedFrameNum=100
accuracyLimit=97
distributionChoice=0 
clearFolderContentChoice=1
cropOffsetDivider=3

#wantedFrameNum=int(input("Type the number of frame that you want examine: "))
#accuracyLimit=int(input("Type the number of accuracy limit that you want (max->100, min->0) : "))
#distributionChoice=int(input("Type the distribution choice (1 -> Same weights, else -> Weighted distribution due to video lengths) : "))
#clearFolderContentChoice=int(input("Type the clear content choice (1 -> Clear, else -> Hold) : "))

holdWantedFrameNum=wantedFrameNum
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

def getVideoLengths():  
    videoLengths=[]  
    for videoName in os.listdir('inputVideos'):
        cap = cv2.VideoCapture('inputVideos/{}'.format(videoName))
        print(videoName)
        videoLengths.append(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS))
    return videoLengths

def clearFolderContent(folderName):
    for f in os.listdir('outputImages/{}'.format(folderName)):
        os.remove('outputImages/{}/{}'.format(folderName,f))

def updateLogAndSaveImg(folderPath,img,title="Untitled",accuracyPercentage="xx"):
    
    logWrite((int(getLastImgNumber())+1),'fileNumberLog.txt')
    if accuracyPercentage=="xx":
        appendErrorLog("{0:50}Img Number: {1:4}".format(title,getLastImgNumber(),accuracyPercentage))
        cv2.imwrite('{}/{}{}_frame{}.jpg'.format(folderPath,title,getLastImgNumber(),frameCnt),img)
    else:
        appendErrorLog("{0:50}Img Number: {1:15}Accuracy: %{2:4}".format(title,getLastImgNumber(),accuracyPercentage))
        cv2.imwrite('{}/{}{}_frame{}_%{}.jpg'.format(folderPath,title,getLastImgNumber(),frameCnt,accuracyPercentage),img)

#Encode testers
print("Tester encoding started!")
encodedTesters = encodeTesters()
print("Tester encoding completed!")

#Load Searcher
sfr = SimpleFacerec()
sfr.load_encoding_images("TargetImagesToSearch/")
print("Started to capture ..!")

#initial states of  logs
logWrite("","faceDetectAccuracyLog.txt");logWrite("0","fileNumberLog.txt")

#Clear selected folders
if clearFolderContentChoice:
    clearFolderContent('target');clearFolderContent('noise');clearFolderContent('imgsWithoutAnyFace');
    clearFolderContent('otherFaces');clearFolderContent('allFrames');clearFolderContent('imgsWithAnyFace')

#Calculate weights
weightedDistributionOfWantedLengths=[]
videoLengths=getVideoLengths()
for lengths in videoLengths:
    #calculate Distribution weights
    weightedDistributionOfWantedLengths.append(
            int(lengths*wantedFrameNum/sum(videoLengths)) if int(lengths*wantedFrameNum/sum(videoLengths))>0 else 1)

#Adjust same weights
wantedFrameNum/=len(os.listdir('inputVideos')) if wantedFrameNum > len(os.listdir('inputVideos')) else wantedFrameNum

#If distribution Choice == 1 , adjust distr to apply weighted version
if distributionChoice == 1:
    #overwrite weights due to set same weights to all videos
    weightedDistributionOfWantedLengths=[math.ceil(wantedFrameNum)]*len(os.listdir('inputVideos'))

for index,videoName in enumerate(os.listdir('inputVideos')):
    
    #Setups to search faces in video
    frameCursor=0

    #load the video
    cap = cv2.VideoCapture('inputVideos/{}'.format(videoName))
    #distrube the weights to set scale factor
    wantedFrameNum=weightedDistributionOfWantedLengths[index]
    
    #get video attributes
    videdoWidth  = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    videoDuration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
    
    wantedFrameNum = 1 if wantedFrameNum>cap.get(cv2.CAP_PROP_FRAME_COUNT) else wantedFrameNum
    frameScaler=math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)/(wantedFrameNum)) #-> it looks frames time: frame/constant
    frameScaler=random.randint(math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT)/1.9),cap.get(cv2.CAP_PROP_FRAME_COUNT)) if wantedFrameNum==1 else frameScaler

    localTimer=time.time()


    #Starts searching
    while(True):
        frameCursor+=1
        
        #cursor * scaler -> if there is 18000frame in video -> for example, scaler is 500 ==> frameCursor would be 500,1000,1500 ... 18000
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameCursor*frameScaler-1)
        ret, frame = cap.read()
        
        if ret == True and frameCursor*frameScaler-1<cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frameCnt+=1
            cv2.imwrite('outputImages/allFrames/frame{}_{}.jpg'.format(frameCnt,getLastImgNumber()),frame)
            progressPercentage =(cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT))*100
            passedTıme=math.floor(time.time()-localTimer)
            infoTexts.clear()
                        
            #searchs faces in the image
            face_locations, face_names =sfr.detect_known_faces(frame) 
            
            for face_loc, face_names in zip(face_locations, face_names):
                #crop image due to face
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                [x1,y1,x2,y2]=offsetCrop([x1,y1,x2,y2],math.floor((x2-x1)/cropOffsetDivider))
                croppedImage= frame[y1:y2,x1:x2]
                
                #if there is this face in the TargetImagesToSearch file
                if face_names!="Unknown":

                    #simple test due to TargetImagesToTest folder
                    testResult=simpleTest(croppedImage,encodedTesters) # result is in type of %
                    if type(testResult)!=type("String") and testResult>accuracyLimit:
                        updateLogAndSaveImg(
                            folderPath='outputImages/target',
                            img=croppedImage,
                            title="target",
                            accuracyPercentage=testResult)
                        infoTexts.append("Target")
                        targetImgCnt+=1
                        break
                    else:
                        updateLogAndSaveImg(
                            folderPath='outputImages/noise',
                            img=croppedImage,
                            title="noise",
                            accuracyPercentage=testResult)
                        infoTexts.append("Noise")
                        noiseImgCnt+=1   
                else:
                    updateLogAndSaveImg(
                        folderPath='outputImages/otherFaces',
                        img=croppedImage,
                        title="otherFace")
                    infoTexts.append("Unknown")
                    unknownFaceCnt+=1
                            
            #neither target face nor noise face
            if len(face_locations)==0:
                zeroFaceCnt+=1
                updateLogAndSaveImg(
                    folderPath='outputImages/imgsWithoutAnyFace',
                    img=frame,
                    title="imgWithoutAnyFace"
                )
                infoTexts.append("There is no face")
            else:
                imgsWithAnyFace+=1
                updateLogAndSaveImg(
                    folderPath='outputImages/imgsWithAnyFace',
                    img=frame,
                    title="imgWithAnyFace"
                )


            #Information about process                
            os.system("cls")
            print("""{0:<3} - Total Progress Percentage: %{1:5} , Progress Percentage of Video: %{2:5} , Video Length:{3:2}h:{4:2}m:{5:2}s , frameScaler: {6:<5}, {7} frames will be examined\n
Video Name: {8}  ,  Frame: {9}  ,  Result: {10}  ,  Time passed: {11}min {12}sec  \n
# of Saved Images: {13}  ,  # of Noise Images: {14}  ,  # of Undetected face: {15}  ,  # of other faces: {16} ,  # of examined frames: {17}
""".format(
                index+1,round(((sum(weightedDistributionOfWantedLengths[0:index])*100)+progressPercentage*weightedDistributionOfWantedLengths[index])/holdWantedFrameNum,2),
                round(progressPercentage,2),int(videoDuration/(60*60)),int((videoDuration/(60))%60),int(videoDuration%60),
                round(frameScaler),wantedFrameNum,videoName,frameScaler*frameCursor,infoTexts,
                int(int(time.time()-globalTımer)/60),int(int(time.time()-globalTımer)%60),targetImgCnt,noiseImgCnt,
                zeroFaceCnt,unknownFaceCnt,frameCnt))
                
        else: 
            #searching is over
            if index+1>= len(os.listdir('inputVideos')):
                os.system("cls")
                print("""%100 Completed , Time passed: {}min {}sec  ,  # of Target Images: {}  ,  # of Noise Images: {}  ,  # of Unknown faces: {}
\n# of examined  frames: {}  ,  # of images with any face: {}  ,  # of images without any face: {}  ,  # of input video: {}

""".format(
                int(int(time.time()-globalTımer)/60),int(int(time.time()-globalTımer)%60),targetImgCnt,noiseImgCnt,
                unknownFaceCnt,frameCnt,imgsWithAnyFace,zeroFaceCnt,index+1))
            break
        #To create folder names
