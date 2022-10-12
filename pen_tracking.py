import numpy as np
import cv2

def mouse(event,x,y,flags,params):
    global points,track_start,prevpoints
    if event==cv2.EVENT_LBUTTONDOWN:
        points=(x,y)
        track_start=True
        prevpoints=np.array([[x,y]],dtype='float32')

points=[]
track_start=False
prevpoints=np.array([[]])
featureparams=dict(winSize=(25,25),
                maxLevel=6,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
color=(0,255,0)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',mouse)    

video=cv2.VideoCapture(1)
ret,firstframe=video.read()
firstframegray=cv2.cvtColor(firstframe,cv2.COLOR_BGR2GRAY)

mask=np.zeros_like(firstframe)
period=1
while video.isOpened():
    success,frame=video.read()
    #frame=cv2.flip(frame,1)
    if success==True:
        grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if track_start==True:
            x,y=points
            #cv2.circle(frame,points,5,(0,0,255),-1)
            newpoints,status,error=cv2.calcOpticalFlowPyrLK(firstframegray,
                                    grayframe,prevpoints,None,**featureparams)
            a,b=newpoints.ravel()
            c,d=prevpoints.ravel()
            #cv2.circle(frame,(int(c),int(d)),10,(0,255,0),5)
            if period!=1:
                cv2.circle(frame,(int(a),int(b)),5,(255,0,0),-1)
                cv2.line(mask,(int(c),int(d)),(int(a),int(b)),color,3)
            prevpoints=newpoints
            firstframegray=grayframe
            period+=1
        frame=cv2.add(frame,mask)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        key=cv2.waitKey(27)
        if key==ord('q'):
            print('vidoe finish')
            break
        elif key==ord('w'):
            track_start=False
        elif key==ord('r'):
            color=(0,0,255)
        elif key==ord('b'):
            color=(255,0,0)
        elif key==ord('y'):
            color=(0,255,255)
        else :
            pass
    else:
        print('video can not read')
        break

video.release()
cv2.destroyAllWindows()