import cv2
import numpy as np
import math
from queue import PriorityQueue
#import matplotlib.pyplot as plt
arr = ['t1.jpg','t2.jpg','t3.jpg','t4.jpg','t5.jpg','t6.jpg','a1.png', 'a2.png', 'a3.png', 'a4.png', 'a5.png', 'a6.png', 'Screenshot.png', 'Screenshot2.png', 'Screenshot3.png', 'Screenshot5.png']

#arr = ['Screenshot3.png']







def process(imageName, ind):
    im1 = cv2.imread(imageName)
    im = cv2.cvtColor(im1, cv2.COLOR_BGR2LAB)
    #lt.subplot(5,4,ind), plt.imshow(im)
    mean, stddev = cv2.meanStdDev(im)
    mean = mean.flatten()
    stddev = stddev.flatten()
    cutoff = 2*stddev
    t1 = mean+cutoff
    t2 = mean-cutoff
    mask1 = cv2.inRange(im, t2, t1)
    mask1 = cv2.bitwise_not(mask1)
    #cv2.imshow("mask1",mask1)
    lab = cv2.bitwise_and(im1,im1,mask=mask1)
    #cv2.imshow("ttyp",lab)
    #lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)

    
    '''blurred = cv2.medianBlur(im,  9)

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    mean1 = cv2.mean(lab, mask=None)[:3]
    mean, stddev = cv2.meanStdDev(lab)
    

    mean = mean.flatten()
    stddev = stddev.flatten()

    cutoff = 2*stddev
    print(mean1)

    for (i, row) in enumerate(lab):
        for (j,cell) in enumerate(row):
            if (abs(cell - mean) < cutoff).all():
                lab[i][j] = np.array([0,125,125])
            else:
                lab[i][j] = np.array([255,125,125])
    img = cv2.cvtColor( lab, cv2.COLOR_LAB2BGR)'''
    imgray = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('b&w', img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('open', imgray)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    #cv2.imshow("temp",thresh)
    
    a,contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    alist = []
    

    for contour in contours:
                
                area = cv2.contourArea(contour)
                peri = cv2.arcLength(contour, True)
                if (area<1200 and area>100 and  peri< 140 and peri>40) :
                        alist.append(contour)
    q = []
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
    for contour in alist:
                x,y,w,h = cv2.boundingRect(contour)
                m , std = cv2.meanStdDev(lab[y:y+h,x:x+w])
                m= m.flatten()
                std = std.flatten()
                #temp = abs(m-[53,-37,78])
                temp = abs(m-mean)
                temp = math.sqrt(temp[0]*temp[0]+temp[1]*temp[1] + temp[2]*temp[2])
                print(temp)
                q.append((temp,contour))
    cout = 1
    i=0
    while(i<len(q)):
            j=i+1
            i+=1
            while(j<len(q)):
                    if(q[j-1][0]<q[j][0]):
                       ty = q[j]
                       q[j]=q[j-1]
                       q[j-1]=ty
                    j+=1
            
    #q.sort()
    
    flist =[]
    for ctuple in q:
            '''while not q.empty():
            d,next = q.get()
            #print(next.shape())'''
            if(cout%30!=0):
                    x,y,w,h = cv2.boundingRect(ctuple[1])
                    cv2.rectangle(im1, (x, y), (x + w, y + h), (0,255,0), 1)
                    cv2.putText(im1, str(ctuple[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
                    #cv2.drawContours(im1, flist, -1, (0,0,255), 2)
                    flist.append(ctuple[1])
                    cout+=1
                    
            
    #for cont in flist:
        
    #cv2.drawContours(im1, flist, -1, (0,0,255), 2)
    cv2.imwrite(imageName+"Processed.png" , im1)
    #plt.subplot(5,4,ind+1), plt.imshow(im)
    print(ind)
    
    
for (ind, i) in enumerate(arr):
    
        process(i, 2*(ind+1)-1)






        '''im = cv2.imread(imageName)
    #lt.subplot(5,4,ind), plt.imshow(im)
    blurred = cv2.medianBlur(im,  9)

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    mean1 = cv2.mean(lab, mask=None)[:3]
    mean, stddev = cv2.meanStdDev(lab)
    

    mean = mean.flatten()
    stddev = stddev.flatten()

    cutoff = 2*stddev
    print(mean1)

    for (i, row) in enumerate(lab):
        for (j,cell) in enumerate(row):
            if (abs(cell - mean) < cutoff).all():
                lab[i][j] = np.array([0,125,125])
            else:
                lab[i][j] = np.array([255,125,125])
    img = cv2.cvtColor( lab, cv2.COLOR_LAB2BGR)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('b&w', img)
    kernel = np.ones((9,9),np.uint8)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('open', imgray)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    a,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alist = []'''
    
