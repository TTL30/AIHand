import numpy as np
import cv2
import time 
from scipy import ndimage

from skimage import io, transform
from skimage.util import random_noise
import random

capture = cv2.VideoCapture(0)
x = 200
y = 200
w = 200
h = 200
var = 0

""" from keras import models, layers
import tensorflow as tf
model = models.load_model('model') """

def capt_data(thresh):
    steps = [120,120,120,10,10,10,60,60,10,10,10,10,10,10,0]
    n = 0
    next = time.time() + steps[0]
    title = "_Digit_4"
    now = time.time()
    if now >= next:
        n += 1
        next = now + steps[n]
        cv2.imwrite('dataset/'+title, thresh)
    time.sleep(1)


def center_image(img, frame):
    global var
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,190,255,0)
    M = cv2.moments(thresh)
    if(M["m00"] != 0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"]) 
        pos_x = x
        pos_y = y
        if (cX+x) < (x+(w/2)):
            pos_x = x-(100-cX)
        else:
            pos_x = x + (cX-(w/2))

        if (cY+y) < (y+(h/2)):
            pos_y = y-(100-cY)

        else:
            pos_y = y + (cY-(h/2))
        pos_x = int(pos_x)
        pos_y = int(pos_y)

        roi = frame[pos_y:pos_y+h,pos_x:pos_x+w]
        img_grey_center = transform.resize(roi, (128,128,1), mode = 'constant')
        
        ret2,thresh2 = cv2.threshold(img_grey_center,190,255,0)

        
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Roi recentered", img_grey_center)
        gray_image2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret2,thresh2 = cv2.threshold(gray_image2,200,255,0)
        cv2.imshow("nois", thresh2)
        if var == 0:
            capt_data(thresh2)
            var = 1
      

        
        """  data = img_grey_center_2/255.0
        data = np.expand_dims(data, axis=0)
        pred = model.predict(data)[0]
        pred2 = pred[:4]
        print(" is : ", np.argmax(pred))
         """
 

while(True):
    ret, frame = capture.read()
    cv2.imshow('frame', frame)
    roi = frame[y:y+h,x:x+w]
    cv2.imshow('roi', roi)

    center_image(roi, frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
 