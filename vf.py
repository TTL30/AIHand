import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import model_from_json
from collections import Counter
import warnings

cap = cv2.VideoCapture(0)

cv2.namedWindow("frame")
cv2.namedWindow("threshold")


x = 0
y = 0
w = 200
h = 200
model = 0
on = False

def nothing(x):
    print("[INFO]New Threshold value : ",x)
    pass

file1 = open("new.txt","r+")  
v = int(file1.read())

cv2.createTrackbar("Value", "threshold", v, 255, nothing)

def load_model():
    print("[INFO] Loading model...")
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/model.h5")
    print("[INFO] Loaded model from disk")
    return loaded_model

def make_prediction(image,frame, loaded_model):
    image_resize = cv2.resize(image, (64,64))
    gray_image = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
    gray_image_inverted = np.invert(gray_image)
    image_dim_inverted = np.expand_dims(gray_image_inverted, axis = 0)
    image_dim_inverted = np.expand_dims(image_dim_inverted, axis = 3)
    pred_inverted = loaded_model.predict(image_dim_inverted)
    pred_inverted_res = np.argmax(pred_inverted)
    return pred_inverted_res

def drawCenterMass(r): 
    N = 2 * r + 1
    for i in range(cX-N, cX+N):
        for j in range(cY-N, cY + N): 
            x = i - r - cX
            y = j - r - cY
            if x * x + y * y <= r * r + 1:
                if(i < 200 and j <200):
                    cpy[j-int((N)/2)][i-int(N/2)] = frame[j-int((N)/2)][i-int((N)/2)]
                else:
                    cpy[j-(j-200)-int((N)/2)][i-(i-200)-int(N/2)] = frame[j-(j-200)-int((N)/2)][i-(i-200)-int((N)/2)]
    cv2.circle(frame, (cX, cY),5,(255,255,0),-1)

                

eq = "Equation : "
eq2 = ""
data = "null"
redLower = (115,  104,  167)
redUpper = (178, 255, 247)
op = False
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
last = False
cX = 0
cY = 0
data_arr =[]
print("[INFO]INITIALISATION...")
print("[INFO]Threshold value : ",v)

while True:
    _, frame = cap.read()
    r = cv2.getTrackbarPos("Value", "threshold")

    with open ('new.txt','w') as f:
             f.write(str(r))

    roi = frame[x:x+w,y:y+h]
    cpy = roi.copy()

    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours2,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if (len(contours2) > 0):
        c = max(contours2, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        if((rect[0][0]>=520 and rect[0][0]<=620) and (rect[0][1]>=220 and rect[0][1]<=270) and on == False):
                print("[INFO] Calculatrice ON...")
                on = True
        if(on):
            if((rect[0][0]>=20 and rect[0][0]<=120) and (rect[0][1]>=220 and rect[0][1]<=270)):
                if(op):
                    if(last):
                        eq = eq[:-1]
                    else:
                        last = True
                    eq += "+"

            if((rect[0][0]>=20 and rect[0][0]<=120) and (rect[0][1]>=290 and rect[0][1]<=340)):
                if(op):
                    if(last):
                        eq = eq[:-1]
                    else:
                        last = True 
                    eq += "-"

            if((rect[0][0]>=20 and rect[0][0]<=120) and (rect[0][1]>=360 and rect[0][1]<=410)):
                if(op):
                    if(last):
                        eq = eq[:-1]
                    else:
                        last = True
                    eq += "*"

            if((rect[0][0]>=520 and rect[0][0]<=620) and (rect[0][1]>=360 and rect[0][1]<=410)):
                if(op):
                    if(len(eq)>11):
                        str_of_ints = "".join(eq[11:])
                        res = eval(str_of_ints)
                        eq2 = " = " + str(res)
                        op = False

            if((rect[0][0]>=520 and rect[0][0]<=620) and (rect[0][1]>=290 and rect[0][1]<=340)):
                if(len(eq)>11):
                    eq = "Equation : "
                    eq2 = ""
                    print("[INFO] Reset...")
            


    
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    

    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray_image,r,255,4)
    ret2,thresh2 = cv2.threshold(gray_image,r,255,0)


    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    big_contour = max(contours, key=cv2.contourArea)
    if(big_contour.shape[0]<50):
        data_arr.clear()
        cv2.putText(frame, "Waiting for your hand", (8, 25) , cv2.FONT_HERSHEY_SIMPLEX,  
                   0.5, (255, 0, 0) , 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Predicted : " + str(data), (10, 25) , cv2.FONT_HERSHEY_SIMPLEX,  
                   0.7, (0, 0, 255) , 1, cv2.LINE_AA)

    cv2.drawContours(cpy, [big_contour], -1, (0,0,0), -1)
    
    cv2.rectangle(frame, (20, 420), (620, 470), (255, 255, 255), -1)
    cv2.putText(frame, eq, (40, 455) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)
    cv2.putText(frame, eq2, (460, 455) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)

    #Button +
    cv2.rectangle(frame, (20, 220), (120, 270), (255, 255, 255), -1)
    cv2.putText(frame, "+", (60, 255) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)
    
    #Button -
    cv2.rectangle(frame, (20, 290), (120, 340), (255, 255, 255), -1)
    cv2.putText(frame, "-", (59, 325) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)


    #Button X
    cv2.rectangle(frame, (20, 360), (120, 410), (255, 255, 255), -1)
    cv2.putText(frame, "x", (62, 391) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)

    #Button =
    cv2.rectangle(frame, (520, 360), (620, 410), (255, 255, 255), -1)
    cv2.putText(frame, "=", (557, 395) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)

    #Button AC
    cv2.rectangle(frame, (520, 290), (620, 340), (255, 255, 255), -1)
    cv2.putText(frame, "AC", (552, 325) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)
    
    cv2.rectangle(frame, (520, 220), (620, 270), (255, 255, 255), -1)
    cv2.putText(frame, "ON", (550, 255) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)


    M = cv2.moments(thresh2)
    if( M["m00"] != 0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    if(big_contour.shape[0]>50):
        drawCenterMass(30) 

    cv2.drawContours(frame, [big_contour], -1, (0,255,0), 1)

    cv2.imshow("frame", frame)
    cv2.imshow("threshold", thresh)
    cv2.imshow("Background Removed", cpy)
    cv2.imshow("hsv", hsv)
    cv2.imshow("hsv mask", mask)

    if(model and big_contour.shape[0]>50):
        data = make_prediction(cpy,frame,model)
        if(len(data_arr)<25):
            data_arr.append(data)
        else:
            p = Counter(data_arr)
            if(p.most_common(1)[0][1] > 20 and on):
                print("[INFO] Prediction validé : ",p.most_common(1)[0][0])
                eq +=  str(p.most_common(1)[0][0])
                print("[INFO]",eq)
                op = True
                last = False
            data_arr.clear()

    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord("l"):
        model = load_model()

cap.release()
cv2.destroyAllWindows()
