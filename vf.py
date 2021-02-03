import cv2
import numpy as np
from keras.models import model_from_json
from collections import Counter

cap = cv2.VideoCapture(0)

cv2.namedWindow("frame")
cv2.namedWindow("threshold")


x = 0
y = 0
w = 200
h = 200
model = 0

def nothing(x):
    pass

file1 = open("new.txt","r+")  
v = int(file1.read())

cv2.createTrackbar("R", "threshold", v, 255, nothing)

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

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
data_arr =[]
while True:
    _, frame = cap.read()

    r = cv2.getTrackbarPos("R", "threshold")
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
    if len(contours2) > 0:
        c = max(contours2, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        if((rect[0][0]>=50 and rect[0][0]<=150) and (rect[0][1]>=220 and rect[0][1]<=270)):
            if(op):
                if(last):
                    eq = eq[:-1]
                else:
                    last = True
                eq += "+"
        if((rect[0][0]>=50 and rect[0][0]<=150) and (rect[0][1]>=290 and rect[0][1]<=340)):
            if(op):
                if(last):
                    eq = eq[:-1]
                else:
                    last = True 
                eq += "-"
        if((rect[0][0]>=50 and rect[0][0]<=150) and (rect[0][1]>=360 and rect[0][1]<=410)):
            if(op):
                if(last):
                    eq = eq[:-1]
                else:
                    last = True
                eq += "*"

        if((rect[0][0]>=450 and rect[0][0]<=550) and (rect[0][1]>=360 and rect[0][1]<=410)):
            if(op):
                if(len(eq)>11):
                    str_of_ints = "".join(eq[11:])
                    res = eval(str_of_ints)
                    eq2 = " = " + str(res)
                    eq = "Equation : "
                    op = False

        if((rect[0][0]>=450 and rect[0][0]<=550) and (rect[0][1]>=290 and rect[0][1]<=340)):
            if(len(eq)>11):
                eq = eq[:-1]
                

    
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
    
    cv2.rectangle(frame, (50, 420), (550, 470), (255, 255, 255), -1)
    cv2.putText(frame, eq, (60, 455) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)
    cv2.putText(frame, eq2, (460, 455) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)

    cv2.rectangle(frame, (50, 220), (150, 270), (255, 255, 255), -1)
    cv2.putText(frame, "+", (90, 255) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)
    
    cv2.rectangle(frame, (50, 290), (150, 340), (255, 255, 255), -1)
    cv2.putText(frame, "-", (90, 315) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)

    cv2.rectangle(frame, (50, 360), (150, 410), (255, 255, 255), -1)
    cv2.putText(frame, "x", (90, 385) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)
    cv2.rectangle(frame, (450, 360), (550, 410), (255, 255, 255), -1)

    cv2.putText(frame, "=", (490, 385) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)

    cv2.rectangle(frame, (450, 290), (550, 340), (255, 255, 255), -1)
    cv2.putText(frame, "<=", (490, 315) , cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA)


    M = cv2.moments(thresh2)
    if( M["m00"] != 0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        drawCenterMass(30) 

    cv2.drawContours(frame, [big_contour], -1, (0,255,0), 1)
    out.write(frame)

    cv2.imshow("frame", frame)
    cv2.imshow("threshold", thresh)
    cv2.imshow("Background Removed", cpy)

    if(model and big_contour.shape[0]>50):
        data = make_prediction(cpy,frame,model)
        if(len(data_arr)<30):
            data_arr.append(data)
        else:
            p = Counter(data_arr)
            if(p.most_common(1)[0][1] > 20):
                eq +=  str(p.most_common(1)[0][0])
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
