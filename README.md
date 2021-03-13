# AIHand

## <-- Still in work -->

<em> A complete README will be done soon </em>

## Demo

![alt text](img/gi2.gif)


### First part detect Hand:

-   Capture 200x200 ROI from the frame
-   Capture the hand with center mass roi
-   Then center it in 128x128 px image

![alt text](img/capt_center_hand.PNG)

### Rework hand detection & digit prediction :

Detecting the hand on a part of the image.
Finding countour & center of mass with thresholding.
Remove the background.
Make prediction with loaded model. 
![alt text](img/hand.png)

### Maths operations :

Detecting red pen to choose the wanted operation

![alt text](img/pen.png)
