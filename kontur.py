import numpy as np
import cv2
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics # for the check the error and accuracy of the model
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


im= cv2.imread('11.jpeg')
im1 = cv2.imread('22.jpeg')
im2 = cv2.imread('33.jpeg')
im3 = cv2.imread('44.jpeg')
im4 = cv2.imread('55.jpeg')
im5 = cv2.imread('66.jpeg')
im6 = cv2.imread('77.jpeg')
im7 = cv2.imread('99.jpeg')
im8 = cv2.imread('12.jpeg')
im9 = cv2.imread('13.jpeg')
im10 = cv2.imread('14.jpeg')
im11 = cv2.imread('15.jpeg')
im12 = cv2.imread('16.jpeg')
im13 = cv2.imread('17.jpeg')
im14 = cv2.imread('18.jpeg')
im15 = cv2.imread('19.jpeg')
im16 = cv2.imread('20.jpeg')
im17 = cv2.imread('21.jpeg')
im18 = cv2.imread('23.jpeg')
im19 = cv2.imread('24.jpeg')
im20 = cv2.imread('88.jpeg')



imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
imgray2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
imgray3 = cv2.cvtColor(im3,cv2.COLOR_BGR2GRAY)
imgray4 = cv2.cvtColor(im4,cv2.COLOR_BGR2GRAY)
imgray5 = cv2.cvtColor(im5,cv2.COLOR_BGR2GRAY)
imgray6 = cv2.cvtColor(im6,cv2.COLOR_BGR2GRAY)
imgray7 = cv2.cvtColor(im7,cv2.COLOR_BGR2GRAY)
imgray8 = cv2.cvtColor(im8,cv2.COLOR_BGR2GRAY)
imgray9 = cv2.cvtColor(im9,cv2.COLOR_BGR2GRAY)
imgray10 = cv2.cvtColor(im10,cv2.COLOR_BGR2GRAY)
imgray11 = cv2.cvtColor(im11,cv2.COLOR_BGR2GRAY)
imgray12 = cv2.cvtColor(im12,cv2.COLOR_BGR2GRAY)
imgray13 = cv2.cvtColor(im13,cv2.COLOR_BGR2GRAY)
imgray14 = cv2.cvtColor(im14,cv2.COLOR_BGR2GRAY)
imgray15 = cv2.cvtColor(im15,cv2.COLOR_BGR2GRAY)
imgray16 = cv2.cvtColor(im16,cv2.COLOR_BGR2GRAY)
imgray17 = cv2.cvtColor(im17,cv2.COLOR_BGR2GRAY)
imgray18 = cv2.cvtColor(im18,cv2.COLOR_BGR2GRAY)
imgray19 = cv2.cvtColor(im19,cv2.COLOR_BGR2GRAY)
imgray20 = cv2.cvtColor(im20,cv2.COLOR_BGR2GRAY)



_,thresh = cv2.threshold(imgray,140,255,4)
_,thresh1 = cv2.threshold(imgray1,140,255,4)
_,thresh2 = cv2.threshold(imgray2,140,255,4)
_,thresh3 = cv2.threshold(imgray3,140,255,4)
_,thresh4 = cv2.threshold(imgray4,140,255,4)
_,thresh5 = cv2.threshold(imgray5,140,255,4)
_,thresh6 = cv2.threshold(imgray6,140,255,4)
_,thresh7 = cv2.threshold(imgray7,140,255,4)
_,thresh8 = cv2.threshold(imgray8,140,255,4)
_,thresh9 = cv2.threshold(imgray9,140,255,4)
_,thresh10 = cv2.threshold(imgray10,140,255,4)
_,thresh11 = cv2.threshold(imgray11,140,255,4)
_,thresh12 = cv2.threshold(imgray12,140,255,4)
_,thresh13 = cv2.threshold(imgray13,140,255,4)
_,thresh14 = cv2.threshold(imgray14,140,255,4)
_,thresh15 = cv2.threshold(imgray15,140,255,4)
_,thresh16 = cv2.threshold(imgray16,140,255,4)
_,thresh17 = cv2.threshold(imgray17,140,255,4)
_,thresh18 = cv2.threshold(imgray18,140,255,4)
_,thresh19 = cv2.threshold(imgray19,140,255,4)
_,thresh20 = cv2.threshold(imgray20,140,255,4)


thresh = cv2.GaussianBlur(thresh,(5,5),2)
thresh1 = cv2.GaussianBlur(thresh1,(5,5),2)
thresh2 = cv2.GaussianBlur(thresh2,(5,5),2)
thresh3 = cv2.GaussianBlur(thresh3,(5,5),2)
thresh4 = cv2.GaussianBlur(thresh4,(5,5),2)
thresh5 = cv2.GaussianBlur(thresh5,(5,5),2)
thresh6 = cv2.GaussianBlur(thresh6,(5,5),2)
thresh7 = cv2.GaussianBlur(thresh7,(5,5),2)
thresh8 = cv2.GaussianBlur(thresh8,(5,5),2)
thresh9 = cv2.GaussianBlur(thresh9,(5,5),2)
thresh10 = cv2.GaussianBlur(thresh10,(5,5),2)
thresh11 = cv2.GaussianBlur(thresh11,(5,5),2)
thresh12 = cv2.GaussianBlur(thresh12,(5,5),2)
thresh13 = cv2.GaussianBlur(thresh13,(5,5),2)
thresh14 = cv2.GaussianBlur(thresh14,(5,5),2)
thresh15 = cv2.GaussianBlur(thresh15,(5,5),2)
thresh16 = cv2.GaussianBlur(thresh16,(5,5),2)
thresh17 = cv2.GaussianBlur(thresh17,(5,5),2)
thresh18 = cv2.GaussianBlur(thresh18,(5,5),2)
thresh19 = cv2.GaussianBlur(thresh19,(5,5),2)
thresh20 = cv2.GaussianBlur(thresh20,(5,5),2)

contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours1,_ = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours2,_ = cv2.findContours(thresh2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours3,_ = cv2.findContours(thresh3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours4,_ = cv2.findContours(thresh4,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours5,_ = cv2.findContours(thresh5,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours6,_ = cv2.findContours(thresh6,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours7,_ = cv2.findContours(thresh7,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours8,_ = cv2.findContours(thresh8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours9,_ = cv2.findContours(thresh9,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours10,_ = cv2.findContours(thresh10,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours11,_ = cv2.findContours(thresh11,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours12,_ = cv2.findContours(thresh12,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours13,_ = cv2.findContours(thresh13,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours14,_ = cv2.findContours(thresh14,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours15,_ = cv2.findContours(thresh15,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours16,_ = cv2.findContours(thresh16,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours17,_ = cv2.findContours(thresh17,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours18,_ = cv2.findContours(thresh18,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours19,_ = cv2.findContours(thresh19,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours20,_ = cv2.findContours(thresh20,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


img = cv2.drawContours(im, contours,-1, (255,0,0), 2)
img1 = cv2.drawContours(im1, contours1,-1, (255,0,0), 2)
img2 = cv2.drawContours(im2, contours2,-1, (255,0,0), 2)
img3 = cv2.drawContours(im3, contours3,-1, (255,0,0), 2)
img4 = cv2.drawContours(im4, contours4,-1, (255,0,0), 1)
img5 = cv2.drawContours(im5, contours5,-1, (255,0,0), 2)
img6 = cv2.drawContours(im6, contours6,-1, (255,0,0), 2)
img7 = cv2.drawContours(im7, contours7,-1, (255,0,0), 2)
img8 = cv2.drawContours(im8, contours8,-1, (255,0,0), 2)
img9 = cv2.drawContours(im9, contours9,-1, (255,0,0), 2)
img10 = cv2.drawContours(im10, contours10,-1, (255,0,0), 2)
img11 = cv2.drawContours(im11, contours11,-1, (255,0,0), 2)
img12 = cv2.drawContours(im12, contours12,-1, (255,0,0), 2)
img13 = cv2.drawContours(im13, contours13,-1, (255,0,0), 2)
img14 = cv2.drawContours(im14, contours14,-1, (255,0,0), 2)
img15 = cv2.drawContours(im15, contours15,-1, (255,0,0), 2)
img16 = cv2.drawContours(im16, contours16,-1, (255,0,0), 2)
img17 = cv2.drawContours(im17, contours17,-1, (255,0,0), 2)
img18 = cv2.drawContours(im18, contours18,-1, (255,0,0), 2)
img19 = cv2.drawContours(im19, contours19,-1, (255,0,0), 2)
img20 = cv2.drawContours(im20, contours20,-1, (255,0,0), 2)



cnt = contours[0]
area = cv2.contourArea(cnt)
cnt1 = contours1[0]
area1 = cv2.contourArea(cnt1)
cnt2 = contours2[0]
area2 = cv2.contourArea(cnt2)
cnt3 = contours3[0]
area3 = cv2.contourArea(cnt3)
cnt4 = contours4[0]
area4 = cv2.contourArea(cnt4)
cnt5 = contours5[0]
area5 = cv2.contourArea(cnt5)
cnt6 = contours6[0]
area6 = cv2.contourArea(cnt6)
cnt7 = contours7[0]
area7 = cv2.contourArea(cnt7)
cnt8 = contours8[0]
area8 = cv2.contourArea(cnt8)
cnt9 = contours9[0]
area9 = cv2.contourArea(cnt9)
cnt10 = contours11[0]
area10 = cv2.contourArea(cnt10)
cnt11 = contours11[0]
area11 = cv2.contourArea(cnt11)
cnt12 = contours12[0]
area12 = cv2.contourArea(cnt12)
cnt13 = contours13[0]
area13 = cv2.contourArea(cnt13)
cnt14 = contours14[0]
area14 = cv2.contourArea(cnt14)
cnt15 = contours15[0]
area15 = cv2.contourArea(cnt15)
cnt16 = contours16[0]
area16 = cv2.contourArea(cnt16)
cnt17 = contours17[0]
area17 = cv2.contourArea(cnt17)
cnt18 = contours18[0]
area18 = cv2.contourArea(cnt18)
cnt19 = contours19[0]
area19 = cv2.contourArea(cnt19)
cnt20 = contours20[0]
area20 = cv2.contourArea(cnt20)

mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)

mask1 = np.zeros(imgray1.shape,np.uint8)
cv2.drawContours(mask1,[cnt1],0,255,-1)

mask2 = np.zeros(imgray2.shape,np.uint8)
cv2.drawContours(mask2,[cnt2],0,255,-1)

mask3 = np.zeros(imgray3.shape,np.uint8)
cv2.drawContours(mask3,[cnt3],0,255,-1)


mask4 = np.zeros(imgray4.shape,np.uint8)
cv2.drawContours(mask4,[cnt4],0,255,-1)

mask5 = np.zeros(imgray5.shape,np.uint8)
cv2.drawContours(mask5,[cnt5],0,255,-1)

mask6 = np.zeros(imgray6.shape,np.uint8)
cv2.drawContours(mask6,[cnt6],0,255,-1)

mask7 = np.zeros(imgray7.shape,np.uint8)
cv2.drawContours(mask7,[cnt7],0,255,-1)

mask8 = np.zeros(imgray8.shape,np.uint8)
cv2.drawContours(mask8,[cnt8],0,255,-1)

mask9 = np.zeros(imgray9.shape,np.uint8)
cv2.drawContours(mask9,[cnt9],0,255,-1)

mask10 = np.zeros(imgray10.shape,np.uint8)
cv2.drawContours(mask10,[cnt10],0,255,-1)

mask11 = np.zeros(imgray11.shape,np.uint8)
cv2.drawContours(mask11,[cnt11],0,255,-1)

mask12 = np.zeros(imgray12.shape,np.uint8)
cv2.drawContours(mask12,[cnt12],0,255,-1)

mask13 = np.zeros(imgray13.shape,np.uint8)
cv2.drawContours(mask13,[cnt13],0,255,-1)

mask14 = np.zeros(imgray14.shape,np.uint8)
cv2.drawContours(mask14,[cnt14],0,255,-1)

mask15 = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask15,[cnt15],0,255,-1)

mask16 = np.zeros(imgray16.shape,np.uint8)
cv2.drawContours(mask16,[cnt16],0,255,-1)

mask17 = np.zeros(imgray17.shape,np.uint8)
cv2.drawContours(mask17,[cnt17],0,255,-1)

mask18 = np.zeros(imgray18.shape,np.uint8)
cv2.drawContours(mask18,[cnt18],0,255,-1)

mask19 = np.zeros(imgray19.shape,np.uint8)
cv2.drawContours(mask19,[cnt19],0,255,-1)

mask20 = np.zeros(imgray20.shape,np.uint8)
cv2.drawContours(mask20,[cnt20],0,255,-1)


b,g,r,_=np.uint8(cv2.mean(im,mask = mask))
color=cv2.cvtColor(np.uint8([[[b,g,r]]]),cv2.COLOR_BGR2HSV)
mean_val= color[0][0][0]

b1,g1,r1,_=np.uint8(cv2.mean(im1,mask = mask1))
color1=cv2.cvtColor(np.uint8([[[b1,g1,r1]]]),cv2.COLOR_BGR2HSV)
mean_val1= color1[0][0][0]

b2,g2,r2,_=np.uint8(cv2.mean(im2,mask = mask2))
color2=cv2.cvtColor(np.uint8([[[b2,g2,r2]]]),cv2.COLOR_BGR2HSV)
mean_val2= color2[0][0][0]

b3,g3,r3,_=np.uint8(cv2.mean(im3,mask = mask3))
color3=cv2.cvtColor(np.uint8([[[b3,g3,r3]]]),cv2.COLOR_BGR2HSV)
mean_val3= color3[0][0][0]

b4,g4,r4,_=np.uint8(cv2.mean(im4,mask = mask4))
color4=cv2.cvtColor(np.uint8([[[b4,g4,r4]]]),cv2.COLOR_BGR2HSV)
mean_val4= color4[0][0][0]

b5,g5,r5,_=np.uint8(cv2.mean(im5,mask = mask5))
color5=cv2.cvtColor(np.uint8([[[b5,g5,r5]]]),cv2.COLOR_BGR2HSV)
mean_val5= color5[0][0][0]

b6,g6,r6,_=np.uint8(cv2.mean(im6,mask = mask6))
color6=cv2.cvtColor(np.uint8([[[b6,g6,r6]]]),cv2.COLOR_BGR2HSV)
mean_val6= color6[0][0][0]

b7,g7,r7,_=np.uint8(cv2.mean(im7,mask = mask7))
color7=cv2.cvtColor(np.uint8([[[b7,g7,r7]]]),cv2.COLOR_BGR2HSV)
mean_val7= color7[0][0][0]

b8,g8,r8,_=np.uint8(cv2.mean(im8,mask = mask8))
color8=cv2.cvtColor(np.uint8([[[b8,g8,r8]]]),cv2.COLOR_BGR2HSV)
mean_val8= color8[0][0][0]

b9,g9,r9,_=np.uint8(cv2.mean(im9,mask = mask9))
color9=cv2.cvtColor(np.uint8([[[b9,g9,r9]]]),cv2.COLOR_BGR2HSV)
mean_val9= color9[0][0][0]

b10,g10,r10,_=np.uint8(cv2.mean(im10,mask = mask10))
color10=cv2.cvtColor(np.uint8([[[b10,g10,r10]]]),cv2.COLOR_BGR2HSV)
mean_val10= color10[0][0][0]

b11,g11,r11,_=np.uint8(cv2.mean(im11,mask = mask11))
color11=cv2.cvtColor(np.uint8([[[b11,g11,r11]]]),cv2.COLOR_BGR2HSV)
mean_val11= color11[0][0][0]

b12,g12,r12,_=np.uint8(cv2.mean(im12,mask = mask12))
color12=cv2.cvtColor(np.uint8([[[b12,g12,r12]]]),cv2.COLOR_BGR2HSV)
mean_val12= color12[0][0][0]

b13,g13,r13,_=np.uint8(cv2.mean(im,mask = mask))
color13=cv2.cvtColor(np.uint8([[[b13,g13,r13]]]),cv2.COLOR_BGR2HSV)
mean_val13= color13[0][0][0]

b14,g14,r14,_=np.uint8(cv2.mean(im14,mask = mask14))
color14=cv2.cvtColor(np.uint8([[[b14,g14,r14]]]),cv2.COLOR_BGR2HSV)
mean_val14= color14[0][0][0]
mean_val15= 4
b16,g16,r16,_=np.uint8(cv2.mean(im16,mask = mask16))
color16=cv2.cvtColor(np.uint8([[[b16,g16,r16]]]),cv2.COLOR_BGR2HSV)
mean_val16= color16[0][0][0]

b17,g17,r17,_=np.uint8(cv2.mean(im17,mask = mask17))
color17=cv2.cvtColor(np.uint8([[[b17,g17,r17]]]),cv2.COLOR_BGR2HSV)
mean_val17= color17[0][0][0]

b18,g18,r18,_=np.uint8(cv2.mean(im18,mask = mask18))
color18=cv2.cvtColor(np.uint8([[[b18,g18,r18]]]),cv2.COLOR_BGR2HSV)
mean_val18= color18[0][0][0]

b19,g19,r19,_=np.uint8(cv2.mean(im19,mask = mask19))
color19=cv2.cvtColor(np.uint8([[[b19,g19,r19]]]),cv2.COLOR_BGR2HSV)
mean_val19= color19[0][0][0]

b20,g20,r20,_=np.uint8(cv2.mean(im20,mask = mask20))
color20=cv2.cvtColor(np.uint8([[[b20,g20,r20]]]),cv2.COLOR_BGR2HSV)
mean_val20= color20[0][0][0]


x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h

x1,y1,w1,h1 = cv2.boundingRect(cnt1)
aspect_ratio1 = float(w1)/h1

x2,y2,w2,h2 = cv2.boundingRect(cnt2)
aspect_ratio2 = float(w2)/h2

x3,y3,w3,h3 = cv2.boundingRect(cnt3)
aspect_ratio3 = float(w3)/h3

x4,y4,w4,h4 = cv2.boundingRect(cnt4)
aspect_ratio4 = float(w4)/h4

x5,y5,w5,h5 = cv2.boundingRect(cnt5)
aspect_ratio5 = float(w5)/h5

x6,y6,w6,h6 = cv2.boundingRect(cnt)
aspect_ratio6 = float(w6)/h6

x7,y7,w7,h7 = cv2.boundingRect(cnt7)
aspect_ratio7 = float(w7)/h7

x8,y8,w8,h8 = cv2.boundingRect(cnt8)
aspect_ratio8 = float(w8)/h8

x9,y9,w9,h9 = cv2.boundingRect(cnt9)
aspect_ratio9 = float(w9)/h9

x10,y10,w10,h10 = cv2.boundingRect(cnt10)
aspect_ratio10 = float(w10)/h10

x11,y11,w11,h11 = cv2.boundingRect(cnt11)
aspect_ratio11 = float(w11)/h11

x12,y12,w12,h12 = cv2.boundingRect(cnt12)
aspect_ratio12 = float(w12)/h12

x13,y13,w13,h13 = cv2.boundingRect(cnt13)
aspect_ratio13 = float(w13)/h13

x14,y14,w14,h14 = cv2.boundingRect(cnt14)
aspect_ratio14 = float(w14)/h14

x15,y15,w15,h15 = cv2.boundingRect(cnt15)
aspect_ratio15 = float(w)/h15

x16,y16,w16,h16 = cv2.boundingRect(cnt16)
aspect_ratio16 = float(w16)/h16

x17,y17,w17,h17 = cv2.boundingRect(cnt17)
aspect_ratio17 = float(w17)/h17

x18,y18,w18,h18 = cv2.boundingRect(cnt18)
aspect_ratio18 = float(w18)/h18

x19,y19,w19,h19 = cv2.boundingRect(cnt19)
aspect_ratio19 = float(w19)/h19

x20,y20,w20,h20 = cv2.boundingRect(cnt20)
aspect_ratio20 = float(w20)/h20

rect_area = w*h
extent = float(area)/rect_area
rect_area1 = w1*h1
extent1 = float(area1)/rect_area1
rect_area2 = w2*h2
extent2 = float(area2)/rect_area2
rect_area3 = w3*h3
extent3 = float(area3)/rect_area3
rect_area4 = w4*h4
extent4 = float(area4)/rect_area4
rect_area5 = w5*h5
extent5 = float(area5)/rect_area5
rect_area6 = w6*h6
extent6 = float(area6)/rect_area6
rect_area7 = w7*h7
extent7 = float(area7)/rect_area7
rect_area8 = w8*h8
extent8 = float(area8)/rect_area8
rect_area9 = w9*h9
extent9 = float(area9)/rect_area9
rect_area10 = w10*h10
extent10 = float(area10)/rect_area10
rect_area11 = w11*h11
extent11 = float(area11)/rect_area11
rect_area12 = w12*h12
extent12 = float(area12)/rect_area12
rect_area13 = w13*h13
extent13 = float(area13)/rect_area13
rect_area14 = w14*h14
extent14 = float(area14)/rect_area14
rect_area15 = w1*h15
extent15 = float(area15)/rect_area15
rect_area16 = w1*h16
extent16 = float(area16)/rect_area16
rect_area17 = w17*h17
extent17 = float(area17)/rect_area17
rect_area18 = w18*h18
extent18 = float(area18)/rect_area18
rect_area19 = w19*h19
extent19 = float(area19)/rect_area19
rect_area20 = w20*h20
extent20 = float(area20)/rect_area20

hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area

hull1 = cv2.convexHull(cnt1)
hull_area1 = cv2.contourArea(hull1)
solidity1 = float(area1)/hull_area1

hull2 = cv2.convexHull(cnt2)
hull_area2 = cv2.contourArea(hull2)
solidity2 = float(area2)/hull_area2

hull3 = cv2.convexHull(cnt3)
hull_area3 = cv2.contourArea(hull3)
solidity3 = float(area3)/hull_area3

hull4 = cv2.convexHull(cnt4)
hull_area4 = cv2.contourArea(hull4)
solidity4 = float(area4)/hull_area4

hull5 = cv2.convexHull(cnt5)
hull_area5 = cv2.contourArea(hull5)
solidity5 = float(area5)/hull_area5

hull6 = cv2.convexHull(cnt6)
hull_area6 = cv2.contourArea(hull6)
solidity6 = float(area6)/hull_area6

hull7 = cv2.convexHull(cnt7)
hull_area7 = cv2.contourArea(hull7)
solidity7 = float(area7)/hull_area7

hull8 = cv2.convexHull(cnt8)
hull_area8 = cv2.contourArea(hull8)
solidity8 = float(area8)/hull_area8

hull9 = cv2.convexHull(cnt9)
hull_area9 = cv2.contourArea(hull9)
solidity9 = float(area9)/hull_area9

hull10 = cv2.convexHull(cnt10)
hull_area10 = cv2.contourArea(hull10)
solidity10 = float(area10)/hull_area10

hull11 = cv2.convexHull(cnt11)
hull_area11 = cv2.contourArea(hull11)
solidity11 = float(area11)/hull_area11

hull12 = cv2.convexHull(cnt12)
hull_area12 = cv2.contourArea(hull12)
solidity12 = float(area12)/hull_area12

hull13 = cv2.convexHull(cnt13)
hull_area13 = cv2.contourArea(hull13)
solidity13 = float(area13)/hull_area13

hull14 = cv2.convexHull(cnt14)
hull_area14 = cv2.contourArea(hull14)
solidity14 = float(area14)/hull_area14

hull15 = cv2.convexHull(cnt15)
hull_area15 = cv2.contourArea(hull15)
solidity15 = float(area15)/hull_area15

hull16 = cv2.convexHull(cnt16)
hull_area16 = cv2.contourArea(hull16)
solidity16 = float(area16)/hull_area16

hull17 = cv2.convexHull(cnt17)
hull_area17 = cv2.contourArea(hull17)
solidity17 = float(area17)/hull_area17

hull18 = cv2.convexHull(cnt18)
hull_area18 = cv2.contourArea(hull18)
solidity18 = float(area18)/hull_area18

hull19 = cv2.convexHull(cnt19)
hull_area19 = cv2.contourArea(hull19)
solidity19 = float(area19)/hull_area19

hull20 = cv2.convexHull(cnt20)
hull_area20 = cv2.contourArea(hull20)
solidity20 = float(area20)/hull_area20


liste = ["x","y","w","h","area","rgb","aspect_ratio","extent","hull_area","solidity","name"]
liste_x = [x,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20]
liste_y = [y,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20]
liste_w = [w,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20]
liste_h = [h,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20]
liste_area =[area,area1,area2,area3,area4,area5,area6,area7,area8,area9,area10,area11,area12,area13,
             area14,area15,area16,area17,area18,area19,area20]
liste_rgb=[mean_val,mean_val1,mean_val2,mean_val3,mean_val4,mean_val5,mean_val6,mean_val7,mean_val8,mean_val9,
           mean_val10,mean_val11,mean_val12,mean_val13,mean_val14,mean_val15,mean_val16,mean_val17,mean_val18,mean_val19,mean_val20]
liste_aspect=[aspect_ratio,aspect_ratio1,aspect_ratio2,aspect_ratio3,aspect_ratio4,aspect_ratio5,aspect_ratio6,aspect_ratio7,aspect_ratio8,aspect_ratio9,
              aspect_ratio10,aspect_ratio11,aspect_ratio12,aspect_ratio13,aspect_ratio14,aspect_ratio15,aspect_ratio16,aspect_ratio17,aspect_ratio18,
              aspect_ratio19,aspect_ratio20]
liste_extent =[extent,extent1,extent2,extent3,extent4,extent5,extent6,extent7,extent8,extent9,extent10,extent11,extent12,extent13,extent14,
               extent15,extent16,extent17,extent18,extent19,extent20]
liste_hull =[hull_area,hull_area1,hull_area2,hull_area3,hull_area4,hull_area5,hull_area6,hull_area7,hull_area8,hull_area9,hull_area10,hull_area11,
             hull_area12,hull_area13,hull_area14,hull_area15,hull_area16,hull_area17,hull_area18,hull_area19,hull_area20]
liste_solidity=[solidity,solidity1,solidity2,solidity3,solidity4,solidity5,solidity6,solidity7,solidity8,solidity9,solidity10,solidity11,solidity12,solidity13,
                solidity14,solidity15,solidity16,solidity17,solidity18,solidity19,solidity20]
liste_name=["1","2","3","3","2","2","1","1","2","3","1","2","1","1","3","2","1","3","1","2","3"]
dataset = pd.DataFrame({liste[0]:liste_x , liste[1]:liste_y , liste[2]:liste_w , liste[3]:liste_h,
                        liste[4]:liste_area , liste[5]:liste_rgb , liste[6]:liste_aspect , 
                        liste[7]:liste_extent , liste[8]:liste_hull , liste[9]:liste_solidity , liste[10]:liste_name })

#dataset.to_csv (r'C:\Users\murat\Desktop\opencv\domates.csv', index = False, header=True)

#dataset=pd.read_csv("domates.csv",sep=",")
dataset.info()
dataset.describe()
sns.countplot(dataset['name'],label="Count")
features_mean= list(dataset.columns[0:11])
corr = dataset[features_mean].corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm')



prediction_var = ['x','y','w','area','rgb','aspect_ratio','extent','solidity']

train, test = train_test_split(dataset, test_size = 0.5)

train_X = train[prediction_var] #eğitim için   girişi
train_y=train.name #eğitim için çıktılar

test_X= test[prediction_var] #test için veri girişi
test_y =test.name #test için çıktılar


model=RandomForestClassifier(n_estimators=5)
model.fit(train_X,train_y) #veri izleme modelimize uyuyor random forest classsifier 

prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
print ("RandomForestClassifier <<<<",metrics.accuracy_score(prediction,test_y))
#SVM de deneyelim

model = svm.SVC(gamma='auto')
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
print ("SVM <<<<",metrics.accuracy_score(prediction,test_y))


prediction_var = features_mean #Tüm özellikler için yapalım
train_X= train[prediction_var]
train_y= train.name
test_X = test[prediction_var]
test_y = test.name

model=RandomForestClassifier(n_estimators=5)

model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
print ("RandomForestClassifier <<<<",metrics.accuracy_score(prediction,test_y))

#tüm özelllikleri kullandığımızda özelliklerin eğitime etkisini görmek için degerleri sıraladık.
#Random forest kullandığımızda nasıl oldugunu gördük
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)

#şimdi tüm özellikleri svm kullandıgımızda nasıl bi dogruluk degeri alacagımızı görelim.
model = svm.SVC( gamma='auto')
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
print ("SVM <<<<",metrics.accuracy_score(prediction,test_y))

def classification_model(model,dataset,prediction_input,output):
    model.fit(dataset[prediction_input],dataset[output].values.ravel()) 
  

    predictions = model.predict(dataset[prediction_input])
  
   
    accuracy = metrics.accuracy_score(predictions,dataset[output])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    kf = KFold(n_splits=5)
    
    error = []
    for train, test in kf.split(prediction_input):
        
        train_X = dataset[prediction_input].iloc[train,:]
       
        train_y = dataset[output].iloc[train].values.ravel()
       
        model.fit(train_X, train_y)
    
        
        test_X=dataset[prediction_input].iloc[test,:]
        test_y=dataset[output].iloc[test]
        error.append(model.score(test_X,test_y))
        # printing the score 
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
        
model = DecisionTreeClassifier()
prediction_var = features_mean
outcome_var= ["name"]
        

classification_model(model,dataset,prediction_var,outcome_var)

model = svm.SVC()


classification_model(model,dataset,prediction_var,outcome_var)
        

model = KNeighborsClassifier()

classification_model(model,dataset,prediction_var,outcome_var)
        

model = RandomForestClassifier(n_estimators=5)

classification_model(model,dataset,prediction_var,outcome_var)
        

     
data_X= dataset[prediction_var]
data_y= dataset['name']
        
def Classification_model_gridsearchCV(model,param_grid,data_X,data_y):
    clf = GridSearchCV(model,param_grid,cv=2,scoring="accuracy")
    clf.fit(train_X,train_y)
    print("The best parameter found on development set is :")
            # this will gie us our best parameter to use
    print(clf.best_params_)
    print("the bset estimator is ")
    print(clf.best_estimator_)
    print("The best score is ")
            # this is the best score that we can achieve using these parameters#
    print(clf.best_score_)
    
param_grid = {'max_features':['auto', 'sqrt', 'log2'],
              'min_samples_split':[2,3,4,5,6], 
              'min_samples_leaf':[2,3,4,5,6] }
model= DecisionTreeClassifier()
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
            
model = KNeighborsClassifier()

k_range = list(range(1, 4))
leaf_size = list(range(1,4))
weight_options = ['uniform', 'distance']
param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)

model=svm.SVC(random_state=2)
param_grid = [
        {'C': [1,2,3,4,5], 
         'kernel': ['linear']
         },
        {'C': [1,2,3,4,5], 
         'gamma': [0.001, 0.0001], 
         'kernel': ['rbf']
         },
        ]
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
        

cv2.waitKey(0)

cv2.destroyAllWindows()