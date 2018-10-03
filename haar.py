import numpy as np 
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

img = cv2.imread('banda.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
i = 0 
im = []
for (x,y,w,h) in faces:
	im.append(img[y:y+h, x:x+w])
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]

for i in im :
	cv2.imshow('img',i)
	cv2.waitKey(10000)
	# cv2.destroyAllWindows()
cv2.destroyAllWindows()	