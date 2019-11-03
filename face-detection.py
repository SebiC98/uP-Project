import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarascade_frontalface_defaul.xml')

cap =cv2.VideoCapture(0)

while(True):

	ret, frame = cap.read()

	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyallWindows()