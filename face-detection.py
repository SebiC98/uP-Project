import cv2 # Import Open CV
import numpy as np
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml") 

labels = {}

with open("labels.pickle", 'rb') as f:
	originalLabels = pickle.load(f)
	newLabels = {v:k for k,v in originalLabels.items()}


capturingImages =cv2.VideoCapture(0) # Start capturing images.

while(True):  #Start an infinite loop of capturing images.	

	#Capture frame by frame
	TrueOrFalse, capturedFrame = capturingImages.read() #The first returned value is a Boolean indicating  if the frame was read correctly (True) or not (False). I will not use it.
	capturedFrameConvertedToGray = cv2.cvtColor(capturedFrame, cv2.COLOR_BGR2GRAY) #Conver the frame to gray because this is how this cascade works
	facesFromCapturedFrameConvertedToGray = face_cascade.detectMultiScale(capturedFrameConvertedToGray, scaleFactor=1.5, minNeighbors = 5) #Values from documentation, I can modify them for better results, but this are defaults.
	for (x, y, width, height) in facesFromCapturedFrameConvertedToGray:
		#print(x, y, width, height)
		regionOfInterestForGray = capturedFrameConvertedToGray[y:y+height, x:x+width]
		regionOfInterestColor = capturedFrame[y:y+height, x:x+width]

		# Recognizer ? We can use deep lerned model predict
		id_, confidence = recognizer.predict(regionOfInterestForGray)
		if confidence >= 4 and confidence <=85:
			#print(id_)
			#print(newLabels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = newLabels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(capturedFrame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		testImage = '10.png'
		cv2.imwrite(testImage, regionOfInterestColor)

		rectangleColor = (0,255,255) #Declaring the color of the rectangle in BGR 0-255
		rectangleStroke = 2 #Rectangle stroke

		rectangleWidth=x+width
		rectangleHeight=y+height

		cv2.rectangle(capturedFrame, (x,y), (rectangleWidth, rectangleHeight), rectangleColor, rectangleStroke) # (x,y) starting coordinates (rectangleWidth, rectangle Height) ending coordinates

		

	#Display the frame
	cv2.imshow('frame',capturedFrame) #Display the frames in color but I am working with them in gray.

	if cv2.waitKey(1) == 27: #Close when pressing the ESC button.
		break

capturingImages.release() #Stop capturing images.
cv2.destroyallWindows()	#Close all windows.