import cv2 # Import Open CV

capturingImages =cv2.VideoCapture(0) # Start capturing images.

while(True):  #Start an infinite loop of capturing images.	

	#Capture frame by frame
														#the first returned value is a Boolean indicating 
	TrueOrFalse, capturedFrame = capturingImages.read() #if the frame was read correctly (True) or not (False).
													
	#Display the frame
	cv2.imshow('frame',capturedFrame)

	if cv2.waitKey(1) == 27: #Close when pressing the ESC button.
		break

capturingImages.release() #Stop capturing images.
cv2.destroyallWindows()	#Close all windows.