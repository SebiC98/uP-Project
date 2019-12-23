import cv2
import os
import numpy as np
from PIL import Image
import pickle
print('started')
baseDirectory = os.path.dirname(os.path.abspath(__file__)) #Where this file is saved we get the path.
imageDirectory = os.path.join(baseDirectory, "images") #Looking for the images folder.

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

currentId = 0;
labelIds={}
y_labels= []
trainingData = []

for root, directories, images in os.walk(imageDirectory):
	for image in images:
		if image.endswith("png") or image.endswith("jpg"):
			pathOfTheImage = os.path.join(root, image)
			label = os.path.basename(root).replace(" ","-").lower() #Making sure the folder names are written as aaa-bbb.
			#print(label) #Printing the labels
			if not label in labelIds:  #Create an id for each person
				labelIds[label]=currentId
				currentId+=1
			id_ = labelIds[label]
			#print(labelIds)
			#y_labels.append(label) #some number
			#trainingData.append(pathOfTheImage) #verify this image, turn into a NUMPY array, GRAY
			pilImage = Image.open(pathOfTheImage).convert("L") #convert into grayscale
			#size = (550, 550)
			#finalImage = pilImage.resize(size, Image.ANTIALIAS) #Resize all images to the same dimension for a better accuracy.
			#imageArray = np.array(finalImage, "uint8") #convert into a NUMPY array. basically converting each picture into numbers. each pixel into a number.
			imageArray = np.array(pilImage, "uint8")
			#print(imageArray)
			faceFromImage = face_cascade.detectMultiScale(imageArray, scaleFactor=1.5, minNeighbors = 5) #We have to detect the face in each picture.

			for (x, y, width, height) in faceFromImage:
				regionOfInterestForGray = imageArray[y:y+height, x:x+width]
				trainingData.append(regionOfInterestForGray)
				y_labels.append(id_)

#print(trainingData)


with open("labels.pickle", 'wb') as f:
	pickle.dump(labelIds, f)


recognizer.train(trainingData, np.array(y_labels))
recognizer.save("trainner.yml")