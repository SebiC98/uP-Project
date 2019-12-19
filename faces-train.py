import os
import numpy as np
from PIL import Image

baseDirectory = os.path.dirname(os.path.abspath(__file__)) #Where this file is saved we get the path.
imageDirectory = os.path.join(baseDirectory, "images") #Looking for the images folder.


y_labels= []
x_train = []

for root, directories, images in os.walk(imageDirectory):
	for image in images:
		if image.endswith("png") or image.endswith("jpg"):
			pathOfTheImage = os.path.join(root, image)
			label = os.path.basename(root).replace(" ","-").lower() #Making sure the folder names are written as aaa-bbb.
			print(label) #Printing the labels
			#y_labels.append(label) #some number
			#x_train.append(pathOfTheImage) #verify this image, turn into a NUMPY array, GRAY
			pilImage = Image.open(pathOfTheImage).convert("L") #convert into grayscale
			imageArray = np.array(pilImage, "uint8")
			print(imageArray)

 