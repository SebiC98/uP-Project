import os

baseDirectory = os.path.dirname(os.path.abspath(__file__)) #Where this file is saved we get the path.
imageDirectory = os.path.join(baseDirectory, "images") #Looking for the images folder.


for root, directories, images in os.walk(imageDirectory):
	for image in images:
		if image.endswith("png") or image.endswith("jpg"):
			pathOfTheImage = os.path.join(root, image)
			label = os.path.basename(os.path.dirname(pathOfTheImage)).replace(" ","-").lower() #Making sure the folder names are written as aaa-bbb.
			print(label)

 