import os
import cv2
import time
from PIL import Image # Image process
from threading import Thread  # Used for the process Training
import numpy as np
#----------------- Image Labels ----------------------------#

def getImageAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print(imagePaths)    
    
    # Create empty face list
    faces = []
    # Create empty ID list
    ids = []
    # now looping through all the image paths 
    # and loading the ids and the images
    for imagePath in imagePaths:
        # Loading the image and converting it to Gray scale
        pillImage = Image.open(imagePath).convert('L')
        # Now Converting the PIL image into numpy Array
        imageNP = np.array(pillImage, 'unit8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # Extract the face from the training image sample
        faces.append(imageNP)
        ids.append(Id)
    return faces,ids

# --------------------- Train Image Function -------------------------#
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImageAndLabels("TrainingImage")
    Thread(target=recognizer.train(faces, np.array(Id))).start()
    # Below line is optimal for a visual counter effect
    Thread(target=counter_img("TrainingImage")).start()
    recognizer.save("TrainingImageLabel" +os.sep+"Training.yml")
    print("All Images")

# Optional, adds a counter for images trained ()
def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + "Images Trained", end = "\r")
        time.sleep(0.008)
        imgcounter += 1

