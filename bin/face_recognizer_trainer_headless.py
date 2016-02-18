#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
import sys

# Path to the Yale Dataset
path = sys.argv[1]
model_path = sys.argv[2]

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "/opt/face_recognizer/lib/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('sad.')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_cv2 = cv2.imread(image_path)
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        # Convert the image format into numpy array
        image = np.array(gray, 'uint8')
        # Get the label of the image
        employee = int(os.path.split(image_path)[1].split(".")[1])
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        if len(faces) > 1:
            faces = [ faces[-1] ]
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(employee)
            print("Added employee {} to recognized faces".format(employee))
    # return the images list and labels list
    return images, labels
 
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(path)
print("Now doing training")
# Perform the tranining
recognizer.train(images, np.array(labels))

recognizer.save(model_path+'/model.xml')

print("pretrained model saved to model.xml")
