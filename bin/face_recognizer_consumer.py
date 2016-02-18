#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
import sys

# Path to the image Dataset
image_path = sys.argv[1]
model_path = sys.argv[2]

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "/opt/face_recognizer/lib/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load(model_path)

predict_image_cv2 = cv2.imread(image_path)
predict_image = cv2.cvtColor(predict_image_cv2, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    predict_image,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )



for (x, y, w, h) in faces:
    employee_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
    print("The model believes you are {} with confidence {}".format(employee_predicted, conf))
    cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
    cv2.rectangle(predict_image,(x,y),(x+w,y+h))
    cv2.waitKey(1000)
cv2.imwrite("/data/images/result.jpg", predict_image)