import cv2
import os
import numpy as np
import faceRecognition as fr

test_img = cv2.imread('C:/Machine_Learning_Projects/Face Recognition/Test Images/test.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)
print("FACE DETECTED: ", faces_detected)

#only run the below 3 lines if running program for the first time
# faces, faceID = fr.labels_for_training_images('C:/Machine_Learning_Projects/Face Recognition/TrainingImages')
# model = fr.train_classifier( faces, faceID )
# model.write('C:/Machine_Learning_Projects/Face Recognition/trainingData.yml')

#comment below line if running code for first time
model = cv2.face.LBPHFaceRecognizer_create()
model.read('C:/Machine_Learning_Projects/Face Recognition/trainingData.yml')

name = {0: 'Shivang'}

for face in faces_detected:
    ( x, y, w, h ) = face
    roi_gray = gray_img[ y:y+h, x:x+h ]
    label, confidence = model.predict(roi_gray)

    fr.draw_rect(test_img, face)
    predicted_name = name[label]

    # print('Confidence: ', confidence)

    # if confidence > 35:
    #     continue

    fr.put_text(test_img, predicted_name, x, y)

    print("Confidence: ", confidence)
    print("Label: ", label)

    # resized_image = cv2.resize(test_img, (500, 700))
    cv2.imshow('face detection', test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()