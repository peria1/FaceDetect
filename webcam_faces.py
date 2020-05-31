# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:22:10 2020

@author: Bill
"""

import cv2
import time

cap = cv2.VideoCapture(0)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

window_name = 'Face it'

frame_counter = 0
face_counter = 0
t0 = time.perf_counter()
while(True):
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow(window_name, image)
    frame_counter += 1
    if len(faces) > 0:
        face_counter += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

t1 = time.perf_counter()

print(int(frame_counter/(t1-t0)),'frames per second')
print('One or more faces in', int(face_counter/frame_counter*100), '% of frames.')

# When everything done, release the capture
cap.release()
cv2.destroyWindow(window_name)


#-----------


