#!/usr/bin/env python3

import cv2
import mediapipe as mp
import time

#get the Video
cap = cv2.VideoCapture(0)

b = False
b2 = False

#package for hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# package for faces
classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
classifier2 = cv2.CascadeClassifier('models/haarcascade_eye.xml')


# draw something
mpDraw = mp.solutions.drawing_utils



pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    # FACES

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = classifier.detectMultiScale(gray)

    # Loop over the faces and draw a rectangle around each face
    for (x, y, w, h) in faces:
        cropped_image = img[(x):(x+w), (y):(y+h)]
        if b == False or cv2.waitKey(1) == ord('f'):
            cv2.imwrite("screenshot/faces/Face-" + str(time.time()) + ".jpg", cropped_image)
            b = True
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # EYES

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    eyes = classifier2.detectMultiScale(gray)

    # Loop over the faces and draw a rectangle around each face
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


    # HANDS

    #convert an image from one color space to another https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #getHandPosition
    results = hands.process(imgRGB)

    #draw point on joint of hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #calcul fps
    cTime = time.time()
    #1sec divisée par le temps entre 2 frame soit fps = 1 / (t2 - t1)
    fps = 1/(cTime-pTime)
    pTime = cTime
    #write text(img, str(fps), position, font, size, color, epaisseur)
    #cv2.putText(img, str(int(fps)), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

    #open window to show 'img'
    if b2 == False or cv2.waitKey(33) == ord('c'):
        cv2.imwrite("screenshot/camera/Camera-" + str(time.time()) + ".jpg", img)
        b2 = True
    cv2.imshow("Hand Tracker", img)
    cv2.waitKey(1)

    if cv2.getWindowProperty("Hand Tracker", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()