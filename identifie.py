#!/usr/bin/env python
import cv2
import pickle
import common as c

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner/trainner.yml")

id_image=0
color_info=(255, 255, 255)
color_ko=(0, 0, 255)
color_ok=(0, 255, 0)


with open("labels.pickle", "rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k, v in og_labels.items()}
cap=cv2.VideoCapture("swinga.mp4")
#cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    #tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Détecte des objets de différentes tailles dans l'image d'entrée. 
    #Les objets détectés sont renvoyés sous forme de liste de rectangles.
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(c.min_size, c.min_size))
    for (x, y, w, h) in faces:
        roi_gray=cv2.resize(gray[y:y+h, x:x+w], (c.min_size, c.min_size))
        id_, conf=recognizer.predict(roi_gray)
        #print(conf)
        if conf<=95:
             color=color_ok
             name=labels[id_]
        else:
            color=color_ko
            name="Inconnu"
        label=name
        
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
   
    cv2.imshow('faceRec', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
               

cap.release()
cv2.destroyAllWindows()
print("Fin")