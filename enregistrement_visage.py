import cv2
import os
import common as c

cap=cv2.VideoCapture("Mustapha swinga New.mp4")
#cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

path= "dataset"
os.chdir(path)
name=input("entrer votre nom :")
if not os.path.isdir(name):
    os.mkdir(name)

sampleNum=0
id=0
while True:
    ret, frame=cap.read()
    if ret is False:
        break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Détecte des visages de différentes tailles dans l'image d'entrée. 
    #Les visages détectés sont renvoyés sous forme de liste de rectangles.
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(c.min_size, c.min_size))
    for x, y, w, h in face:
         #incrémentation du numéro d'échantillon 
        sampleNum=sampleNum+1
        #enregistrement du visage capturé dans le dossier dataset\name
        cv2.imwrite("{}/p-{:d}.png".format(name, id), frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        id+=1
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    elif sampleNum>1000:
        break
    
    cv2.imshow('enregistreent des visages', frame)
    
    for cpt in range(4):
        ret, frame=cap.read()


cap.release()
cv2.destroyAllWindows()