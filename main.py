import cv2 # importa a biblioteca do cv2
import numpy as np # importa a biblioteca numpy e nome ela como np

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

while True:
    ret , frame = video_capture.read() # aqui ele recebe o frame do video 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= 1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x,y,w,h) in faces: # aqui indentifica a face onde tem 0,255,0 é cor e o 2 é largura
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

    eyes = eyeCascade.detectMultiScale(gray, 1.2, 18) # aqui indentifica os olhos 0,255,0 é cor e o 2 é largura
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    smiles = smileCascade.detectMultiScale(gray, 1.7, 20) #aqui indentifica o sorriso 0,255,0 é cor e o 2 é largura
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)    

    cv2.imshow('video', frame) # aqui ele mostra o video mostrando o frame 


    if cv2.waitKey(1) & 0xFF == ord('q'): # aqui deixa o sitema funcioando e so vai parar quando aperta
        #a tecla Q do codigo 0xFF
        break # aqui ele para por que apertamos a letra Q
video_capture.release() 
cv2.destroyAllWindows()    