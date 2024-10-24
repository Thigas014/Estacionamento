import cv2
import pickle
import cvzone
import numpy as np

cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)

largura, altura = 107, 48

def verificarVaga(imgPro):
      
    contadorEspaco = 0

    for pos in posList:
        x,y = pos
       
        imgCorte = imgPro[y:y+altura, x: x+largura]
        #cv2.imshow(str(x*y), imgCorte)  
        count = cv2.countNonZero(imgCorte)
       

        if count <900:
             color = (0,255,0)
             espessura = 5
             contadorEspaco += 1
        else:
             color = (0,0,255)
             espessura = 2
        cv2.rectangle(img, pos, (pos[0] + largura, pos[1] + altura), color, espessura)
        cvzone.putTextRect(img,str(count),(x,y+altura-3), scale=1, 
                           thickness=2 , offset=0, colorR=color)

    cvzone.putTextRect(img, f'Vagas Livres: {contadorEspaco}/{len(posList)}',(100,50), scale=3, 
                           thickness=5 , offset=20, colorR=(0,200,0))
    
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    sucesso, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur =cv2.GaussianBlur(imgGray, (3,3),1)
    imgLimite = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25,16)

    imgMedio = cv2.medianBlur(imgLimite,5)

    kernel = np.ones((3,3), np.uint8)
    imgDilatada = cv2.dilate(imgMedio, kernel, iterations=1)

    verificarVaga(imgDilatada)

        
    cv2.imshow("Image", img)
    #cv2.imshow("ImageBlur", imgBlur)
    #cv2.imshow("Imagelimite", imgLimite)
    cv2.waitKey(10)