import cv2
import pickle
import cvzone
import numpy as np

# Carrega o vídeo
cap = cv2.VideoCapture('carPark.mp4')

# Carrega a lista de posições de vagas de estacionamento
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# Define a largura e altura das áreas das vagas
largura, altura = 107, 48

# Função para verificar a ocupação das vagas
def verificarVaga(imgPro):
    contadorEspaco = 0

    for pos in posList:
        x, y = pos
        imgCorte = imgPro[y:y+altura, x:x+largura]
        count = cv2.countNonZero(imgCorte)

        if count < 900:
            color = (0, 255, 0)
            espessura = 5
            contadorEspaco += 1
        else:
            color = (0, 0, 255)
            espessura = 2

        cv2.rectangle(img, pos, (pos[0] + largura, pos[1] + altura), color, espessura)
        cvzone.putTextRect(img, str(count), (x, y + altura - 3), scale=1, thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Vagas Livres: {contadorEspaco}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))

# Loop principal
while True:
    # Reinicia o vídeo quando atinge o último quadro
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    sucesso, img = cap.read()
    if not sucesso:
        break  # Encerra o loop se a leitura do vídeo falhar

    # Processamento da imagem
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgLimite = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedio = cv2.medianBlur(imgLimite, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilatada = cv2.dilate(imgMedio, kernel, iterations=1)

    # Verifica as vagas de estacionamento
    verificarVaga(imgDilatada)

    # Exibe a imagem com as vagas marcadas
    cv2.imshow("Image", img)

    # Adiciona a verificação para a tecla 'q' para sair
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Libera o vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
