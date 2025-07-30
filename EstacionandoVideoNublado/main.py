import cv2
import pickle
import cvzone
import numpy as np

# Carrega vídeo da câmera RTSP
 #cap = cv2.VideoCapture('rtsp://admin:admin123456@10.12.12.181:8554/profile0')

cap = cv2.VideoCapture('VideoEstacionandoNublado.mp4')  # caminho do video

# Carrega as posições das vagas (cada vaga tem 4 pontos)
with open('IMGVideoEstacionandoNublado.pkl', 'rb') as f:
    posList = pickle.load(f)

# Dimensão base da imagem onde marcou as vagas
IMG_WIDTH, IMG_HEIGHT = 1920, 1080

def verificarVaga(imgPro, frame_width, frame_height):
    contadorEspaco = 0

    for pontos in posList:
        # Escala os pontos para o tamanho do frame atual
        scaled_points = []
        for pt in pontos:
            x_scaled = int(pt[0] * frame_width / IMG_WIDTH)
            y_scaled = int(pt[1] * frame_height / IMG_HEIGHT)
            scaled_points.append((x_scaled, y_scaled))

        pts = np.array(scaled_points, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Cria uma máscara da vaga
        mascara = np.zeros_like(imgPro)
        cv2.fillPoly(mascara, [pts], 255)

        # Aplica a máscara na imagem processada
        vagaArea = cv2.bitwise_and(imgPro, imgPro, mask=mascara)

        # Conta pixels brancos (ocupação)
        count = cv2.countNonZero(vagaArea)

        # Define estado da vaga
        if count < 17000:
            color = (0, 255, 0)  # Livre
            espessura = 5
            contadorEspaco += 1
        else:
            color = (0, 0, 255)  # Ocupada
            espessura = 2

        # Desenha o polígono da vaga
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=espessura)
        cvzone.putTextRect(img, str(count), (pts[0][0][0], pts[0][0][1] - 5),
                           scale=1, thickness=2, offset=0, colorR=color)

    # Exibe total de vagas livres
    cvzone.putTextRect(img, f'Vagas Livres: {contadorEspaco}/{len(posList)}',
                       (50, 100), scale=2, thickness=3, offset=10, colorR=(0, 200, 0))

# Janela redimensionável
cv2.namedWindow("Detecção de Vagas", cv2.WINDOW_NORMAL)

# Loop principal
while True:
    sucesso, img = cap.read()
    if not sucesso:
        print("Falha ao capturar vídeo.")
        break

    frame_height, frame_width = img.shape[:2]

    # Pré-processamento da imagem
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    _, imgThreshold = cv2.threshold(imgBlur, 100, 255, cv2.THRESH_BINARY_INV)
    imgMedio = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilatada = cv2.dilate(imgMedio, kernel, iterations=1)

    # Verifica as vagas com escalonamento
    verificarVaga(imgDilatada, frame_width, frame_height)

    # Mostra o resultado
    cv2.imshow("Detecção de Vagas", img)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
