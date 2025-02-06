import cv2
import pickle
import cvzone
import numpy as np

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Define largura da webcam
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Define altura da webcam

# Carrega posições das vagas
with open('mdf2pos', 'rb') as f:
    posList = pickle.load(f)

# Define o tamanho das vagas
largura, altura = 200,100

# Função para verificar a ocupação das vagas


def verificarVaga(imgPro):
    contadorEspaco = 0

    for pos in posList:
        x, y = pos
        imgCorte = imgPro[y:y+altura, x:x+largura]
        # Conta pixels brancos (ocupação da vaga)
        count = cv2.countNonZero(imgCorte)

        # **MODO DEBUG:** Exibir valores para calibração
       # print(f'Vaga em ({x}, {y}) - Pixels brancos: {count}')

        # Ajuste do limiar para definir se está ocupado ou não
        if count < 5000:  # Aumentado para melhor detecção
            color = (0, 255, 0)  # Verde = vaga livre
            espessura = 5
            contadorEspaco += 1
        else:
            color = (0, 0, 255)  # Vermelho = vaga ocupada
            espessura = 2

        # Desenha o retângulo na imagem
        cv2.rectangle(img, (x, y), (x + largura, y + altura), color, espessura)
        cvzone.putTextRect(img, str(count), (x, y + altura - 3),
                           scale=1, thickness=2, offset=0, colorR=color)

    # Exibe a contagem de vagas livres
    cvzone.putTextRect(img, f'Vagas Livres: {contadorEspaco}/{len(posList)}', (50, 50), scale=2,
                       thickness=3, offset=10, colorR=(0, 200, 0))


# Loop principal
while True:
    sucesso, img = cap.read()
    if not sucesso:
        break  # Sai se a captura falhar

    # **Novo Pré-processamento**
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    # Substituindo o `adaptiveThreshold` por um limiar fixo
    _, imgThreshold = cv2.threshold(imgBlur, 100, 255, cv2.THRESH_BINARY_INV)

    imgMedio = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilatada = cv2.dilate(imgMedio, kernel, iterations=1)

    # Verifica ocupação das vagas
    verificarVaga(imgDilatada)

    # Exibe o resultado
    cv2.imshow("Detecção de Vagas", img)

    # Pressione 'q' para sair
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
