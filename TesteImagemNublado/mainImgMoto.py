import cv2
import pickle
import cvzone
import numpy as np

# Carrega imagem original
img = cv2.imread('imgMoto.jpg')
if img is None:
    print("Erro ao carregar a imagem!")
    exit()

# Carrega posições (com resolução base salva)
with open('imgMoto.pkl', 'rb') as f:
    data = pickle.load(f)
    base_width, base_height = data["resolucao"]
    posList = data["posicoes"]

frame_height, frame_width = img.shape[:2]

# Define escala para exibição
display_scale = 0.6
disp_width = int(frame_width * display_scale)
disp_height = int(frame_height * display_scale)


def verificarVaga(imgPro):
    livresNormais = ocupadasNormais = 0
    livresEspeciais = ocupadasEspeciais = 0

    for vaga in posList:
        pontos = vaga["pontos"]
        tipo = vaga["tipo"]
        pts = np.array(pontos, np.int32).reshape((-1, 1, 2))

        # Máscara
        mascara = np.zeros_like(imgPro)
        cv2.fillPoly(mascara, [pts], 255)
        vagaArea = cv2.bitwise_and(imgPro, imgPro, mask=mascara)
        count = cv2.countNonZero(vagaArea)
        area_px = cv2.contourArea(pts)
        perc_ocup = (count / area_px) * 100

        # --- NOVA LÓGICA ---
        # Ajuste de limiares
        LIMIAR_CARRO = 25      # %ocup acima disso = carro
        LIMIAR_MOTO_MIN = 15    # %ocup entre 8 e 25 = moto
        LIMIAR_MOTO_MAX = 25

        if perc_ocup < LIMIAR_MOTO_MIN:
            livre = True
            cor = (0, 255, 0)
            status = "LIVRE"
        elif LIMIAR_MOTO_MIN <= perc_ocup <= LIMIAR_MOTO_MAX:
            livre = False
            cor = (0, 165, 255)  # Laranja = moto
            status = "OCUPADA (MOTO)"
        else:
            livre = False
            cor = (0, 0, 255)
            status = "OCUPADA (CARRO)"

        # Contagem
        if tipo == "normal":
            if livre:
                livresNormais += 1
            else:
                ocupadasNormais += 1
        else:
            if livre:
                livresEspeciais += 1
            else:
                ocupadasEspeciais += 1

        # Desenha
        cv2.polylines(img, [pts], isClosed=True, color=cor, thickness=4)
        cvzone.putTextRect(
            img,
            f"{status} ({perc_ocup:.1f}%)",
            (pts[0][0][0], pts[0][0][1] - 10),
            scale=0.9,
            thickness=2,
            offset=2,
            colorR=cor
        )

    # Totais
    total_vagas = len(posList)
    total_normais = livresNormais + ocupadasNormais
    total_especiais = livresEspeciais + ocupadasEspeciais

    cvzone.putTextRect(img, f'Total Vagas: {total_vagas}',
                       (50, 80), scale=2, thickness=3, offset=10, colorR=(243, 73, 243))
    cvzone.putTextRect(img, f'Normais Livres: {livresNormais}/{total_normais}',
                       (50, 120), scale=2, thickness=3, offset=10, colorR=(0, 200, 0))
    cvzone.putTextRect(img, f'Especiais Livres: {livresEspeciais}/{total_especiais}',
                       (50, 160), scale=2, thickness=3, offset=10, colorR=(255, 255, 0))


# --- PRÉ-PROCESSAMENTO ---
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
_, imgThreshold = cv2.threshold(imgBlur, 100, 255, cv2.THRESH_BINARY_INV)
imgMedio = cv2.medianBlur(imgThreshold, 5)
kernel = np.ones((3, 3), np.uint8)
imgDilatada = cv2.dilate(imgMedio, kernel, iterations=1)

verificarVaga(imgDilatada)

# Exibir
img_show = cv2.resize(img, (disp_width, disp_height))
cv2.namedWindow("Detecção de Vagas", cv2.WINDOW_NORMAL)
cv2.imshow("Detecção de Vagas", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
