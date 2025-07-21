import cv2
import pickle
import numpy as np

IMG_PATH = 'estacionamento.jpeg'
POS_FILE = 'estacionamentopos.pkl'

# Carrega as posições salvas
try:
    with open(POS_FILE, 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# Lista temporária de pontos sendo marcados
pontosTemp = []

def mouseClick(event, x, y, flags, param):
    global pontosTemp

    # Clique esquerdo: adicionar ponto
    if event == cv2.EVENT_LBUTTONDOWN:
        pontosTemp.append((x, y))
        if len(pontosTemp) == 4:
            posList.append(pontosTemp.copy())
            pontosTemp = []
            salvar()

    # Clique direito: remover vaga
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, vaga in enumerate(posList):
            pts = np.array(vaga, dtype=np.int32)
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                posList.pop(i)
                salvar()
                break

def salvar():
    with open(POS_FILE, 'wb') as f:
        pickle.dump(posList, f)

while True:
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Erro ao carregar imagem: {IMG_PATH}")
        break

    img = cv2.resize(img, (1200, 720))

    # Desenhar todas as vagas salvas
    for vaga in posList:
        pts = np.array(vaga, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

    # Desenhar pontos temporários
    for pt in pontosTemp:
        cv2.circle(img, pt, 5, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Marcação de Vagas - 4 Pontos", img)
    cv2.setMouseCallback("Marcação de Vagas - 4 Pontos", mouseClick)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
