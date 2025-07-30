import cv2
import pickle
import numpy as np

IMG_PATH = 'IMGVideoEstacionandoNublado.jpeg'
POS_FILE = 'IMGVideoEstacionandoNublado.pkl'

# Carrega as posições salvas
try:
    with open(POS_FILE, 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

pontosTemp = []

def mouseClick(event, x, y, flags, param):
    global pontosTemp
    if event == cv2.EVENT_LBUTTONDOWN:
        pontosTemp.append((x, y))
        if len(pontosTemp) == 4:
            posList.append(pontosTemp.copy())
            pontosTemp = []
            salvar()
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

#tela ocupando todo o espaço
screen_width, screen_height = 1920, 1080
taskbar_height = 40

window_width = screen_width
window_height = screen_height - taskbar_height

cv2.namedWindow("Marcação de Vagas - 4 Pontos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Marcação de Vagas - 4 Pontos", window_width, window_height)

while True:
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Erro ao carregar imagem: {IMG_PATH}")
        break

    img = cv2.resize(img, (window_width, window_height))

    for vaga in posList:
        pts = np.array(vaga, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

    for pt in pontosTemp:
        cv2.circle(img, pt, 5, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Marcação de Vagas - 4 Pontos", img)
    cv2.setMouseCallback("Marcação de Vagas - 4 Pontos", mouseClick)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
