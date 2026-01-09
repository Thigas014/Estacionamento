import cv2
import pickle
import numpy as np

IMG_PATH = 'imgOcupada.jpeg'
POS_FILE = 'imgOcupada.pkl'

# Carrega as posições salvas (caso existam)
try:
    with open(POS_FILE, 'rb') as f:
        data = pickle.load(f)
        posList = data["posicoes"]
        base_res = data["resolucao"]
except:
    posList = []
    base_res = None  # ainda não definido

pontosTemp = []
modoEspecial = False  # False = normal, True = especial

# Carrega a imagem original
img_base = cv2.imread(IMG_PATH)
if img_base is None:
    print(f"Erro ao carregar imagem: {IMG_PATH}")
    exit()

# Usa a resolução real da imagem
window_height, window_width = img_base.shape[:2]


def mouseClick(event, x, y, flags, param):
    global pontosTemp, modoEspecial
    if event == cv2.EVENT_LBUTTONDOWN:
        pontosTemp.append((x, y))
        if len(pontosTemp) == 4:
            tipo = 'especial' if modoEspecial else 'normal'
            posList.append({"pontos": pontosTemp.copy(), "tipo": tipo})
            pontosTemp = []
            salvar()
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, vaga in enumerate(posList):
            pts = np.array(vaga["pontos"], dtype=np.int32)
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                posList.pop(i)
                salvar()
                break


def salvar():
    data = {
        "resolucao": (window_width, window_height),  # resolução real
        "posicoes": posList
    }
    with open(POS_FILE, 'wb') as f:
        pickle.dump(data, f)


cv2.namedWindow("Marcação de Vagas - 4 Pontos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Marcação de Vagas - 4 Pontos", window_width, window_height)

while True:
    img = img_base.copy()

    for vaga in posList:
        pts = np.array(vaga["pontos"], np.int32).reshape((-1, 1, 2))
        cor = (255, 0, 255) if vaga["tipo"] == "normal" else (255, 255, 0)
        cv2.polylines(img, [pts], isClosed=True, color=cor, thickness=2)

    for pt in pontosTemp:
        cv2.circle(img, pt, 5, (0, 255, 0), cv2.FILLED)

    cor_texto = (255, 0, 255) if not modoEspecial else (255, 255, 0)
    cv2.putText(img, f'Modo: {"Especial" if modoEspecial else "Normal"}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_texto, 2)

    cv2.imshow("Marcação de Vagas - 4 Pontos", img)
    cv2.setMouseCallback("Marcação de Vagas - 4 Pontos", mouseClick)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        modoEspecial = not modoEspecial

cv2.destroyAllWindows()
