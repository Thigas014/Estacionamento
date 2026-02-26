import cv2
import pickle
import numpy as np

IMG_PATH = 'imgaovivo.png'
POS_FILE = 'imgaovivo.pkl'

# =========================
# Carrega posições salvas
# =========================
try:
    with open(POS_FILE, 'rb') as f:
        data = pickle.load(f)
        posList = data["posicoes"]
        base_res = data["resolucao"]
except:
    posList = []
    base_res = None

pontosTemp = []
modoEspecial = False

# =========================
# Carrega imagem original
# =========================
img_original = cv2.imread(IMG_PATH)
if img_original is None:
    print(f"Erro ao carregar imagem: {IMG_PATH}")
    exit()

h, w = img_original.shape[:2]

# =========================
# Redimensionamento proporcional
# =========================
target_w = 1920
target_h = 1080

scale = min(target_w / w, target_h / h)

new_w = int(w * scale)
new_h = int(h * scale)

img_resized = cv2.resize(img_original, (new_w, new_h))

# Cria fundo preto 1920x1080
img_base = np.zeros((target_h, target_w, 3), dtype=np.uint8)

# Centraliza imagem
x_offset = (target_w - new_w) // 2
y_offset = (target_h - new_h) // 2

img_base[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

window_width = target_w
window_height = target_h

# =========================
# Função Mouse
# =========================
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

# =========================
# Salvar posições
# =========================
def salvar():
    data = {
        "resolucao": (window_width, window_height),
        "posicoes": posList
    }
    with open(POS_FILE, 'wb') as f:
        pickle.dump(data, f)

# =========================
# Janela
# =========================
cv2.namedWindow("Marcação de Vagas - 4 Pontos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Marcação de Vagas - 4 Pontos", window_width, window_height)
cv2.setMouseCallback("Marcação de Vagas - 4 Pontos", mouseClick)

# =========================
# Loop principal
# =========================
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

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        modoEspecial = not modoEspecial

cv2.destroyAllWindows()