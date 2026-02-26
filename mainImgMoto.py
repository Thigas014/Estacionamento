import cv2
import pickle
import cvzone
import numpy as np

# ---------------- CONFIGURAÇÃO ----------------
# RTSP da câmera (modo tempo real)
RTSP_URL = 'rtsp://admin:admin123456@10.12.15.173:8554/profile0'
# RTSP_URL = 'rtsp://admin:admin123456@10.12.12.181:8554/profile0'

# Arquivos usados para teste por imagem (fallback)
IMG_PATH = 'imgaovivo.png'
PKL_PATH = 'imgaovivo.pkl'
#IMG_PATH = 'teste-2.png'
#PKL_PATH = 'teste-2.pkl'

# Exibição
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 500

# Limiarização / classificação (ajuste se necessário)
LIMIAR_MOTO_MIN = 18    # % >= isto -> considera possível moto/ocupação parcial
LIMIAR_CARRO = 25      # % >= isto -> considera carro

# ------------------------------------------------
# Carrega posições das vagas (com resolução base salva)
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
    base_width, base_height = data["resolucao"]
    posList = data["posicoes"]


def escalar_coordenadas(pontos, base_width, base_height, frame_width, frame_height):
    """
    Escala as coordenadas dos pontos da resolução base para a resolução do frame atual.
    
    Args:
        pontos: Lista de tuplas (x, y) com coordenadas na resolução base
        base_width, base_height: Resolução da imagem onde as vagas foram marcadas
        frame_width, frame_height: Resolução do frame atual da câmera
    
    Returns:
        Lista de tuplas (x, y) escaladas para a resolução do frame
    """
    scale_x = frame_width / base_width
    scale_y = frame_height / base_height
    
    pontos_escalados = []
    for x, y in pontos:
        x_novo = int(x * scale_x)
        y_novo = int(y * scale_y)
        pontos_escalados.append((x_novo, y_novo))
    
    return pontos_escalados


def verificarVaga_frame(img_color, imgPro):
    """
    img_color: frame color (onde desenhamos)
    imgPro: frame pré-processado binário (usado para contagem)
    """
    frame_height, frame_width = img_color.shape[:2]
    
    livresNormais = ocupadasNormais = 0
    livresEspeciais = ocupadasEspeciais = 0

    for idx, vaga in enumerate(posList, start=1):
        pontos_base = vaga["pontos"]
        tipo = vaga["tipo"]
        
        pontos_escalados = escalar_coordenadas(
            pontos_base, 
            base_width, base_height,
            frame_width, frame_height
        )
        
        pts = np.array(pontos_escalados, np.int32).reshape((-1, 1, 2))

        # Máscara da vaga
        mascara = np.zeros_like(imgPro)
        cv2.fillPoly(mascara, [pts], 255)
        vagaArea = cv2.bitwise_and(imgPro, imgPro, mask=mascara)

        # Contagem de pixels "ocupados" dentro da vaga
        count = cv2.countNonZero(vagaArea)
        area_px = cv2.contourArea(pts) if cv2.contourArea(pts) > 0 else 1
        perc_ocup = (count / area_px) * 100

        # Classificação simples: livre / moto / carro
        if perc_ocup < LIMIAR_MOTO_MIN:
            livre = True
            cor = (0, 255, 0)
            status = "LIVRE"
        elif LIMIAR_MOTO_MIN <= perc_ocup < LIMIAR_CARRO:
            livre = False
            cor = (0, 165, 255)  # laranja = moto / ocupação parcial
            status = "OCUPADA (MOTO)"
        else:
            livre = False
            cor = (0, 0, 255)  # vermelho = carro
            status = "OCUPADA (CARRO)"

        # Ajuste de cor para vagas especiais (deficientes)
        if tipo != "normal":
            if livre:
                cor = (255, 180, 0)   # azul claro (vaga especial livre)
            else:
                cor = (255, 255, 0)    # azul escuro (vaga especial ocupada)

        # Contagem por tipo
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

        # Desenhos na imagem colorida
        cv2.polylines(img_color, [pts], isClosed=True, color=cor, thickness=3)
        cvzone.putTextRect(
            img_color,
            f"{status} ({perc_ocup:.1f}%)",
            (pts[0][0][0], pts[0][0][1] - 12),
            scale=0.8, thickness=2, offset=2, colorR=cor
        )

    # Painel de contagem
    total_vagas = len(posList)
    total_normais = livresNormais + ocupadasNormais
    total_especiais = livresEspeciais + ocupadasEspeciais

    cvzone.putTextRect(img_color, f'Total: {total_vagas}', (20, 30),
                       scale=1.2, thickness=2, offset=10, colorR=(243,73,243))
    cvzone.putTextRect(img_color, f'Normais Livres: {livresNormais}/{total_normais}', (20, 70),
                       scale=1.0, thickness=2, offset=10, colorR=(0,200,0))
    cvzone.putTextRect(img_color, f'Especiais Livres: {livresEspeciais}/{total_especiais}', (20, 110),
                       scale=1.0, thickness=2, offset=10, colorR=(255,255,0))


# ---------------- MODO 1: TEMPO-REAL (RTSP) ----------------
def run_rtsp():
    print("Tentando conectar à câmera RTSP...")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("⚠️  Não foi possível conectar à câmera RTSP.")
        return False  # Falha -> permite fallback automático

    ret, test_frame = cap.read()
    if ret:
        frame_height, frame_width = test_frame.shape[:2]
        print(f"✅ Conectado à câmera RTSP.")
        print(f"📐 Resolução da imagem base (marcação): {base_width}x{base_height}")
        print(f"📐 Resolução da câmera ao vivo: {frame_width}x{frame_height}")
        print(f"📊 Fator de escala: X={frame_width/base_width:.2f}, Y={frame_height/base_height:.2f}")
        print("Pressione 'q' para sair.")
    else:
        print("⚠️  Não foi possível ler frame de teste.")
        cap.release()
        return False

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️  Falha ao ler frame da câmera.")
            break

        # Pré-processamento
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        _, imgThreshold = cv2.threshold(imgBlur, 100, 255, cv2.THRESH_BINARY_INV)
        imgMedio = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilatada = cv2.dilate(imgMedio, kernel, iterations=1)

        verificarVaga_frame(frame, imgDilatada)

        img_show = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Deteccao RTSP (Tempo Real)", img_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


# ---------------- MODO 2: TESTE POR IMAGEM ----------------
def run_image_test():
    print("🖼️  Modo imagem local ativado (fallback).")
    img = cv2.imread(IMG_PATH)
    if img is None:
        print("Erro ao carregar a imagem de teste:", IMG_PATH)
        return

    frame_height, frame_width = img.shape[:2]
    print(f"📐 Resolução da imagem base (marcação): {base_width}x{base_height}")
    print(f"📐 Resolução da imagem de teste: {frame_width}x{frame_height}")
    if frame_width != base_width or frame_height != base_height:
        print(f"⚠️  ATENÇÃO: Resoluções diferentes! Aplicando escala automática.")
        print(f"📊 Fator de escala: X={frame_width/base_width:.2f}, Y={frame_height/base_height:.2f}")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    _, imgThreshold = cv2.threshold(imgBlur, 100, 255, cv2.THRESH_BINARY_INV)
    imgMedio = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilatada = cv2.dilate(imgMedio, kernel, iterations=1)

    verificarVaga_frame(img, imgDilatada)

    img_show = cv2.resize(img, (DISPLAY_WIDTH - 20, DISPLAY_HEIGHT))
    cv2.imshow("Deteccao (Imagem Local)", img_show)
    print("Pressione qualquer tecla para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Tenta rodar com a câmera RTSP
    sucesso = run_rtsp()

    # Se não conseguir conectar, ativa automaticamente o modo imagem
    if not sucesso:
        print("\n⚙️  Entrando automaticamente no modo de teste por imagem...\n")
        run_image_test()
