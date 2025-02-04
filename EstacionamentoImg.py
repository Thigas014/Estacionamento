import cv2
import pickle

largura, altura = 150,200


try:
    with open('teste1pos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []


def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + largura and y1 < y < y1 + altura:
                posList.pop(i)

    with open('teste1pos', 'wb') as f:
        pickle.dump(posList, f)


while True:
    img = cv2.imread('teste1.jpeg')
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + largura, pos[1] + altura), (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()