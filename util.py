import cv2

def redimensionarImagem(img, limite=1000):
    altura, largura = img.shape[:2]

    if largura < limite and altura < limite:
        return img.copy()
    
    if largura > altura:
        fatorEscala = limite/largura
        novaLargura = limite
        novaAltura = int(altura * fatorEscala)

    if altura > largura:
        fatorEscala = limite/altura
        novaAltura = limite
        novaLargura = int(largura * fatorEscala)

    imgRedimensionada = cv2.resize(img, (novaLargura, novaAltura), interpolation=cv2.INTER_AREA)
    return imgRedimensionada
    