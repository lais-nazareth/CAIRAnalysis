import sys
import os

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(BASE_DIR, "Seam Carving"))
import seam_carving
sys.path.pop(0)

sys.path.insert(0, os.path.join(BASE_DIR, "Mesh Based"))
import Mesh
sys.path.pop(0)

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

if __name__ == "__main__":
    # Alterar aqui o nome do arquivo e do diretório
    nomeImg = "pedra.jpg"
    nomePasta = "pedra"
    imgOriginal = cv2.imread(f"Mesh Based/{nomePasta}/{nomeImg}")
    
    if imgOriginal is None:
        print(f"Erro: Não foi possível encontrar '{nomeImg}'")
    else:
        # reduzir a imagem para os testes rodarem mais rápido
        imgOriginal = redimensionarImagem(imgOriginal, limite=500)
        larguraOriginal, alturaOriginal = imgOriginal.shape[1], imgOriginal.shape[0]
        
        print(f"Imagem carregada: {larguraOriginal}x{alturaOriginal}")

        experimentos = [
            # reduzir 100 pixels de largura
            (larguraOriginal - 100, alturaOriginal, f"{nomeImg}_reduz_largura.jpg"),

            # reduzir 100 pixels de altura
            (larguraOriginal, alturaOriginal - 100, f"{nomeImg}_reduz_altura.jpg"),

            # expandir 50% da largura
            (int(larguraOriginal * 1.5), alturaOriginal, f"{nomeImg}_expande_largura.jpg"),

            # expandir 50% da altura
            (larguraOriginal, int(alturaOriginal * 1.5), f"{nomeImg}_expande_altura.jpg"),
            
            # reduz a largura por 50px e aumenta a altura por 30%
            (larguraOriginal - 50, int(alturaOriginal * 1.3), f"{nomeImg}_misto.jpg")
        ]

        # criar a pasta de saída do MultiOp se não existir
        pasta_saida = os.path.join("Mesh Based", "output")
        os.makedirs(pasta_saida, exist_ok=True)

        for idx, (larguraAlvo, alturaAlvo, nomeSaida) in enumerate(experimentos, 1):
            print(f"\n======================================")
            print(f"RODANDO EXPERIMENTO {idx} (Alvo: {larguraAlvo}x{alturaAlvo})")
            print(f"======================================")
            
            #se o step size selecionado for muito pequeno em relacao ao numero de pixels pra remover/aumentar, o codigo vai demorar MUITO
            output = Mesh.mesh_based(imgOriginal, larguraAlvo, alturaAlvo)

            
            pathSaida = f"Mesh Based/output/{nomeSaida}"
            cv2.imwrite(pathSaida, output)
            print(f"\n[!] Salvo: {pathSaida}")
