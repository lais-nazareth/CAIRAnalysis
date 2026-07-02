import sys
import os
import cv2
import numpy as np

# import seam carving
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_raiz = os.path.dirname(pasta_atual)
pasta_seam_carving = os.path.join(pasta_raiz, 'Seam Carving')
sys.path.append(pasta_seam_carving)
#reaproveitar as funcoes do seamcarving.py
from seamcarving import computeEnergySobel, computeCumulativeEnergy, backtrackSeam, removeSeam, redimensionarImagem, seamCarvingInsertWidth


#reduzir por escala
def reduzir_escala(image,num_pixels):
    if num_pixels == 0:
        return image
        
    altura, largura = image.shape[:2]
    nova_largura = largura - num_pixels
    nova_imagem = cv2.resize(image, (nova_largura, altura), interpolation=cv2.INTER_CUBIC)
    
    return nova_imagem


#reduzir por cropping
def reduzir_crop(image, num_pixels):
    if num_pixels == 0:
        return image
    
    corte_esq = num_pixels // 2
    corte_dir = num_pixels - corte_esq
    
    largura = image.shape[1]
    nova_imagem = image[:, corte_esq : largura - corte_dir]
    
    return nova_imagem

# funcao para reduzir pixel por seam carving
def reduzir_seamcarving(image, num_pixels):
    if num_pixels == 0:
        return image
    
    nova_imagem = image
    
    for _ in range(num_pixels):
        energyMap = computeEnergySobel(nova_imagem)
        cumulativeMap = computeCumulativeEnergy(energyMap)
        seam = backtrackSeam(cumulativeMap)
        nova_imagem = removeSeam(nova_imagem, seam)
        
    return nova_imagem


#aumentar por escala
def aumentar_escala(image, num_pixels):
    if num_pixels == 0:
        return image
        
    altura, largura = image.shape[:2]
    nova_largura = largura + num_pixels
    nova_imagem = cv2.resize(image, (nova_largura, altura), interpolation=cv2.INTER_CUBIC)
    
    return nova_imagem

#aumentar por seam carving
def aumentar_seamcarving(image, num_pixels):
    if num_pixels == 0:
        return image
    
    # aproveitamos a funcao que ja existe no seamcarving.py
    nova_imagem = seamCarvingInsertWidth(image, num_pixels)
    return nova_imagem



# METODOLOGIA DO ARTIGO (DONG ET AL. 2012)
# Substitui o (MUITO) LENTO A-DTW de Rubinstein por Energia + Cores Dominantes (DCD)

# extrai o dominant color descriptor usando clustering k-means
def extrair_dcd(imagem, k=8):
    # reduz a imagem drasticamente para rodar o k-means muito rapido
    img_reduzida = cv2.resize(imagem, (100, 100))
    pixels = np.float32(img_reduzida.reshape(-1, 3))
    
    # executa o k-means do OpenCV para achar as K cores principais
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # calcula a porcentagem (peso) de cada cor na imagem
    counts = np.bincount(labels.flatten())
    porcentagens = counts / float(counts.sum())
    
    return centers, porcentagens


# calcula a perda de informacao visual baseada na cor e seu peso
def calcular_diferenca_dcd(dcd1, dcd2):
    centers1, perc1 = dcd1
    centers2, perc2 = dcd2
    
    # ordena os clusters do maior pro menor pra comparar de forma justa
    idx1 = np.argsort(perc1)[::-1]
    idx2 = np.argsort(perc2)[::-1]
    
    diferenca = 0.0
    for i, j in zip(idx1, idx2):
        # penaliza se a cor mudou ou se a quantidade daquela cor na foto mudou
        dist_cor = np.linalg.norm(centers1[i] - centers2[j])
        dist_perc = abs(perc1[i] - perc2[j])
        diferenca += (dist_cor + (dist_perc * 255.0))
        
    return diferenca


# avaliacao unificada do artigo: Energia + DCD
def calcular_custo(img_original, img_alvo):
    if img_original.shape == img_alvo.shape:
        return 0.0
        
    # Energia de Sobel (local)
    # se o seam carving deformar demais, cria picos de energia
    # se o scale borrar demais, derruba a media da energia
    energia_orig = computeEnergySobel(img_original)
    energia_alvo = computeEnergySobel(img_alvo)
    
    diff_energia_media = abs(np.mean(energia_orig) - np.mean(energia_alvo))
    diff_energia_max = abs(np.max(energia_orig) - np.max(energia_alvo))
    custo_energia = diff_energia_media + diff_energia_max
    
    # DCD (global)
    # ex: ajuda o algoritmo a nao comer todo o azul do ceu (preservar a composicao)
    dcd_orig = extrair_dcd(img_original)
    dcd_alvo = extrair_dcd(img_alvo)
    custo_dcd = calcular_diferenca_dcd(dcd_orig, dcd_alvo)
    
    # o artigo aplica o peso omega_dcd (0.2 a 0.5) para misturar as metricas
    custo_final = custo_energia + (custo_dcd * 0.2)
    
    return custo_final


#calcula os melhores caminhos pra achar a imagem com melhor custo
def otimizar_dimensao(img_original, pixels_remover, step_size=10):
    #arredonda pra multiplo do step size
    pixels_remover = (pixels_remover // step_size) * step_size
    
    #inicializa
    melhor_custo = float('inf')
    melhor_imagem = None
    melhor_caminho = (0, 0, 0)

    print(f"Remoção de {pixels_remover}px (Step: {step_size}px)")

    #utilizando uma ordem fixa de seam, crop e scale, sao percorridas as combinacoes
    for seam_px in range(0, pixels_remover + 1, step_size):
        sobra_apos_seam = pixels_remover - seam_px
        for crop_px in range(0, sobra_apos_seam + 1, step_size):
            scale_px = pixels_remover - seam_px - crop_px
            
            img_temp = reduzir_seamcarving(img_original, seam_px)
            img_temp = reduzir_crop(img_temp, crop_px)
            img_temp = reduzir_escala(img_temp, scale_px)
            
            custo = calcular_custo(img_original, img_temp)
            
            if custo < melhor_custo:
                melhor_custo = custo
                melhor_imagem = img_temp
                melhor_caminho = (seam_px, crop_px, scale_px)

    print(f"Vencedor: Seam={melhor_caminho[0]}px, Crop={melhor_caminho[1]}px, Scale={melhor_caminho[2]}px | Custo: {melhor_custo:.2f}")
    return melhor_imagem


#calcula os melhores caminhos para aumentar a imagem (sem cropping)
def otimizar_aumento_dimensao(img_original, pixels_adicionar, step_size=10):
    pixels_adicionar = (pixels_adicionar // step_size) * step_size
    
    melhor_custo = float('inf')
    melhor_imagem = None
    melhor_caminho = (0, 0) #apenas seam e scale

    print(f"Adição de {pixels_adicionar}px (Step: {step_size}px)")

    # combinacoes apenas entre seam isertion e scale
    for seam_px in range(0, pixels_adicionar + 1, step_size):
        scale_px = pixels_adicionar - seam_px
        
        img_temp = aumentar_seamcarving(img_original, seam_px)
        img_temp = aumentar_escala(img_temp, scale_px)
        
        custo = calcular_custo(img_original, img_temp)
        
        if custo < melhor_custo:
            melhor_custo = custo
            melhor_imagem = img_temp
            melhor_caminho = (seam_px, scale_px)

    print(f"Vencedor Aumento: Seam={melhor_caminho[0]}px, Scale={melhor_caminho[1]}px | Custo: {melhor_custo:.2f}")
    return melhor_imagem


#implementacao do metodo regular path, em que a ordem das operacoes eh fixada
def multi_operator_regular_path(img_original, largura_alvo, altura_alvo, step_size=10):
    altura_atual, largura_atual = img_original.shape[:2]
    
    #LARGURA
    diff_largura = largura_alvo - largura_atual
    
    if diff_largura < 0:
        print("Otimizando Reducao de Largura")
        img_temp = otimizar_dimensao(img_original, abs(diff_largura), step_size)
    elif diff_largura > 0:
        print("Otimizando Aumento de Largura")
        img_temp = otimizar_aumento_dimensao(img_original, diff_largura, step_size)
    else:
        img_temp = img_original

    # ALTURA
    altura_atual = img_temp.shape[0]
    diff_altura = altura_alvo - altura_atual

    if diff_altura != 0:
        print(f"Otimizando {'Aumento' if diff_altura > 0 else 'Reducao'} de Altura")
        # transpoe pra fazer altura
        img_transposta = np.transpose(img_temp, (1, 0, 2)) if len(img_temp.shape) == 3 else np.transpose(img_temp, (1, 0))
        
        if diff_altura < 0:
            img_reduzida = otimizar_dimensao(img_transposta, abs(diff_altura), step_size)
        else:
            img_reduzida = otimizar_aumento_dimensao(img_transposta, diff_altura, step_size)
            
        # destranspoe
        img_final = np.transpose(img_reduzida, (1, 0, 2)) if len(img_temp.shape) == 3 else np.transpose(img_reduzida, (1, 0))
    else:
        img_final = img_temp

    return img_final


if __name__ == "__main__":
    # coloque o nome do arquivo da imagem e a pasta onde ela está
    nomeImg = "pedra.jpg"
    pastaImg = "pedra"
    
    imgOriginal = cv2.imread(f"Multi Operator/{pastaImg}/{nomeImg}")
    
    if imgOriginal is None:
        print(f"Erro: Não foi possível encontrar {nomeImg}")
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

        for idx, (larguraAlvo, alturaAlvo, nomeSaida) in enumerate(experimentos, 1):
            print(f"\n======================================")
            print(f"RODANDO EXPERIMENTO {idx} (Alvo: {larguraAlvo}x{alturaAlvo})")
            print(f"======================================")
            
            
            output = multi_operator_regular_path(imgOriginal, larguraAlvo, alturaAlvo, step_size=20)

            # salva na mesma pasta da imagem original
            pathSaida = f"Multi Operator/{pastaImg}/{nomeSaida}"
            cv2.imwrite(pathSaida, output)
            print(f"\n[!] Salvo: {pathSaida}")