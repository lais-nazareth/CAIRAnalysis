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

#TODO implementar a do artigo
#reduzir por cropping
def reduzir_crop(image, num_pixels):
    if num_pixels == 0:
        return image
    
    # V1 simples: Cortando tudo do lado direito
    # nova_imagem = image[:, :-num_pixels]
    
    # V2 mais equilibrada: Cortando um pouco de cada lado
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

# ==========================================

#TODO funcao dummy, trocar pelo BDW
def calcular_custo_teste(img_original, img_alvo):
    """
    Função DUMMY. 
    Retorna um custo falso (diferença de cor média) para o loop poder rodar.
    Depois trocaremos pelo Asymmetric-DTW.
    """
    if img_original.shape == img_alvo.shape:
        return 0
    # Custo falso baseado na média de cor da imagem
    return abs(np.mean(img_original) - np.mean(img_alvo))


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
            
            custo = calcular_custo_teste(img_original, img_temp)
            
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
        
        custo = calcular_custo_teste(img_original, img_temp)
        
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
    # Alterar aqui o nome do arquivo e do diretório
    nomeImg = "ellie.jpeg"
    nomePasta = "ellie"
    pasta_multiop = os.path.join(pasta_raiz, 'Multi Operator')
    imgOriginal = cv2.imread(f"Multi Operator/{nomePasta}/{nomeImg}")
    
    if imgOriginal is None:
        print(f"Erro: Não foi possível encontrar '{nomeImg}'")
    else:
        # Reduzir a imagem para os testes rodarem mais rápido
        imgOriginal = redimensionarImagem(imgOriginal, limite=500)
        larguraOriginal, alturaOriginal = imgOriginal.shape[1], imgOriginal.shape[0]
        
        print(f"Imagem carregada: {larguraOriginal}x{alturaOriginal}")

        experimentos = [
            (larguraOriginal, 300, f"{nomeImg}_panoramica_altura_reduzida.jpg"),
            (200, alturaOriginal, f"{nomeImg}squish_largura_reduzida.jpg"),
            (400, alturaOriginal, f"{nomeImg}_expansao_largura_inserida.jpg"),
            (larguraOriginal ,700, f"{nomeImg}_expansao_altura_inserida.jpg"),
        ]

        # Criar a pasta de saída do MultiOp se não existir
        pasta_saida = os.path.join(pasta_atual, "output")
        os.makedirs(pasta_saida, exist_ok=True)

        for idx, (larguraAlvo, alturaAlvo, nomeSaida) in enumerate(experimentos, 1):
            print(f"\n======================================")
            print(f"RODANDO EXPERIMENTO {idx} (Alvo: {larguraAlvo}x{alturaAlvo})")
            print(f"======================================")
            
            #se o step size selecionado for muito pequeno em relacao ao numero de pixels pra remover/aumentar, o codigo vai demorar MUITO
            output = multi_operator_regular_path(imgOriginal, larguraAlvo, alturaAlvo, step_size=50)

            
            pathSaida = f"Multi Operator/output/{nomeSaida}"
            cv2.imwrite(pathSaida, output)
            print(f"\n[!] Salvo: {pathSaida}")
