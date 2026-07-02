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
from MySeamCarving import computeEnergySobel, computeCumulativeEnergy, backtrackSeam, removeSeam, redimensionarImagem, seamCarvingInsertWidth


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


#calcula a distancia entre um bloco de pixels
def calc_patch_dist(img_s, img_t, y, x_s, x_t, raio):
    dist = 0.0
    # percorre o quadrado ao redor do pixel
    for dy in range(-raio, raio + 1):
        for dx in range(-raio, raio + 1):
            for c in range(3):
                val_s = img_s[y + dy, x_s + dx, c]
                val_t = img_t[y + dy, x_t + dx, c]
                dist += abs(val_s - val_t)
    return dist


def asymmetric_dtw_patch(img_s, img_t, y, raio):
    # desconta a borda artificial para saber o tamanho real do loop
    len_s = img_s.shape[1] - (2 * raio)
    len_t = img_t.shape[1] - (2 * raio)
    
    # matriz de prog dinamica com valores infinitos
    M = np.full((len_s + 1, len_t + 1), np.inf)
    
    M[0, 0] = 0
    M[0, 1:] = 0
    
    # laço principal do A-DTW
    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            px_s = i - 1 + raio
            px_t = j - 1 + raio
            
            d = calc_patch_dist(img_s, img_t, y, px_s, px_t, raio)
            
            M[i, j] = min(
                M[i-1, j-1] + d,
                M[i, j-1],
                M[i-1, j] + d
            )
            
    return M[len_s, len_t]


#bi-directional warping do artigo (versao patches)
def calcular_bdw(img_original, img_alvo):
    if img_original.shape == img_alvo.shape:
        return 0.0
        
    altura = img_original.shape[0]
    
    # define o tamanho do patch (raio 3 = patch de 7x7 pixels)
    raio_patch = 3 
    
    # adiciona uma borda replicada para o patch nao sair da imagem
    img_orig_pad = cv2.copyMakeBorder(img_original, raio_patch, raio_patch, raio_patch, raio_patch, cv2.BORDER_REPLICATE).astype(np.float32)
    img_alvo_pad = cv2.copyMakeBorder(img_alvo, raio_patch, raio_patch, raio_patch, raio_patch, cv2.BORDER_REPLICATE).astype(np.float32)
    
    max_ST = 0.0
    max_TS = 0.0
    
    for i in range(altura):
        # a coordenada Y real na imagem com padding é i + raio_patch
        y_real = i + raio_patch
        
        # erro original -> alvo
        erro_st = asymmetric_dtw_patch(img_orig_pad, img_alvo_pad, y_real, raio_patch)
        if erro_st > max_ST:
            max_ST = erro_st
            
        # erro alvo -> original
        erro_ts = asymmetric_dtw_patch(img_alvo_pad, img_orig_pad, y_real, raio_patch)
        if erro_ts > max_TS:
            max_TS = erro_ts
            
    #custo final eh a soma dos 2 erros
    custo_final = max_ST + max_TS
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
            
            custo = calcular_bdw(img_original, img_temp)
            
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
        
        custo = calcular_bdw(img_original, img_temp)
        
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

def multi_operator(image_path, target_width, target_height):
    img_original = cv2.imread(image_path)
    
    if img_original is None:
        print(f"Erro: Não foi possível encontrar '{image_path}'")
        return
    
    # reduzir a imagem para os testes rodarem mais rápido
    img_original = redimensionarImagem(img_original, limite=500)
    largura_original, altura_original = img_original.shape[1], img_original.shape[0]
    
    print(f"Imagem carregada: {largura_original}x{altura_original}")
    
    output = multi_operator_regular_path(img_original, target_width, target_height, step_size=20)
    
    path_saida = f"Multi Operator/output/output_{target_width}x{target_height}.jpg"
    cv2.imwrite(path_saida, output)
    print(f"Salvo: {path_saida}")

if __name__ == "__main__":
    # Alterar aqui o nome do arquivo e do diretório
    nomeImg = "ellie.jpeg"
    nomePasta = "ellie"
    pasta_multiop = os.path.join(pasta_raiz, 'Multi Operator')
    imgOriginal = cv2.imread(f"Multi Operator/{nomePasta}/{nomeImg}")
    
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
        pasta_saida = os.path.join(pasta_atual, "output")
        os.makedirs(pasta_saida, exist_ok=True)

        for idx, (larguraAlvo, alturaAlvo, nomeSaida) in enumerate(experimentos, 1):
            print(f"\n======================================")
            print(f"RODANDO EXPERIMENTO {idx} (Alvo: {larguraAlvo}x{alturaAlvo})")
            print(f"======================================")
            
            #se o step size selecionado for muito pequeno em relacao ao numero de pixels pra remover/aumentar, o codigo vai demorar MUITO
            output = multi_operator_regular_path(imgOriginal, larguraAlvo, alturaAlvo, step_size=20)

            
            pathSaida = f"Multi Operator/output/{nomeSaida}"
            cv2.imwrite(pathSaida, output)
            print(f"\n[!] Salvo: {pathSaida}")
