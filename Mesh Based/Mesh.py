import cv2
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as splinalg

def createMeshGrid(alturaO, larguraO, patch_size=32):
    xs = np.arange(0, larguraO + 1, patch_size)
    ys = np.arange(0, alturaO + 1, patch_size)

    # garante que ultimo patch termina na larguraO/alturaO
    if xs[-1] != larguraO: xs[-1] = larguraO
    if ys[-1] != alturaO: ys[-1] = alturaO

    return xs, ys

def calculatePatchSaliency(saliencyMap, xs, ys):
    numPatchesY = len(ys) - 1
    numPatchesX = len(xs) - 1

    patchEnergies = np.zeros((numPatchesY, numPatchesX), dtype=float)

    for i in range(numPatchesY):
        for j in range(numPatchesX):
            xStart, xEnd = xs[j], xs[j+1]
            yStart, yEnd = ys[i], ys[i+1]
            
            patchRegiao = saliencyMap[yStart:yEnd, xStart:xEnd]

            # o paper faz isso
            patchRegiao = patchRegiao ** 2
            patchEnergies[i, j] = np.mean(patchRegiao)

    return patchEnergies

def optimizeMesh(larguraO, alturaO, larguraT, alturaT, patchEnergies, patchSize):
    '''
        Monta e resolve sistema linear Ax = B
        Utiliza restrições do artigo:
            1. Shape distortion (patch deve continuar quadrado)
            2. Orientation distortion (arestas não devem inclinar)
            3. Scale orientation (fator de escala global)
        Utiliza método dos mínimos quadrados (MMQ) para medir erro
    '''

    xs = np.arange(0, larguraO + 1, patchSize)
    ys = np.arange(0, alturaO + 1, patchSize)
    if xs[-1] != larguraO: xs[-1] = larguraO
    if ys[-1] != alturaO: ys[-1] = alturaO

    numVX = len(xs)
    numVY = len(ys)
    totalVertices = numVX * numVY

    # calcular indice global unico
    def getIndiceGlobal(linha, coluna):
        return linha*numVX + coluna
    
    # calcular fator de escala
    if larguraT > larguraO or alturaT > alturaO:
        so = min(larguraT/larguraO, alturaT/alturaO)
    else:
        so = max(larguraT/larguraO, alturaT/alturaO)

    if patchEnergies.max() > 0:
        patchEnergies = patchEnergies / patchEnergies.max()

    # inicializar matriz esparsa A
    # como cada vertice tem coordenada (x, y), entao o numero de colunas é 2 * totalVertices
    # para acessar o componente x de um vertice devemos acessar a coluna (idx * 2) e para acessar Y devemos acessar (idx * 2 + 1)
    A = lil_matrix((0, 2 * totalVertices), dtype=float)

    # inicializar vetor B
    B = []

    # hiperparâmetros que podem ser ajustados
    wShape = 1.0
    wOrient = 1.0
    wScale = 1.0

    numPatchesX = numVX - 1
    numPatchesY = numVY - 1

    for i in range(numPatchesY):
        for j in range(numPatchesX):
            salienciaQuadrada = patchEnergies[i, j] + 1.0
            

            larguraPatch = xs[j+1] - xs[j]
            alturaPatch = ys[i+1] - ys[i]
            ratio = alturaPatch/larguraPatch

            # pegar indice global das quatro quinas do patch

            v0 = getIndiceGlobal(i, j)
            v1 = getIndiceGlobal(i+1, j)
            v2 = getIndiceGlobal(i+1, j+1)
            v3 = getIndiceGlobal(i, j+1)

            # SHAPE DISTORTION
            # adiciona espaço para 2 linhas (uma eq para X e outra para Y)
            idxLinha = A.shape[0]
            A.resize((idxLinha + 2, totalVertices * 2))

            pesoShape = wShape * salienciaQuadrada

            # eq para componente X
            A[idxLinha, v0 * 2] = 1 * pesoShape
            A[idxLinha, v1 * 2] = -1 * pesoShape
            A[idxLinha, v1 * 2 + 1] = ratio * pesoShape
            A[idxLinha, v2 * 2 + 1] = -ratio * pesoShape
            B.append(0)

            # eq para componente Y
            # eq para componente X
            A[idxLinha + 1, v0 * 2 + 1] = 1 * pesoShape
            A[idxLinha + 1, v1 * 2 + 1] = -1 * pesoShape
            A[idxLinha + 1, v1 * 2] = -ratio * pesoShape
            A[idxLinha + 1, v2 * 2] = ratio * pesoShape
            B.append(0)


            # ORIENTATION DISTORTION
            idxLinha = A.shape[0]
            A.resize((idxLinha + 4, 2 * totalVertices))
            pesoOrient = wOrient * salienciaQuadrada

            # y0' - y3' = 0
            A[idxLinha, v0 * 2 + 1] = pesoOrient
            A[idxLinha, v3 * 2 + 1] = -pesoOrient
            B.append(0)

            # y1' - y2' = 0
            A[idxLinha + 1, v1 * 2 + 1] = pesoOrient
            A[idxLinha + 1, v2 * 2 + 1] = -pesoOrient
            B.append(0)

            # x0' - x1' = 0
            A[idxLinha + 2, v0 * 2] = pesoOrient
            A[idxLinha + 2, v1 * 2] = -pesoOrient
            B.append(0)

            # x3' - x2' = 0
            A[idxLinha + 3, v3 * 2] = pesoOrient
            A[idxLinha + 3, v2 * 2] = -pesoOrient
            B.append(0)

            # SCALE DISTORTION
            idxLinha = A.shape[0]
            A.resize((idxLinha + 2, 2 * totalVertices))
            pesoScale = wScale * salienciaQuadrada

            # x3' - x0' = larguraPatch * so
            A[idxLinha, v3 * 2] = pesoScale
            A[idxLinha, v0 * 2] = -pesoScale
            B.append(larguraPatch * so * pesoScale)

            # y1' - y0' = alturaPatch * so
            pesoScaleY = 10.0  
            A[idxLinha + 1, v1 * 2 + 1] = pesoScaleY
            A[idxLinha + 1, v0 * 2 + 1] = -pesoScaleY
            B.append(alturaPatch * 1.0 * pesoScaleY)

    # lidar com bordas da imagem
    pesoBorda = 1000.0  
    for i in range(numVY):
        for j in range(numVX):
            idxVertice = getIndiceGlobal(i, j)
            
            # borda esquerda deve ser 0
            if j == 0:
                idxLinha = A.shape[0]
                A.resize((idxLinha + 1, totalVertices * 2))
                A[idxLinha, idxVertice * 2] = pesoBorda
                B.append(0)
            
            if j == numVX - 1:
                idxLinha = A.shape[0]
                A.resize((idxLinha + 1, totalVertices * 2))
                A[idxLinha, idxVertice * 2] = pesoBorda
                B.append(larguraT * pesoBorda)
            
            if i == 0:
                idxLinha = A.shape[0]
                A.resize((idxLinha + 1, totalVertices * 2))
                A[idxLinha, idxVertice * 2 + 1] = pesoBorda
                B.append(0)

            if i == numVY - 1:
                idxLinha = A.shape[0]
                A.resize((idxLinha + 1, totalVertices * 2))
                A[idxLinha, idxVertice * 2 + 1] = pesoBorda
                B.append(alturaT * pesoBorda)

    B = np.array(B, dtype=float)
    A_csr = A.tocsr()

    solucaoLinear = splinalg.lsqr(A_csr, B)[0]

    novosVertices = solucaoLinear.reshape((totalVertices, 2))

    return novosVertices

    
def warpImage(imgOriginal, larguraT, alturaT, patchSize, novosVertices):
    alturaO, larguraO = imgOriginal.shape[:2]
    xs, ys = createMeshGrid(alturaO, larguraO, patchSize)
    numVX = len(xs)

    imgResultado = np.zeros((alturaT, larguraT, 3), dtype=np.uint8)

    numPatchesX = len(xs) - 1
    numPatchesY = len(ys) - 1

    def getIndiceGlobal(linha, coluna):
        return linha * numVX + coluna
    
    for i in range(numPatchesY):
        for j in range(numPatchesX):
            v0 = getIndiceGlobal(i, j)
            v1 = getIndiceGlobal(i+1, j)
            v2 = getIndiceGlobal(i+1, j+1)
            v3 = getIndiceGlobal(i, j+1)

            pt0_0 = np.array([xs[j], ys[i]], dtype=np.float32)
            pt1_0 = np.array([xs[j], ys[i+1]], dtype=np.float32)
            pt2_0 = np.array([xs[j+1], ys[i+1]], dtype=np.float32)
            pt3_0 = np.array([xs[j+1], ys[i]], dtype=np.float32)

            pt0_T = novosVertices[v0].astype(np.float32)
            pt1_T = novosVertices[v1].astype(np.float32)
            pt2_T = novosVertices[v2].astype(np.float32)
            pt3_T = novosVertices[v3].astype(np.float32)

            srcTri1 = np.array([pt0_0, pt1_0, pt2_0], dtype=np.float32)
            dstTri1 = np.array([pt0_T, pt1_T, pt2_T], dtype=np.float32)
            M1 = cv2.getAffineTransform(srcTri1, dstTri1)

            warp1 = cv2.warpAffine(imgOriginal, M1, (larguraT, alturaT), flags=cv2.INTER_LINEAR)
            mask1 = np.zeros((alturaT, larguraT), dtype=np.uint8)
            cv2.fillConvexPoly(mask1, dstTri1.astype(np.int32), 255)
            imgResultado[mask1 == 255] = warp1[mask1 == 255]

            srcTri2 = np.array([pt0_0, pt2_0, pt3_0], dtype=np.float32)
            dstTri2 = np.array([pt0_T, pt2_T, pt3_T], dtype=np.float32)
            M2 = cv2.getAffineTransform(srcTri2, dstTri2)
            warp2 = cv2.warpAffine(imgOriginal, M2, (larguraT, alturaT), flags=cv2.INTER_LINEAR)
            mask2 = np.zeros((alturaT, larguraT), dtype=np.uint8)
            
            cv2.fillConvexPoly(mask2, dstTri2.astype(np.int32), 255)
            
            imgResultado[mask2 == 255] = warp2[mask2 == 255]
        
    return imgResultado
    
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Saliency import scaleInvariantSaliency

    # Alterar aqui o nome do arquivo e do diretório
    nomeImg = "braga.jpeg"
    nomePasta = "braga"
    img = cv2.imread(f"Seam Carving/{nomePasta}/{nomeImg}")
    if img is None:
        print("[ERRO] Não foi possível carregar a imagem. Verifique o caminho.")
    else:
        h_orig, w_orig = img.shape[:2]
        print(f"=== PASSO 1: IMAGEM CARREGADA ===")
        print(f"Dimensões Originais: {w_orig}x{h_orig}")
        
        # Resoluções para o teste de debug
        W_ALVO = 1400
        H_ALVO = h_orig
        TAM_PATCH = 10
        print(f"Dimensões Alvo: {W_ALVO}x{H_ALVO} | Tamanho do Patch: {TAM_PATCH}")
        
        mapa_saliencia = scaleInvariantSaliency(img)
        
        xs, ys = createMeshGrid(h_orig, w_orig, TAM_PATCH)
        
        patch_energies = calculatePatchSaliency(mapa_saliencia, xs, ys)
        novos_vertices = optimizeMesh(w_orig, h_orig, W_ALVO, H_ALVO, patch_energies, TAM_PATCH)
        
        resultado_retarget = warpImage(img, W_ALVO, H_ALVO, TAM_PATCH, novos_vertices)
        
        pixels_pretos = np.sum(resultado_retarget == 0)
        total_pixels = resultado_retarget.size
        
        cv2.imwrite("mesh_result.jpeg", resultado_retarget)
