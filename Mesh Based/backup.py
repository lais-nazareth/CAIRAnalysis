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

    # inicializar matriz esparsa A
    # como cada vertice tem coordenada (x, y), entao o numero de colunas é 2 * totalVertices
    # para acessar o componente x de um vertice devemos acessar a coluna (idx * 2) e para acessar Y devemos acessar (idx * 2 + 1)
    A = lil_matrix((0, 2 * totalVertices), dtype=float)

    # inicializar vetor B
    B = []

    # hiperparâmetros que podem ser ajustados
    wShape = 1.0
    wOrient = 0.5
    wScale = 0.2

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
            A[idxLinha + 1, v1 * 2 + 1] = pesoScale
            A[idxLinha + 1, v0 * 2 + 1] = -pesoScale
            B.append(alturaPatch * so * pesoScale)

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
