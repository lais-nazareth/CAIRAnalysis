import numpy as np
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin")
import cv2


def computeEnergySobel(output):

    if len(output.shape) == 3:
        outputGray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        print("Converter para GRAYSCALE")
    else:
        outputGray = output

    outputGray = outputGray.astype(float)

    sxKernel = np.array([[-1, 0 , 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    syKernel= np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    v = {}
    idx = 0

    #slicing para obter os vizinhos de cada pixel, ignorando as bordas
    for i in range(3):
        for j in range(3):
            v[idx] = outputGray[i : outputGray.shape[0]-2+i, j : outputGray.shape[1]-2+j]
            idx += 1

    sx = (v[0]*sxKernel[0,0] + v[1]*sxKernel[0,1] + v[2]*sxKernel[0,2] +
          v[3]*sxKernel[1,0] + v[4]*sxKernel[1,1] + v[5]*sxKernel[1,2] +
          v[6]*sxKernel[2,0] + v[7]*sxKernel[2,1] + v[8]*sxKernel[2,2])
    
    sy = (v[0]*syKernel[0,0] + v[1]*syKernel[0,1] + v[2]*syKernel[0,2] +
          v[3]*syKernel[1,0] + v[4]*syKernel[1,1] + v[5]*syKernel[1,2] +
          v[6]*syKernel[2,0] + v[7]*syKernel[2,1] + v[8]*syKernel[2,2])
    
    mag = np.sqrt(sx**2 + sy**2)

    max_val = np.max(mag)
    if max_val > 0:
        mag = (mag / max_val) * 255
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    mag = cv2.copyMakeBorder(mag, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    return mag


def computeCumulativeEnergy(enegrgyMap):
    rows, cols = enegrgyMap.shape
    m = np.copy(enegrgyMap).astype(float)

    for i in range(1, rows):
        prevRow = m[i-1]
        left = np.insert(prevRow[:-1], 0, np.inf)
        center = prevRow
        right = np.insert(prevRow[1:], len(prevRow)-1, np.inf)

        m[i] += np.minimum(left, np.minimum(center, right))

    return m

def backtrackSeam(cumulativeEnergy):
    rows, cols = cumulativeEnergy.shape
    seam = np.zeros(rows, dtype=int)

    currentCol = np.argmin(cumulativeEnergy[-1])
    seam[rows-1] = currentCol #ultima linha

    #comecamos da penultima linha e vamos ate a primeira de modo decrescente
    for i in range(rows-2, -1, -1):
        start = max(0, currentCol - 1)
        end = min(cols, currentCol + 2)

        neighborIdx = np.argmin(cumulativeEnergy[i, start:end])

        currentCol = start + neighborIdx
        seam[i] = currentCol

    return seam

def removeSeam(output, seam):
    rows, cols = output.shape[:2]
    mask = np.ones((rows, cols), dtype=bool)

    for i in range(rows):
        mask[i, seam[i]] = False
    
    #se output for colorida
    if len(output.shape) == 3:
        mask = np.stack([mask] * 3, axis=-1)
        newOutput = output[mask].reshape((rows, cols - 1, 3))
    else:
        newOutput = output[mask].reshape((rows, cols - 1))
    
    return newOutput

def insertSeam(output, seams):
    rows, cols = output.shape[:2]
    mask = np.zeros((rows, cols), dtype=bool)

    if len(output.shape) == 3:
        newOutput = np.zeros((rows, cols + 1, 3), dtype=output.dtype)
    else:
        newOutput = np.zeros((rows, cols + 1), dtype=output.dtype)

    for i in range(rows):
        idx = seams[i]
        if idx == 0:
            neighborIdx = idx + 1
        else:
            neighborIdx = idx - 1

        interpolatedPixel = (output[i, idx].astype(float) + output[i, neighborIdx].astype(float)) // 2

        newOutput[i, :idx] = output[i, :idx]

        # adiciona o pixel interpolado na posicao idx+1
        newOutput[i, idx] = output[i, idx]
        newOutput[i, idx + 1] = interpolatedPixel

        # copia o restante da linha e desloca tudo para a direita
        newOutput[i, idx + 2:] = output[i, idx + 1:]
    
    return newOutput

def seamCarvingRemovalWidth(output, numRemove):
    output = np.copy(output)

    for i in range(numRemove):
        energyMap = computeEnergySobel(output)

        cumulativeMap = computeCumulativeEnergy(enegrgyMap=energyMap)

        seam = backtrackSeam(cumulativeEnergy=cumulativeMap)

        output = removeSeam(output, seam)
    
    return output

def seamCarvingRemovalHeight(output, numRemove):
    # transpor a imagem para manter logica
    # se for colorida, transpor mantendo as 3 camadas de cor
    if len(output.shape) == 3:
        outputTransposta = np.transpose(output, (1, 0, 2))
    else:
        outputTransposta = np.transpose(output, (1, 0))

    outputReduzido = seamCarvingRemovalWidth(outputTransposta, numRemove=numRemove)

    #transpor de volta
    if len(output.shape) == 3:
        outputFinal = np.transpose(outputReduzido, (1, 0, 2))
    else:
        outputFinal = np.transpose(outputReduzido, (1, 0))

    return outputFinal

def seamCarvingInsertWidth(output, numInsert):
    tempOutput = np.copy(output)
    seams = []

    #mapeia regioes de menor energia e remove seams temporariamente, armazenando eles s na lista seams para insercao depois
    for i in range(numInsert):
        energyMap = computeEnergySobel(tempOutput)
        cumulativeMap = computeCumulativeEnergy(enegrgyMap=energyMap)
        seam = backtrackSeam(cumulativeEnergy=cumulativeMap)
        seams.append(seam)
        tempOutput = removeSeam(tempOutput, seam)
    
    outputFinal = np.copy(output)
    while len(seams) > 0:
        currentSeam = seams.pop(0)
        outputFinal = insertSeam(outputFinal, currentSeam)
        seams = update_seams(seams, currentSeam)

    return outputFinal

def seamCarvingInsertHeight(output, numInsert):
    if len(output.shape) == 3:
        outputTransposta = np.transpose(output, (1, 0, 2))
    else:
        outputTransposta = np.transpose(output, (1, 0))

    outputReduzido = seamCarvingInsertWidth(outputTransposta, numInsert)

    if len(output.shape) == 3:
        outputFinal = np.transpose(outputReduzido, (1, 0, 2))
    else:
        outputFinal = np.transpose(outputReduzido, (1, 0))

    return outputFinal

def update_seams(remaining_seams, current_seam):
        output = []
        for seam in remaining_seams:
            updatedSeam = np.copy(seam)
            updatedSeam[np.where(updatedSeam >= current_seam)] += 1
            output.append(updatedSeam)
        return output

if __name__ == "__main__":
    imgOriginal = cv2.imread("pedra.jpg")
    
    if imgOriginal is None:
        print("Erro: Não foi possível encontrar 'pedra.jpg'")
    else:
        # reduzir imagem para teste rapido (comentar para usar a imagem original)
        # imgOriginal = cv2.resize(imgOriginal, (800, 600))
        
        larguraOriginal, alturaOriginal = imgOriginal.shape[1], imgOriginal.shape[0]
    
        experimentos = [
            (4000, 2250, "pedra_A_panoramica_altura_reduzida.jpg"),
            (2500, 3000, "pedra_B_squish_largura_reduzida.jpg"),
            (4500, 3000, "pedra_C_expansao_largura_inserida.jpg")
        ]

        for idx, (larguraAlvo, alturaAlvo, nome_saida) in enumerate(experimentos, 1):
            print(f"\nRODANDO EXPERIMENTO {idx}...")
            output = np.copy(imgOriginal)

            pixelsParaAlterarLargura = larguraOriginal - larguraAlvo
            pixelsParaAlterarAltura = alturaOriginal - alturaAlvo

            if pixelsParaAlterarLargura != 0:
                if pixelsParaAlterarLargura > 0:
                    output = seamCarvingRemovalWidth(output, numRemove=pixelsParaAlterarLargura)
                else:
                    output = seamCarvingInsertWidth(output, numInsert=-pixelsParaAlterarLargura)
            
            if pixelsParaAlterarAltura != 0:
                if pixelsParaAlterarAltura > 0:
                    output = seamCarvingRemovalHeight(output, numRemove=pixelsParaAlterarAltura)
                else:
                    output = seamCarvingInsertHeight(output, numInsert=-pixelsParaAlterarAltura)

            cv2.imwrite(nome_saida, output)
            print(f"Salvo: {nome_saida}")
