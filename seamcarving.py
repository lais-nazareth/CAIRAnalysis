import numpy as np
import cv2

def computeEnergySobel(img):

    if len(img.shape) == 3:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Converter para GRAYSCALE")
    else:
        imgGray = img

    imgGray = imgGray.astype(float)

    sxKernel = np.array([[-1, 0 , 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    syKernel= np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    v = {}
    idx = 0

    for i in range(3):
        for j in range(3):
            v[idx] = imgGray[i : imgGray.shape[0]-2+i, j : imgGray.shape[1]-2+j]
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

def removeSeam(img, seam):
    rows, cols = img.shape[:2]
    mask = np.ones((rows, cols), dtype=bool)

    for i in range(rows):
        mask[i, seam[i]] = False
    
    #se img for colorida
    if len(img.shape) == 3:
        mask = np.stack([mask] * 3, axis=-1)
        newImg = img[mask].reshape((rows, cols - 1, 3))
    else:
        newImg = img[mask].reshape((rows, cols - 1))
    
    return newImg

def seamCarvingWidth(img, numRemove):
    output = np.copy(img)

    for i in range(numRemove):
        energyMap = computeEnergySobel(output)

        cumulativeMap = computeCumulativeEnergy(enegrgyMap=energyMap)

        seam = backtrackSeam(cumulativeEnergy=cumulativeMap)

        output = removeSeam(output, seam)
    
    return output


if __name__ == "__main__":
    img = cv2.imread("pedra.jpg")

    if img is not None:
        larguraOriginal = img.shape[1]
        larguraDesejada = 2000

        pixelsParaAlterar = larguraOriginal - larguraDesejada

        if pixelsParaAlterar > 0:
            output = seamCarvingWidth(img, numRemove=pixelsParaAlterar)
            cv2.imwrite('pedra_seam_carving.jpg', output)
            cv2.waitKey(0)

        elif pixelsParaAlterar < 0:
            print("logica para insercao ainda deve ser implementada")
        
        else:
            print("imagem ja esta do tamanho desejado")

        