import cv2
import numpy as np
from skimage.measure import label

def gaussianFilter(image):
    imgFloat = image.astype(float)
    imgPadded = cv2.copyMakeBorder(imgFloat, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    gaussianKernel = 1/273*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    
    v = {}
    idx = 0

    for i in range(len(gaussianKernel)):
        for j in range(len(gaussianKernel)):
            v[idx] = imgPadded[i : imgPadded.shape[0]-4+i, j : imgPadded.shape[1]-4+j]
            idx += 1

    imgGaussian = (v[0]*gaussianKernel[0,0] + v[1]*gaussianKernel[0,1] + v[2]*gaussianKernel[0,2]+v[3]*gaussianKernel[0,3]+v[4]*gaussianKernel[0,4]+
                   v[5]*gaussianKernel[1,0] + v[6]*gaussianKernel[1,1] + v[7]*gaussianKernel[1,2]+v[8]*gaussianKernel[1,3]+v[9]*gaussianKernel[1,4]+
                   v[10]*gaussianKernel[2,0] + v[11]*gaussianKernel[2,1] + v[12]*gaussianKernel[2,2]+v[13]*gaussianKernel[2,3]+v[14]*gaussianKernel[2,4]+
                   v[15]*gaussianKernel[3,0] + v[16]*gaussianKernel[3,1] + v[17]*gaussianKernel[3,2]+v[18]*gaussianKernel[3,3]+v[19]*gaussianKernel[3,4]+
                   v[20]*gaussianKernel[4,0] + v[21]*gaussianKernel[4,1] + v[22]*gaussianKernel[4,2]+v[23]*gaussianKernel[4,3]+v[24]*gaussianKernel[4,4])
    
    # imgGaussian = np.clip(imgGaussian, 0, 255).astype(np.uint8)

    return imgGaussian

def downsampleImage(image):
    imageGaussian = gaussianFilter(image)
    imageDownsampled = cv2.resize(imageGaussian, (imageGaussian.shape[1]//2, imageGaussian.shape[0]//2), interpolation=cv2.INTER_NEAREST)
    return imageDownsampled

def calculateContrast(image):
    image = image.astype(float)
    altura, largura = image.shape[:2]

    contrastMap = np.zeros((altura, largura), dtype=float)
    centroX, centroY = largura//2.0, altura//2.0

    r_l_max = np.sqrt(centroX**2 + centroY**2)

    colunas, linhas = np.meshgrid(np.arange(largura), np.arange(altura))

    
    # distancia do pixel (i,j) até o centro (vetorizado)
    r_ijl = np.sqrt((colunas - centroX)**2 + (linhas - centroY)**2)

    # matriz de pesos (para considerar heuristica de que o centro de uma imagem costuma ter mais informacao)
    w_ijl = 1.0 - (r_ijl/r_l_max)
    

    #r_ijl = 1.0
    #w_ijl = 1.0
    # matriz para somar vizinhança
    sumDistances = np.zeros((altura, largura), dtype=float)

    deslocamentos = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    for dx, dy in deslocamentos:
        # deslocar matriz de forma circular nas direções especificadas
        vizinhos = np.roll(image, shift=(-dx, -dy), axis=(0,1))

        #d(p_ijl, pq)
        diferencaQuadrado = (image - vizinhos) ** 2

        if len(image.shape) == 3:
            somaCanais = np.sum(diferencaQuadrado, axis=-1)
            distanciaL2 = np.sqrt(somaCanais)
        else:
            distanciaL2 = np.sqrt(diferencaQuadrado)


        sumDistances += distanciaL2

    contrastMap = w_ijl * sumDistances

    return contrastMap

    



def scaleInvariantSaliency(image):

    imgLuv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    imgLuv = imgLuv.astype(float)

    altura, largura = imgLuv.shape[0], imgLuv.shape[1]
    nLevels = int(np.log2(min(altura, largura)/10))

    piramide = [imgLuv]

    for level in range(1, nLevels):
        imgReduzida = downsampleImage(piramide[-1])
        piramide.append(imgReduzida)
    
    saliencyMap = np.zeros((altura, largura), dtype=float)

    for level in range(nLevels):
        contrasteNivelK = calculateContrast(piramide[level])
        contrasteRedimensionado = cv2.resize(contrasteNivelK, (largura, altura), interpolation=cv2.INTER_LINEAR)
        saliencyMap += contrasteRedimensionado

    return saliencyMap


def segmentRegions(imageBGR, spatialRadius=120, colorRadius = 125):
    segmented = cv2.pyrMeanShiftFiltering(imageBGR, spatialRadius, colorRadius)
    altura, largura = segmented.shape[:2]
    cores, coresIdx = np.unique(segmented.reshape(-1, 3), axis=0, return_inverse=True)
    mapaCores = coresIdx.reshape(altura, largura)
    labels = label(mapaCores, background=-1, connectivity=2)

    return labels, segmented


def regionEnhancedSaliency(saliencyMap, labels):
    altura, largura = saliencyMap.shape
    numRegioes = np.max(labels) + 1

    labelsAchatados = labels.ravel()
    saliencyAchatada = saliencyMap.ravel()

    somaPorLabel = np.bincount(labelsAchatados, weights=saliencyAchatada)
    contagemPorLabel = np.bincount(labelsAchatados)
    mediaPorLabel = somaPorLabel / np.maximum(contagemPorLabel, 1)

    enhanced = mediaPorLabel[labelsAchatados].reshape(altura, largura)
    return enhanced

def adaBoost(imageBGR, saliencyMap):
    
    gray = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    enhancedMap = saliencyMap.copy()
    for (x, y, w, h) in faces:
        enhancedMap[y:y+h, x:x+w] = enhancedMap[y:y+h, x:x+w] * 2.0

    return enhancedMap
    

def saliency_pipeline(img):
    if img is None:
        print("Não foi possível encontrar o arquivo. Verifique o caminho!")
        return None
    else:
        print("Imagem carregada com sucesso. Iniciando pipeline...")
        
        mapa_saliencia_bruto = scaleInvariantSaliency(img)
        mapa_adaBoost = adaBoost(img, mapa_saliencia_bruto)
        
        mapa_salience_final = np.maximum(mapa_adaBoost, mapa_saliencia_bruto)

        # labels, segmented = segmentRegions(img)
        # mapa_salience_final = regionEnhancedSaliency(mapa_salience_final, labels)
        

        saliency_min = np.min(mapa_salience_final)
        saliency_max = np.max(mapa_salience_final)

        
        if saliency_max - saliency_min != 0:
            saliency_final = ((mapa_salience_final - saliency_min) / (saliency_max - saliency_min) * 255).astype(np.uint8)
        else:
            saliency_final = np.zeros_like(mapa_salience_final, dtype=np.uint8)

        cv2.imwrite("Mesh Based/output/saliency_result_segmented.jpeg", mapa_salience_final)
        # cv2.imwrite("Mesh Based/output/segmentation.jpeg", segmented)
        cv2.imwrite("Mesh Based/output/saliency_result_ada_boost.jpeg", saliency_final)
        return mapa_salience_final