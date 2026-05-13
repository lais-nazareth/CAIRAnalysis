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

    return mag


img = cv2.imread('pedra.jpg')
resultado = computeEnergySobel(img)
cv2.imwrite('pedra_borda.jpg', resultado)

cv2.imshow('Sobel', resultado)
cv2.waitKey(0)
