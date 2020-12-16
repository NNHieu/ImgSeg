#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt
#%%
#load image
PATH = './mtest.jpg'
img = cv2.imread(PATH)

#applying structured edge detection
edgedetector = cv2.ximgproc.createStructuredEdgeDetection('./model.yml')
src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
edges = edgedetector.detectEdges(np.float32(src) / 255.0)

#normalizing to 0 - 255
data = edges * 255
normImg = data.astype(np.uint8)

#global thresholding
ret, glob_th = cv2.threshold(normImg, 127, 255, cv2.THRESH_BINARY)

#adaptive thresholding
adapt_th1 = cv2.adaptiveThreshold(normImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
adapt_th2 = cv2.adaptiveThreshold(normImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 0)

#Otsu's thresholding
ret1, otsu_th1 = cv2.threshold(normImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(normImg, (5,5), 0)
ret2, otsu_th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #otsu's thresholding after gaussian filtering


#%%
#finding contour
def findContour(th):
    _, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourImg = np.zeros((img.shape[0], img.shape[1]))
    contourImg.fill(255)
    contourImg = cv2.drawContours(contourImg, contours, -1, (0,0,0), 1)
    return contourImg
def plotc(src, edges, th):
    plt.subplot(221),plt.imshow(src)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(edges, cmap = 'gray')
    plt.title('Structured Edges Detection'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(th, cmap = 'gray')
    plt.title('Binarization'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(findContour(th), cmap ='gray')
    plt.title('Contours Map'), plt.xticks([]), plt.yticks([])
    plt.show()

# %%
