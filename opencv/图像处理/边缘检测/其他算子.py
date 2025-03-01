import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# 1 读取图像
img = cv.imread('opencv\hello.jpg')
# 2 laplacian转换
result = cv.Laplacian(img,cv.CV_16S)
Scale_abs = cv.convertScaleAbs(result)


# 2 Canny边缘检测
lowThreshold = 120
max_lowThreshold = 150
canny = cv.Canny(img, lowThreshold, max_lowThreshold) 

# 3 图像展示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(131),plt.imshow(img[:,:,::-1],cmap=plt.cm.gray),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(Scale_abs[:,:,::-1],cmap = plt.cm.gray),plt.title('Laplacian检测后结果')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(canny,cmap = plt.cm.gray),plt.title('Canny检测后结果')
plt.xticks([]), plt.yticks([])
plt.show()