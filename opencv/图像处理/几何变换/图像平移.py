import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# 1. 读取图像
img1 = cv.imread("opencv\gray.jpg")

# 2. 图像平移
rows,cols = img1.shape[:2]
M = np.float32([[1,0,100],[0,1,50]])# 平移矩阵
dst = cv.warpAffine(img1,M,(cols,rows))

# 3. 图像显示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img1[:,:,::-1])
axes[0].set_title("原图")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("平移后结果")
plt.show()