import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# # 1 直接以灰度图的方式读入
# img = cv.imread('opencv\hello.jpg',0)
# # 2 统计灰度图
# histr = cv.calcHist([img],[0],None,[128],[0,256])
# # 3 绘制灰度图
# plt.figure(figsize=(10,6),dpi=100)
# plt.plot(histr)
# plt.grid()
# plt.show()

# 创建掩码

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 1. 直接以灰度图的方式读入
img = cv.imread('opencv\hello.jpg',0)
# 2. 创建蒙版
mask = np.zeros(img.shape[:2], np.uint8)
mask[0:100, 0:100] = 255
# 3.掩模
masked_img = cv.bitwise_and(img,img,mask = mask)
# 4. 统计掩膜后图像的灰度图
mask_histr = cv.calcHist([img],[0],mask,[256],[1,256])
# 5. 图像展示
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img,cmap=plt.cm.gray)
axes[0,0].set_title("原图")
axes[0,1].imshow(mask,cmap=plt.cm.gray)
axes[0,1].set_title("蒙版数据")
axes[1,0].imshow(masked_img,cmap=plt.cm.gray)
axes[1,0].set_title("掩膜后数据")
axes[1,1].plot(mask_histr)
axes[1,1].grid()
axes[1,1].set_title("灰度直方图")
plt.show()