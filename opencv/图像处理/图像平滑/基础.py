import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# # 1 图像读取
# img = cv.imread('opencv\hello.jpg')
# # 2 均值滤波
# blur = cv.blur(img,(3,3))
# # 3 图像显示
# plt.figure(figsize=(10,8),dpi=100)
# plt.subplot(121),plt.imshow(img[:,:,::-1]),plt.title('原图')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur[:,:,::-1]),plt.title('均值滤波后结果')
# plt.xticks([]), plt.yticks([])
# plt.show()

# # 高斯滤波
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# # 1 图像读取
# img = cv.imread('opencv\hello.jpg')
# # 2 高斯滤波
# blur = cv.GaussianBlur(img,(10,10),1)
# # 3 图像显示
# plt.figure(figsize=(10,8),dpi=100)
# plt.subplot(121),plt.imshow(img[:,:,::-1]),plt.title('原图')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur[:,:,::-1]),plt.title('高斯滤波后结果')
# plt.xticks([]), plt.yticks([])
# plt.show()

# 中值滤波
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# 1 图像读取
img = cv.imread('opencv\hello.jpg')
# 2 中值滤波
blur = cv.medianBlur(img,5)
# 3 图像展示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(211),plt.imshow(img[:,:,::-1]),plt.title('origin')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(blur[:,:,::-1]),plt.title('MedianBlur')
plt.xticks([]), plt.yticks([])
plt.show()