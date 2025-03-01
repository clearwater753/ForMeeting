import cv2
import numpy as np

img_path = r'D:\Desktop\lee_code-hot100\opencv\hello.jpg'
img = cv2.imread(img_path)

# 获取像素值并修改
px = img[100:200, 100:200]
blue = img[100, 100, 0]
px[:] = [255, 255, 255]

# cv2.imshow('image', img)
# cv2.waitKey(0)

# 获取图像属性
print(img.shape, img.size, img.dtype)

# 图像通道的拆分与合并
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))

# 色彩空间的转换
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('image', img)
cv2.waitKey(0)

