import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 1 读取图像
img = cv.imread('opencv\hello.jpg')
# 2 计算Sobel卷积结果
x = cv.Sobel(img, cv.CV_16S, 1, 0)
y = cv.Sobel(img, cv.CV_16S, 0, 1)
# 3 将数据进行转换
Scale_absX = cv.convertScaleAbs(x)  # convert 转换  scale 缩放
Scale_absY = cv.convertScaleAbs(y)
# 4 结果合成
result1 = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)

# 2. 计算 Scharr 卷积结果
x = cv.Scharr(img, cv.CV_16S, 1, 0)  # 计算 x 方向的梯度
y = cv.Scharr(img, cv.CV_16S, 0, 1)  # 计算 y 方向的梯度

# 3. 将数据进行转换
Scale_absX = cv.convertScaleAbs(x)  # 转换为绝对值并缩放
Scale_absY = cv.convertScaleAbs(y)

# 4. 结果合成
result2 = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)  # 将 x 和 y 方向的梯度合成

# 5 图像显示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(131),plt.imshow(img[:,:,::-1]),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(result1[:,:,::-1]),plt.title('Sobel')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(result2,cmap = plt.cm.gray),plt.title('Scharr')
plt.xticks([]), plt.yticks([])
plt.show()