# cv2.getRotationMatrix2D(center, angle, scale)
# 参数：
# center：旋转中心
# angle：旋转角度
# scale：缩放比例
# 返回：
# M：旋转矩阵
# 调用cv.warpAffine完成图像的旋转
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img = cv.imread("opencv\gray.jpg")

# 2 图像旋转
rows,cols = img.shape[:2]
# 2.1 生成旋转矩阵
M = cv.getRotationMatrix2D((rows/2, cols/2),45,1)

# new_center_x = cols / 2
# new_center_y = rows / 2
# # 平移量
# tx = new_center_x - (cols / 2 * M[0, 0] + rows / 2 * M[0, 1])
# ty = new_center_y - (cols / 2 * M[1, 0] + rows / 2 * M[1, 1])
# M[0, 2] += tx
# M[1, 2] += ty

# 2.2 进行旋转变换
dst = cv.warpAffine(img,M,(cols,rows))

# 3 图像展示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("原图")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("旋转后结果")
plt.show()
