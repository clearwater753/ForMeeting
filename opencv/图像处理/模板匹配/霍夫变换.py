# import numpy as np
# import random
# import cv2 as cv
# import matplotlib.pyplot as plt
# # 1.加载图片，转为二值图
# img = cv.imread('opencv\hello.jpg')

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 50, 150)

# # 2.霍夫直线变换
# lines = cv.HoughLines(edges, 1, np.pi / 45, 150)
# # 3.将检测的线绘制在图像上（注意是极坐标噢）
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#     cv.line(img, (x1, y1), (x2, y2), (0, 255, 0))
# # 4. 图像显示
# plt.figure(figsize=(10,8),dpi=100)
# plt.imshow(img[:,:,::-1]),plt.title('霍夫变换线检测')
# plt.xticks([]), plt.yticks([])
# plt.show()

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# 1 读取图像，并转换为灰度图
planets = cv.imread('opencv\hello.jpg')
gay_img = cv.cvtColor(planets, cv.COLOR_BGRA2GRAY)
# 2 进行中值模糊，去噪点
img = cv.medianBlur(gay_img, 7)  
# 3 霍夫圆检测
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 200, param1=100, param2=30, minRadius=0, maxRadius=100)
circles = np.uint16(np.around(circles))  # 将浮点数转换为整数
# 4 将检测结果绘制在图像上
for i in circles[0, :]:  # 遍历矩阵每一行的数据

    cv.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # 绘制圆心
    cv.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)
# 5 图像显示
plt.figure(figsize=(10,8),dpi=100)
plt.imshow(planets[:,:,::-1]),plt.title('霍夫变换圆检测')
plt.xticks([]), plt.yticks([])
plt.show()