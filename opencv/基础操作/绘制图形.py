import cv2
import numpy as np

# 创建空白图像
img = np.zeros((512, 512, 3), np.uint8)
# 绘制图形
# cv2.line(img, pt1, pt2, color, thickness=1)
# cv2.circle(img, center, radius, color, thickness=1)
# cv2.rectangle(img, pt1, pt2, color, thickness=1)
# cv2.putText(img, text, pt, fontFace, fontScale, color, thickness=1)
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
cv2.putText(img, 'OpenCV', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# 显示图像
cv2.imshow('image', img)
cv2.waitKey(0)
