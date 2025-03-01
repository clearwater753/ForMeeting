import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\Desktop\lee_code-hot100\opencv\hello.jpg', 0)
cv2.imshow('image', img)
cv2.waitKey(0)

path = './opencv/gray.jpg'
cv2.imwrite(path, img)

# # 与matplotlib的显示方式不同，opencv的显示方式是BGR，而matplotlib的显示方式是RGB
# plt.imshow(img)
# # plt.imshow(img[:, :, ::-1]) # bgr -> rgb
# plt.title('image'), plt.xticks([]), plt.yticks([]) # 隐藏坐标轴,加上标题
# plt.show()