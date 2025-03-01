# 1. 图像缩放
# cv2.resize(src,dsize,fx=0,fy=0,interpolation=cv2.INTER_LINEAR)
# src : 输入图像
# dsize: 绝对尺寸，直接指定调整后图像的大小
# fx,fy: 相对尺寸，将dsize设置为None，然后将fx和fy设置为比例因子即可
# interpolation：插值方法，
import cv2 as cv
import matplotlib.pyplot as plt

# 1. 读取图片
img1 = cv.imread("opencv\gray.jpg")
# 2.图像缩放
# 2.1 绝对尺寸
rows,cols = img1.shape[:2]
res = cv.resize(img1,(2*cols,2*rows),interpolation=cv.INTER_CUBIC)

# 2.2 相对尺寸
res1 = cv.resize(img1,None,fx=0.5,fy=0.5)

# 3 图像显示
# 3.1 使用opencv显示图像(不推荐)
# cv.imshow("orignal",img1)
# cv.imshow("enlarge",res)
# cv.imshow("shrink",res1)
# cv.waitKey(0)

# 3.2 使用matplotlib显示图像
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10,8),dpi=100)
axes[0].imshow(res[:,:,::-1])
axes[0].set_title("绝对尺度（放大）")
axes[1].imshow(img1[:,:,::-1])
axes[1].set_title("原图")
axes[2].imshow(res1[:,:,::-1])
axes[2].set_title("相对尺度（缩小）")
plt.show()