import cv2
import numpy as np

def rotate_image(image_path, angle):
    # 读取图像
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    
    # 计算旋转矩阵
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的图像尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 执行实际的旋转和缩放
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))
    
    return rotated

# 示例使用
rotated_image = rotate_image("opencv\gray.jpg", 45)
cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()