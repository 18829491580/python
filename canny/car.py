import numpy as np
import cv2

def grabCut(img):

    # 创建一个和加载图像一样形状的 填充为0的掩膜
    ph1 = np.zeros(img.shape[:2], np.uint8)
    # 创建以0填充的前景和背景模型
    background = np.zeros((1, 65), np.float64)
    frontground = np.zeros((1, 65), np.float64)
    # 定义一个矩形
    juxing = (100, 110, 400, 350)
    # 使用grabCut算法,共计算5次
    cv2.grabCut(img, ph1, juxing, background, frontground, 5, cv2.GC_INIT_WITH_RECT)

    ph2 = np.where((ph1 == 2) | (ph1 == 0), 0, 1).astype("uint8") #ph1=2或ph1=0,ph2=0,否则,ph2=1

    img = img*ph2[:, :, np.newaxis]

    cv2.imshow('car',img)

# 读入图片
image = 'car1.jpg'
img = cv2.imread(image)
cv2.imshow('origin',img)

grabCut(img)

cv2.waitKey()
cv2.destroyAllWindows()