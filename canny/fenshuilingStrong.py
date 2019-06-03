import numpy as np
import cv2

img = cv2.imread('fenshuiling.jpg')
#获取灰度图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#将图像转化为黑白两部分
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('img1',thresh)


# 消除噪声
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) # 形态学运算
cv2.imshow('opening',opening)

#进行膨胀处理
sure_bg = cv2.dilate(opening,kernel,iterations=1)
cv2.imshow('sure_bg',sure_bg)

#进行距离变换
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv2.imshow('sure_fg',sure_fg)


# 确定未知区域
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('unknown',unknown)


# 标记标签，markers包含前景区域的所有信息,ret连通区域边缘的条数
ret, markers = cv2.connectedComponents(sure_fg)

# 分水岭变化
markers = markers+1#将背景区域设为1
markers[unknown==255] = 0#将未知区域设为0

markers = cv2.watershed(img,markers)#获取栏栅
img[markers == -1] = [255,0,0]#将栏栅区域设为蓝色

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()