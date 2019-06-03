import cv2

image = 'shuye.jpg'
img = cv2.imread(image)
cv2.imshow('img',img)
image_canny = cv2.Canny(img, 100, 200)
cv2.imshow('img-Canny', image_canny)

cv2.waitKey()
cv2.destroyAllWindows()
