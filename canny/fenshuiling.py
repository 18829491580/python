import numpy as np
import cv2

rho = 1#rho的步长，即直线到图像原点(0,0)点的距离
theta = np.pi / 180#theta的范围
threshold = 15#累加器中的值高于它时才认为是一条直线
min_line_len = 30##线的最短长度，比这个短的都被忽略
max_line_gap = 8#两条直线之间的最大间隔，小于此值，认为是一条直线

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

def fenshuiling(img):
  #获取灰度图像
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  cv2.imshow('huidu',gray)
  #将图像转化为黑白两部分
  ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  cv2.imshow('erzhihua',thresh)
  # 消除噪声
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) # 形态学运算开操作，先膨胀后腐蚀
  cv2.imshow('open',opening)
  #进行膨胀处理
  sure_bg = cv2.dilate(opening,kernel,iterations=1)
  cv2.imshow('pengzhang',sure_bg)
  #进行距离变换
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
  cv2.imshow('jvli',sure_fg)
  # 确定未知区域
  sure_fg = np.uint8(sure_fg)

  unknown = cv2.subtract(sure_bg,sure_fg)
  # 标记标签，markers包含前景区域的所有信息,ret连通区域边缘的条数
  ret, markers = cv2.connectedComponents(sure_fg)

  # 分水岭变化
  markers = markers+1#将背景区域设为1
  markers[unknown==255] = 0#将未知区域设为0

  markers = cv2.watershed(img,markers)#获取栏栅

  img[markers == -1] = [255,0,0]#将栏栅区域设为蓝色

  cv2.imshow('un',img)
  

  lines = cv2.HoughLinesP(unknown, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)#函数输出的直接就是一组直线点的坐标位置（每条直线用两个点表示[x1,y1],[x2,y2]）
  left_lines, right_lines = [], []#用于存储左边和右边的直线
  for line in lines:#对直线进行分类
    for x1, y1, x2, y2 in line:
      k = (y2 - y1) / (x2 - x1)
      if k < 0:
        left_lines.append(line)
      else:
        right_lines.append(line)

  if (len(left_lines) <= 0 or len(right_lines) <= 0):
    return img

  clean_lines(left_lines, 0.001)#弹出左侧不满足斜率要求的直线
  clean_lines(right_lines, 0.001)#弹出右侧不满足斜率要求的直线
  left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]#提取左侧直线族中的所有的第一个点
  left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]#提取左侧直线族中的所有的第二个点
  right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]#提取右侧直线族中的所有的第一个点
  right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]#提取右侧侧直线族中的所有的第二个点
  left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])#拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标,-80,325 -182,480
  right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])#拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标255,325 423,480
  roi_vtx = np.array([[(-90, img.shape[0]), (300, 300), (640, 325),(7000,img.shape[0])]])#目标区域的四个点坐标，roi_vtx是一个三维的数组
  process_an_image(unknown, roi_vtx)

def calc_lane_vertices(point_list, ymin, ymax):
  x = [p[0] for p in point_list]#提取x
  y = [p[1] for p in point_list]#提取y
  fit = np.polyfit(y, x, 1)#用一次多项式x=a*y+b拟合这些点，fit是(a,b)
  fit_fn = np.poly1d(fit)#生成多项式对象a*y+b

  xmin = int(fit_fn(ymin))#计算这条直线在图像中最左侧的横坐标
  xmax = int(fit_fn(ymax))#计算这条直线在图像中最右侧的横坐标

  return [(xmin, ymin), (xmax, ymax)]

#将不满足斜率要求的直线弹出
def clean_lines(lines, threshold):
    slope=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            k=(y2-y1)/(x2-x1)
            slope.append(k)
    while len(lines) > 0:
        mean = np.mean(slope)#计算斜率的平均值，因为后面会将直线和斜率值弹出
        diff = [abs(s - mean) for s in slope]#计算每条直线斜率与平均值的差值
        idx = np.argmax(diff)#计算差值的最大值的下标
        if diff[idx] > threshold:#将差值大于阈值的直线弹出
          slope.pop(idx)#弹出斜率
          lines.pop(idx)#弹出线段
        else:
          break

def roi_mask(img, vertices):
  mask = np.zeros_like(img)#生成与输入图像相同大小的图像，并使用0填充,图像为黑色
  if len(img.shape) > 2:
    channel_count = img.shape[2]  
    mask_color = (255,) * channel_count
  else:
    mask_color = 255
  #使用白色填充多边形，形成蒙板
  cv2.fillPoly(mask, vertices, mask_color)
  height, width = img.shape
  for row in range(height):
      for list in range(width):
          pv = img[row, list]
          img[row, list] = 255 - pv
  cv2.imshow('img',img)
  masked_img = cv2.bitwise_and(img, mask)#img&mask，经过此操作后，兴趣区域以外的部分被蒙住了，只留下兴趣区域的图像
  return masked_img

def process_an_image(img,roi_vtx):
  roi_edges = roi_mask(img, roi_vtx)#对边缘检测的图像生成图像蒙板，去掉不感兴趣的区域，保留兴趣区
  cv2.imshow('resImg',roi_edges)


img = cv2.imread('xingchexian1.jpg')
cv2.imshow('origin',img)
fenshuiling(img)
cv2.waitKey(0)
cv2.destroyAllWindows()