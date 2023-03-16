import cv2
import numpy as np
import myutils
import os
import sys


def cv_show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


src = input('输入识别的图片路径:')
img_src = sys.path[0]
img_card_bgr = cv2.imread(src)
print(img_src)
img_num_bgr = cv2.imread('/'.join([img_src, 'images/ocr_a_reference.png']))
img_num_gray = cv2.imread('/'.join([img_src, 'images/ocr_a_reference.png']), 0)
cv_show(img_num_bgr)

# 分割数字模板
numbers = {}
img_num_binary = cv2.threshold(img_num_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
# cv_show(img_num_binary)
contours_num = cv2.findContours(img_num_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
# 排序轮廓
contours_num = myutils.sort_contours(contours_num)[0]
img_num_copy = img_num_bgr.copy()
for i, co in enumerate(contours_num):
    x, y, w, h = cv2.boundingRect(co)
    # img_num_copy = cv2.rectangle(img_num_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi = img_num_binary[y:y + h, x:x + w]
    numbers[i] = roi
    # cv_show(roi)
# cv_show(img_num_copy)

# for op in numbers:
#     cv_show(numbers[op])

# 重置大小 宽统一600 高按比例
img_card_bgr = myutils.resize(img_card_bgr, 600)

img_card_gray = cv2.cvtColor(img_card_bgr, cv2.COLOR_BGR2GRAY)
card_copy = img_card_bgr.copy()

# 卷积核
rectKernel = np.ones((2, 14), np.uint8)
kernel = np.ones((3, 3), np.uint8)
small_kernel = np.ones((2, 2), np.uint8)

# sobel算子检测图像梯度
sobel_x = cv2.Sobel(img_card_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_card_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
sobel_xy = cv2.convertScaleAbs(sobel_xy)
# cv_show(sobel_xy)

# # 中值滤波
# sobel_xy = cv2.medianBlur(sobel_xy, 3)

img_card_binary = cv2.threshold(sobel_xy, 100, 255, cv2.THRESH_BINARY)[1]
# img_card_binary = cv2.morphologyEx(img_card_binary, cv2.MORPH_CLOSE, small_kernel)
# cv_show(img_card_binary)

# 闭操作
close1 = cv2.morphologyEx(sobel_xy, cv2.MORPH_CLOSE, rectKernel, iterations=2)
# cv_show(close1)

# 二值化
card_binary = cv2.threshold(close1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show(card_binary)

# 第二次闭操作 填补漏洞
close2 = cv2.morphologyEx(card_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
check1 = np.hstack((card_binary, close2))
# cv_show(close2)

# 边缘检测
contours = cv2.findContours(close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
contours = myutils.sort_contours(contours)[0]
for co in contours:
    x, y, w, h = cv2.boundingRect(co)
    ar = w / float(h)
    x -= 3
    y -= 3
    w += 5
    h += 5
    if 2.3 < ar < 3.5 and 90 < w < 120:
        # print(x,y,w,h)
        # print(ar)
        card_copy = cv2.rectangle(card_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for i in range(x, x + w - int(w / 8), int(w / 4)):
            # print(x,i)
            roi = img_card_binary[y:y+h, i:i+int(w/4)]
            print(myutils.find_number(roi, numbers), end='')
            # cv_show(roi)
        print(' ', end='')
# for co in contours:
#     card_copy = cv2.drawContours(card_copy, co, -1, (0,0,255),2)
cv_show(card_copy)
last = input('输出任意字符结束')
