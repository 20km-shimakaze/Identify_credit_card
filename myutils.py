import cv2
import numpy as np
from functools import cmp_to_key


def cv_show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, weigh, height=None, inter=cv2.INTER_AREA):
    if height is None:
        h, w = image.shape[:2]
        # print(h, w)
        height = int(h * (weigh / w))
    return cv2.resize(image, (weigh, height), interpolation=inter)


def sort_contours_cmp_less(x, y):
    x1, y1, w1, h1 = cv2.boundingRect(x)
    x2, y2, w2, h2 = cv2.boundingRect(y)
    if h1 * w1 == h2 * w2:
        return 0
    if h1 * w1 > h2 * w2:
        return -1
    if h1 * w1 < h2 * w2:
        return 1


def sort_contours(contours, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    if method == 'greater':
        bounding_boxes = sorted(contours, key=cmp_to_key(sort_contours_cmp_less))
        return contours, bounding_boxes

    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))
    return contours, bounding_boxes


def find_number(img_num, digits):
    score = []
    # img_step1 = img_num.copy()
    small_kernel = np.ones((2, 2), np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    img_num = resize(img_num, digits[0].shape[1], digits[0].shape[0])
    img_num = cv2.morphologyEx(img_num, cv2.MORPH_OPEN, small_kernel, iterations=1)
    img_num = cv2.morphologyEx(img_num, cv2.MORPH_CLOSE, kernel, iterations=4)
    # img_step2 = img_num.copy()

    img_num = cv2.erode(img_num, small_kernel, iterations=1)
    # img_step3 = img_num.copy()
    img_num = resize(img_num, digits[0].shape[1], digits[0].shape[0])
    contours = cv2.findContours(img_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    contours = sort_contours(contours, 'greater')[1]
    x, y, w, h = cv2.boundingRect(contours[0])
    img_num = img_num[y:y + h, x:x + w]

    img_num = resize(img_num, digits[0].shape[1], digits[0].shape[0])
    # img_step1 = resize(img_step1, digits[0].shape[1], digits[0].shape[0])
    # img_step2 = resize(img_step2, digits[0].shape[1], digits[0].shape[0])
    # img_step3 = resize(img_step3, digits[0].shape[1], digits[0].shape[0])
    # cv_show(np.hstack((img_step1, img_step2, img_step3, img_num, digits[0])))
    # cv_show(img_num)

    # print(digits[0].shape, img_num.shape)
    # print(img_num.shape)
    # print(digits[0].shape)
    for i, di in enumerate(digits):
        res = cv2.matchTemplate(img_num, digits[i], cv2.TM_SQDIFF_NORMED)
        min_val = cv2.minMaxLoc(res)[0]
        # print(min_val, i)
        score.append((min_val, i))
    score = sorted(score)
    # print(score)
    # print(score)
    # img = np.hstack((digits[score[0][1]], img_num))
    # cv_show(img)
    return score[0][1]
# img_num_bgr = cv2.imread('images/ocr_a_reference.png')
# img_num_gray = cv2.imread('images/ocr_a_reference.png', 0)
#
# # 分割数字模板 255, cv2.THRESH_BINARY_INV)[1]
# # cv_show(img_num_binary)
# contours_num = cv2.findContours(img_num_binary
# # img_num_binary = cv2.threshold(img_num_gray, 127,, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
# img_num_copy = img_num_bgr.copy()
# for co in contours_num:
#     x, y, w, h = cv2.boundingRect(co)
#     img_num_copy = cv2.rectangle(img_num_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
# cv_show(img_num_copy)
