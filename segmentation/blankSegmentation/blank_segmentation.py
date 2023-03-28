import functools
import cv2
import os
import numpy as np
import math


class Model:
    def __init__(self, debug=False):
        self.rects = []  # 记录一张图可以标记分割的矩阵 格式为[x, y, w, h]
        self.crop_img = [] # 保存分割的图片
        self.img = None  # 图片
        self.debug = debug  # debug模式
        self.name = ''  # 图片名

    def process(self, img_path, name):  # 运行过程
        binary = self.__preProcessing(img_path, name)
        horizon = self.__detectLines(binary)
        self.__contourExtraction(horizon)
        result = self.__segmentation()
        self.rects.clear()
        self.img = None
        self.name = ''
        return result

    def __preProcessing(self, img_path, name):  # 图片预处理，输出二值图
        img = cv2.imread(img_path)
        self.img = img
        self.name = name
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh, binary = cv2.threshold(blur, int(_ * 0.95), 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def __detectLines(img):  # 检测水平线
        horizon_k = int(math.sqrt(img.shape[1]) * 1.2)  # w
        # hors_k = int(binary.shape[1]/ 16)  # w
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizon_k, 1))  # 设置内核形状
        horizon = ~cv2.dilate(img, kernel, iterations=1)  # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizon_k / 0.9), 1))
        horizon = cv2.dilate(horizon, kernel, iterations=1)
        return horizon

    def __contourExtraction(self, img, debug=False):  # 轮廓检测
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_y, border_x = img.shape
        # 去除邻近上边界和下边界检测的轮廓
        for cnt in cnts[0]:
            x, y, w, h = cv2.boundingRect(cnt)
            if y < 4 or y > border_y - 8:
                continue
            if self.debug and debug:
                cv2.rectangle(self.img, (x, y), (w + x, h + y), (0, 0, 255), 2)
            self.rects.append([x, y, w, h])
        # 排序
        self.rects = sorted(self.rects, key=functools.cmp_to_key(self.__cmp_rect_r))

        pre = None
        idx_lst = []
        # 标记不相关的轮廓
        for idx, cnt in enumerate(self.rects):
            x, y, w, h = cnt
            if w < 150:
                continue
            if pre is None:
                pre = [x, y, w]
            elif 6 < abs(pre[1] - (y + h / 2)) < 70:  # and 10 < abs(pre[0] - x) < pre[2]
                continue
            pre[1] = y + h / 2
            pre[0] = x
            pre[2] = w
            idx_lst.append(idx)

        # 再次筛选
        self.rects = [self.rects[x] for x in idx_lst]
        self.rects = sorted(self.rects, key=functools.cmp_to_key(self.__cmp_rect))
        # 将检测的水平线扩充成矩形框
        pre_y, pre_h = -1, -1
        for idx, cnt in enumerate(self.rects):
            x, y, w, h = cnt
            if pre_h == -1:
                pre_y = y
                h = y - 5
                y = 5
                pre_h = h
            else:
                if abs(pre_y - y) < 10:
                    h = pre_h
                    y = max(y - h, 0)
                else:
                    pre_h = abs(y - pre_y) - 10
                    pre_y = y
                    h = pre_h
                    y = pre_y - h
            self.rects[idx] = [x, y, w, h + 15]

    def __segmentation(self):  # 分割
        if self.debug:  # debug模式只标记不分割
            if not os.path.exists('debug'):
                os.mkdir('debug')
            for idx, rect in enumerate(self.rects):
                x, y, w, h = rect
                cv2.rectangle(self.img, (x, y), (w + x, h + y), (255, 0, 255), 2)
                cv2.putText(self.img, str(idx + 1), (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imwrite('./debug/{}.png'.format(self.name), self.img)
        else:
            if not os.path.exists('res'):
                os.mkdir('res')
            for idx, rect in enumerate(self.rects):
                x, y, w, h = rect
                crop_img = self.img[y:y + h, x:x + w]
                crop_img = crop_img.copy()
                self.crop_img.append(crop_img)
                # cv2.imwrite('./res/{}-{}.png'.format(self.name, idx + 1), crop_img)
                return self.crop_img

    @staticmethod
    def __cmp_rect(a, b):
        if (abs(a[1] - b[1]) < 10 and a[0] > b[0]) or a[1] > b[1]:
            return 1
        elif abs(a[1] - b[1]) < 10 and abs(a[0] - b[0]) < 20:
            return 0
        else:
            return -1

    @staticmethod
    def __cmp_rect_r(a, b):
        if (abs(a[1] - b[1]) < 5 and a[0] < b[0]) or a[1] > b[1]:
            return -1
        elif abs(a[1] - b[1]) < 5 and abs(a[0] - b[0]) < 5:
            return 0
        else:
            return 1


if __name__ == '__main__':
    path = 'img'  # 文件夹名
    folder = os.listdir(path)
    count = 0
    model = Model(debug=True)
    # model = Model()
    for i in folder:
        pic_path = os.path.join(path, i)
        name_ = i[:-4]
        res = model.process(pic_path, name_)  # res存储分割的图片
