import cv2
import math
import numpy as np
from main.dataprocessing.image import *


class Img:
    def __init__(self, image, rows, cols, center=[0, 0]):
        self.src = image  # 原始图像
        self.rows = rows  # 原始图像的行
        self.cols = cols  # 原始图像的列
        self.center = center  # 旋转中心，默认是[0,0]

    def Move(self, image, delta_x, delta_y):  # 平移
        left = 0
        right = 0
        upon = 0
        low = 0

        if delta_x < 0:
            left = -delta_x
        else:
            right = delta_x
        if delta_y < 0:
            upon = -delta_y
        else:
            low = delta_y

        changeImge = np.pad(image, ((upon, low), (left, right)), mode='constant')[upon:low, left:right]
        return changeImge

    def Zoom(self, factor):  # 缩放
        # factor>1表示缩小；factor<1表示放大
        self.transform = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])

    def Horizontal(self):  # 水平镜像
        self.transform = np.array([[1, 0, 0], [0, -1, self.cols - 1], [0, 0, 1]])

    def Vertically(self):  # 垂直镜像
        self.transform = np.array([[-1, 0, self.rows - 1], [0, 1, 0], [0, 0, 1]])

    def Rotate(self, beta):  # 旋转
        # beta>0表示逆时针旋转；beta<0表示顺时针旋转
        self.transform = np.array([[math.cos(beta), -math.sin(beta), 0],
                                   [math.sin(beta), math.cos(beta), 0],
                                   [0, 0, 1]])

    def Process(self):
        self.dst = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([i - self.center[0], j - self.center[1], 1])
                [x, y, z] = np.dot(self.transform, src_pos)
                x = int(x) + self.center[0]
                y = int(y) + self.center[1]

                if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                    self.dst[i][j] = 255
                else:
                    self.dst[i][j] = self.src[x][y]


if __name__ == '__main__':
    # 0 去掉返回三通道
    src = IMP.readPicture('D:\\', "xxx.jpeg")
    channels = IMP.conversionChannels(src)
    IMP.showPicture(IMP.reversalRGB(IMP.conversionChannels(channels, False)))

    # IMP.showGrayscale(kk, np.array(['B', 'G', 'R']))

    bb = IMP.flip(src, IMP.DIAGONAL_MIRRORING)
    # IMP.showGrayscale(bb)
    cv2.imshow('src', bb)
    # cv2.imshow('src', IMP.conversionChannels(kk, False))

    rows = src.shape[0]
    cols = src.shape[1]
    # cv2.imshow('src', bb)

    img = Img(src, rows, cols, [340, 454])


    # img.Vertically()  # 镜像
    # img.Process()
    # img.Rotate(-math.radians(180)) #旋转
    # img.Process()

    # img.Zoom(0.5) #缩放
    # img.Process()

    # img.Move(-50,-50) #平移
    # img.Process()
    # 这个更快
    # img.dst = np.pad(src, ((0, 0), (300, 0)), mode='constant')[:, :-300]
    #
    # cv2.imshow('dst', kk[0])
    # IMP.showGrayscale(np.array([[kk[0]]]))
    # 旋转更快
    def rotate(image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated


    # cv2.imshow('dst', rotate(src, 40, scale=1.5))

    cv2.waitKey(0)


