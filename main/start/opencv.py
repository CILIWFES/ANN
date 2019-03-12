import cv2
import math
import numpy as np


class Img:
    def __init__(self, image, rows, cols, center=[0, 0]):
        self.src = image  # 原始图像
        self.rows = rows  # 原始图像的行
        self.cols = cols  # 原始图像的列
        self.center = center  # 旋转中心，默认是[0,0]

    def Move(self, delta_x, delta_y):  # 平移
        # delta_x>0左移，delta_x<0右移
        # delta_y>0上移，delta_y<0下移
        self.transform = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])

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
    src = cv2.imread('D:\\xxx.jpg', 0)
    rows = src.shape[0]
    cols = src.shape[1]
    cv2.imshow('src', src)

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
    # cv2.imshow('dst', img.dst)
    # 旋转更快
    def rotate(image, angle, center=None, scale=1.0):  # 1
        (h, w) = image.shape[:2]  # 2
        if center is None:  # 3
            center = (w // 2, h // 2)  # 4

        M = cv2.getRotationMatrix2D(center, angle, scale)  # 5

        rotated = cv2.warpAffine(image, M, (w, h))  # 6
        return rotated  # 7


    cv2.imshow('dst', rotate(src, 40,scale=2))

    cv2.waitKey(0)
