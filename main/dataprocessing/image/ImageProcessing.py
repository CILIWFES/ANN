import numpy as np
import cv2
import math
import random
import matplotlib.pyplot as plt
from main.analysis.performanceMeasure import *


class ImageProcessing:
    HORIZONTAL_MIRRORING = 1
    VERTICAL_MIRRORING = 0
    DIAGONAL_MIRRORING = -1

    # 读取图片,返回图片矩阵(1个像素3通道)
    def readPicture(self, path, fileName):
        src = cv2.imread(path + fileName, 1)
        src = self.reversalRGB(src, cv2.COLOR_BGR2RGB)
        return src

    # 读取图片,返回图片矩阵(1个像素1通道)
    def readChannels(self, path, fileName):
        src = cv2.imread(path + fileName, 1)
        src = self.reversalRGB(src, cv2.COLOR_BGR2RGB)
        src = self.conversionChannels(src)
        return src

    # 读取灰度图片
    def readGrayscale(self, path, fileName):
        src = cv2.imread(path + fileName, 0)
        return src

    # 通过通道数转化,通道前变为通道后
    def conversionChannels(self, imgs, toChannelFirst=True):
        if len(imgs.shape) == 4:
            if toChannelFirst:
                return np.array([cv2.split(item) for item in imgs])
            else:
                return np.array([cv2.merge(item) for item in imgs])

        elif len(imgs.shape) == 3:
            if toChannelFirst:
                return np.array(cv2.split(imgs))
            else:
                return cv2.merge(imgs)

    # openCv 是BGR 这里与RGB相互转化(其实只是R与G互换,执行两遍换回来)
    def reversalRGB(self, imgs, mode=cv2.COLOR_RGB2BGR):
        # Opencv是 BGR ,变为RGB格式
        showImgs = cv2.cvtColor(imgs, mode)
        return showImgs

    # imgs [图片数, 高, 宽]
    # imgs [高, 宽]
    def showPicture_RBG(self, imgs, labels: np.ndarray = None):
        if len(imgs.shape) == 3:
            img_size = 1
            showImgs = [imgs]
        elif len(imgs.shape) == 4:
            img_size = imgs.shape[0]
            showImgs = imgs
        else:
            raise Exception("错误传入类型")

        if labels is not None and labels.shape[0] != img_size:
            raise Exception("输入图片与标签不符合")
        matrix_size = math.ceil(pow(img_size, 0.5))
        self._showPyplot(showImgs, img_size, matrix_size, labels=labels)

    # imgs [图片数, 高, 宽]
    # imgs [高, 宽]
    def showGrayscale(self, imgs: np.ndarray, labels: np.ndarray = None):
        if len(imgs.shape) == 2:
            img_size = 1
            showImgs = [imgs]
        elif len(imgs.shape) == 3:
            img_size = imgs.shape[0]
            showImgs = imgs
        else:
            raise Exception("错误传入类型")

        if labels is not None and labels.shape[0] != img_size:
            raise Exception("输入图片与标签不符合")
        matrix_size = math.ceil(pow(img_size, 0.5))

        self._showPyplot(showImgs, img_size, matrix_size, labels=labels, cmap='Greys_r')

    # IMP.showPicture_BRG(IMP.reversalRGB(picture), wait=True)
    def showPicture_BRG(self, image, name=None, wait=False):
        if name is None:
            name = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba', 8))
        cv2.imshow(name, image)
        if wait is True:
            cv2.waitKey(0)

    # 旋转
    def rotate(self, image, angle, center=None, scale=1.0, channelPicture=False):

        if channelPicture:
            (c, h, w) = image.shape[:3]
        else:
            (h, w) = image.shape[:2]
            c = 1

        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        if c == 1:
            rotated = cv2.warpAffine(image, M, (w, h))
        else:
            rotated = np.array([cv2.warpAffine(item, M, (w, h)) for item in image])

        return rotated

    # 平移
    def translate(self, image, x, y, channelPicture=False):
        M = np.float32([[1, 0, x], [0, 1, y]])

        if channelPicture:
            (h, w) = image.shape[1:3]
            shifted = np.array([cv2.warpAffine(item, M, (w, h)) for item in image])
        else:
            (h, w) = image.shape[:2]
            shifted = cv2.warpAffine(image, M, (w, h))

        return shifted

    # 对称
    def flip(self, image, mode, channelPicture=False):
        # python 图像翻转,使用openCV flip()方法翻转
        # mode=0 垂直 1 水平 -1对角线
        if channelPicture:
            xImg = np.array([cv2.flip(item, mode, dst=None) for item in image])
        else:
            xImg = cv2.flip(image, mode, dst=None)

        return xImg

    # 缩放
    # INTER_NEAREST最近邻插值
    # INTER_LINEAR 线性插值
    # CV_INTER_AREA：区域插值
    # INTER_CUBIC 三次样条插值
    # INTER_LANCZOS4 Lanczos插值
    def resize(self, picture, toWidth=0, toHigh=0, mode=cv2.INTER_LINEAR, isChannelPicture=False):
        toWidth = math.ceil(toWidth)
        toHigh = math.ceil(toHigh)

        if isChannelPicture:
            changePicture = cv2.resize(picture, (toWidth, toHigh), mode)
        else:
            changePicture = np.array([cv2.resize(item, (toWidth, toHigh), mode) for item in picture])

        return changePicture

    # 裁剪图片 point:(x,y)  width:宽度 high:高度
    # IMP.cutOut(channels, (120, 200), 100, 200)
    def cutOut(self, image, point, width, high, channelPicture=False):
        picture = image.copy()
        if channelPicture:
            picture[point[0]:point[0] + width, point[1]:point[1] + high] = 0
        else:
            picture[:, point[0]:point[0] + width, point[1]:point[1] + high] = 0
        return picture

    def _showPyplot(self, showImgs, img_size, matrix_size, cmap=None, labels=None):
        fig = plt.figure()
        # 向上转型
        for i in range(img_size):
            # 添加图层
            ax = fig.add_subplot(matrix_size, matrix_size, i + 1)
            # 设置标签
            if labels is not None:
                ax.set_title(labels[i])
            # 图像展示
            if cmap is None:
                plt.imshow(showImgs[i])
            else:
                plt.imshow(showImgs[i], cmap=cmap)

            # 坐标轴关闭
            plt.axis("off")
        # 图层展示
        plt.show()
