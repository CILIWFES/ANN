import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


class ImageProcessing:
    HORIZONTAL_MIRRORING = 1
    VERTICAL_MIRRORING = 0
    DIAGONAL_MIRRORING = -1

    def readPicture(self, path, fileName):
        src = cv2.imread(path + fileName, 1)
        return src

    def readChannels(self, path, fileName):
        src = cv2.imread(path + fileName, 1)
        src = self.conversionChannels(src)
        return src

    def readGrayscale(self, path, fileName):
        src = cv2.imread(path + fileName, 0)
        return src

    # 通过通道数转化,通道前变为通道后
    def conversionChannels(self, imgs, toChannelFirst=True):
        if len(imgs) == 4:
            if toChannelFirst:
                return np.array([cv2.split(item) for item in imgs])
            else:
                return np.array([cv2.merge(item) for item in imgs])

        elif len(imgs.shape) == 3:
            if toChannelFirst:
                return np.array(cv2.split(imgs))
            else:
                return cv2.merge(imgs)

    def reversalRGB(self, imgs):
        # Opencv是 BGR ,变为RGB格式
        if len(imgs.shape) == 3:
            img_size = 1
            showImgs = np.array([imgs])
        elif len(imgs.shape) == 4:
            img_size = imgs.shape[0]
            showImgs = imgs
        else:
            raise Exception("错误传入类型")

        for item in showImgs:
            for picture in item:
                for channels in picture:
                    channels[0], channels[2] = channels[2], channels[0]
        return showImgs

    def showPicture(self, imgs, labels: np.ndarray = None):
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

    # delta_x 宽(右+ 左-)
    # delta_y 高(上- 下+)
    # imgs [ 宽,高]
    # imgs [ 宽,高,通道数]
    def move(self, image, delta_x=0, delta_y=0, mode='constant'):  # 平移
        left = 0
        right = 0
        upon = 0
        low = 0
        width = image.shape[1]
        high = image.shape[0]
        if delta_x < 0:
            left = -delta_x
        else:
            right = delta_x
        if delta_y < 0:
            upon = -delta_y
        else:
            low = delta_y

        changeTuple = ((low, upon), (right, left), (0, 0)) if len(image.shape) == 3 else ((low, upon), (right, left))
        changeImge = np.pad(image, changeTuple, mode=mode)[upon:high + upon,
                     left:width + left]
        return changeImge

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def translate(self, image, x, y):
        (h, w) = image.shape[:2]
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (w, h))
        return shifted

    def flip(self, image, mode):
        # python 图像翻转,使用openCV flip()方法翻转
        xImg = cv2.flip(image, mode, dst=None)
        return xImg

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
