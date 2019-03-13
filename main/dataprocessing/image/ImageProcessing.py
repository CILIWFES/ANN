import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


class ImageProcessing:
    LEFT = 0
    RIGHT = 1
    UPON = 2
    DOWN = 3

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

    # 1.传入 [图片数, 通道, 高, 宽]
    # 2.传入 [ 通道, 高, 宽]
    # 2.传入(灰度channelFirst=None) [ 图片数, 高, 宽]
    def analyse(self, imgs: np.ndarray, channelFirst=True):
        size = imgs.shape
        if len(size) == 4:
            i = 1
        elif len(size) == 3:
            i = 0
        else:
            raise Exception("未知参数 尺寸:", size)
        # 通道数,高,宽
        if channelFirst:
            return 1 if channelFirst is None else imgs.shape[i], imgs.shape[i + 1], imgs.shape[i + 2]
        else:
            return 1 if channelFirst is None else imgs.shape[i], imgs.shape[i], imgs.shape[i + 1]

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

        fig = plt.figure()
        # 向上转型
        matrix_size = math.ceil(pow(img_size, 0.5))
        for i in range(img_size):
            # 添加图层
            ax = fig.add_subplot(matrix_size, matrix_size, i + 1)
            # 设置标签
            if labels is not None:
                ax.set_title(labels[i])
            # 图像展示
            plt.imshow(showImgs[i], cmap="Greys_r")
            # 坐标轴关闭
            plt.axis("off")
        # 图层展示
        plt.show()

    def changeImage(self, imgs: np.ndarray):
        channels, high, width = self.analyse(imgs)

        for i in range(imgs.shape[0]):
            # 有50%概率做左右翻转
            if np.random.random() < 0.5:
                # fliplr()用于左右翻转
                for j in range(channels):
                    imgs[i][j] = np.fliplr(imgs[i][j])
            # 左右移动最多2个像素，注意randint(a,b)的范围为a到b-1
            amt = np.random.randint(0, 3)
            if amt > 0:  # 如果需要移动…
                if np.random.random() < 0.5:  # 左移动还是右移动？
                    # pad()用于加上外衬，因移动后减少的区域需补零
                    # 然后用[:]取所要的部分
                    imgs[i][0] = np.pad(imgs[i][0], ((0, 0), (amt, 0)), mode='constant')[:, :-amt]
                else:
                    imgs[i][0] = np.pad(imgs[i][0], ((0, 0), (0, amt)), mode='constant')[:, amt:]

            # 上下移动最多2个像素
            amt = np.random.randint(0, 3)
            if amt > 0:
                if np.random.random() < 0.5:
                    imgs[i][0] = np.pad(imgs[i][0], ((amt, 0), (0, 0)), mode='constant')[:-amt, :]
                else:
                    imgs[i][0] = np.pad(imgs[i][0], ((0, amt), (0, 0)), mode='constant')[amt:, :]

            # 随机清零最大5*5的区域
            x_size = np.random.randint(1, 6)
            y_size = np.random.randint(1, 6)
            x_begin = np.random.randint(0, 28 - x_size + 1)
            y_begin = np.random.randint(0, 28 - y_size + 1)
            imgs[i][0][x_begin:x_begin + x_size, y_begin:y_begin + y_size] = 0

    # delta_x,delta_y
    # imgs [ 宽(右+ 左-),高(上- 下+)]
    def move(self, image, delta_x=0, delta_y=0):  # 平移
        left = 0
        right = 0
        upon = 0
        low = 0
        width = image.shape[-1]
        high = image.shape[0]
        if delta_x < 0:
            left = -delta_x
        else:
            right = delta_x
        if delta_y < 0:
            upon = -delta_y
        else:
            low = delta_y

        changeImge = np.pad(image, ((low, upon), (right, left)), mode='constant')[upon:high + upon, left:width + left]
        return changeImge
