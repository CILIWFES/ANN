import numpy as np
import math
import matplotlib.pyplot as plt


class ImageProcessing:
    LEFT = 0
    RIGHT = 1
    UPON = 2
    DOWN = 3

    # 1.传入 [图片数, 通道, 高, 宽]
    # 2.传入 [ 通道, 高, 宽]
    def analyse(imgs: np.ndarray):
        i = 0
        size = imgs.shape
        if len(size) == 4:
            i = 1
        elif len(size) == 3:
            i = 0
        else:
            raise Exception("未知参数 尺寸:", size)
        # 通道数,高,宽
        return imgs.shape[i], imgs.shape[i + 1], imgs.shape[i + 2]

    def showGrayscale(self, imgs: np.ndarray, labels: np.ndarray = None):
        # imgs [图片数, 通道, 高, 宽]
        img_size = imgs.shape[0]
        if labels is not None and labels.shape[0] != img_size:
            raise Exception("输入图片与标签不符合")

        channels, high, width = self.analyse(imgs)
        if channels != 1:
            raise Exception("灰度图像通道数不能大于1")

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
            plt.imshow(imgs[i][0], cmap="Greys_r")
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

    # 平移moveLength个像素
    # direction:方向
    # imgs [图片数, 通道, 高, 宽]
    def move(self, img: np.ndarray, channels, high, width, moveX=0, moveY=0):
        img[i][0] = np.pad(aug_img[i][0], ((0, 0), (amt, 0)), mode='constant')[:, :-amt]

