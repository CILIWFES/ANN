from main.configuration import *
import numpy as np
import math
import gzip
import struct
import matplotlib.pyplot as plt


class FashionMnist:
    # section配置
    _section = GLOCT.TRAINING_SECTION

    # 训练集,测试集 的标签/数据路径
    _train_label_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                                     GLOCT.TRAINING_FASHION_MNIST_PATH,
                                                     GLOCT.TRAINING_FASHION_MNIST_TRAINING_LABEL])

    _train_img_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                                   GLOCT.TRAINING_FASHION_MNIST_PATH,
                                                   GLOCT.TRAINING_FASHION_MNIST_TRAINING_DATA])

    _test_label_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                                    GLOCT.TRAINING_FASHION_MNIST_PATH,
                                                    GLOCT.TRAINING_FASHION_MNIST_TEST_LABEL])

    _test_img_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                                  GLOCT.TRAINING_FASHION_MNIST_PATH,
                                                  GLOCT.TRAINING_FASHION_MNIST_TEST_DATA])

    def read_TrainImg(self):
        return self._read_Img(self._train_img_path, self._train_label_path)

    def read_TestImg(self):
        return self._read_Img(self._test_img_path, self._test_label_path)

    def _read_Img(self, image_url, label_url):
        with gzip.open(label_url) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.int8)
        with gzip.open(image_url, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8)
            # 图片 x[通道 x(长x宽)]
            image = image.reshape(len(label), 1, rows, cols)
            # 归一化
            image = image.astype(np.float32) / 255.0
        return (image, label, rows, cols)

    def change(self, image_url, label_url):
        return ''
    def changeOne(self, img):
        # 有50%概率做左右翻转
        if np.random.random() < 0.5:
            # img[0]为第i号样本的0号通道，灰度图像只有0号通道
            # fliplr()用于左右翻转
            img[0] = np.fliplr(img[0])

        # 左右移动最多2个像素，注意randint(a,b)的范围为a到b-1
        amt = np.random.randint(0, 3)
        if amt > 0:  # 如果需要移动…
            if np.random.random() < 0.5:  # 左移动还是右移动？
                # pad()用于加上外衬，因移动后减少的区域需补零
                # 然后用[:]取所要的部分
                img[0] = np.pad(img[0], ((0, 0), (amt, 0)), mode='constant')[:, :-amt]
            else:
                img[0] = np.pad(img[0], ((0, 0), (0, amt)), mode='constant')[:, amt:]

        # 上下移动最多2个像素
        amt = np.random.randint(0, 3)
        if amt > 0:
            if np.random.random() < 0.5:
                img[0] = np.pad(img[0], ((amt, 0), (0, 0)), mode='constant')[:-amt, :]
            else:
                img[0] = np.pad(img[0], ((0, amt), (0, 0)), mode='constant')[amt:, :]

        # 随机清零最大5*5的区域
        x_size = np.random.randint(1, 6)
        y_size = np.random.randint(1, 6)
        x_begin = np.random.randint(0, 28 - x_size + 1)
        y_begin = np.random.randint(0, 28 - y_size + 1)
        img[0][x_begin:x_begin + x_size, y_begin:y_begin + y_size] = 0


    def showImg(self, imgs: np.ndarray, labels: np.ndarray = None):
        img_size = imgs.shape[0]

        if labels is not None and labels.shape[0] != img_size:
            raise Exception("输入图片与标签不符合")

        fig = plt.figure()
        matrix_size = math.ceil(pow(img_size, 0.5))
        for i in range(img_size):
            ax = fig.add_subplot(matrix_size, matrix_size, i + 1)
            if labels is not None:
                ax.set_title(labels[i])
            plt.imshow(imgs[i][0], cmap="Greys_r")
            # 坐标轴关闭
            plt.axis("off")
        plt.show()
