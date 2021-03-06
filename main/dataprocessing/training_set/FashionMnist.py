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
