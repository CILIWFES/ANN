from main.configuration import *
from main.dataprocessing.orm import *
from main.dataprocessing.image import *
import numpy as np
import math
import pickle


class Cifar:

    def toConversion(self):
        filePath = Cifar._train_root_path
        fileName = Cifar._train_data_name
        fileRange = list(Cifar._train_data_range)
        (batchSize, tags, pictureInfo) = self.__readInfo(Cifar._info_path)
        tags = {i: item for i, item in enumerate(tags)}
        for item in fileRange:
            data = self.__conversion(filePath + fileName + item, pictureInfo)
            data[Cifar.data] = self.toImage(data[Cifar.data], batchSize, pictureInfo)
            self.toSave(data, filePath + Cifar.conversion_picture_path, tags)

    def toSave(self, data, path, tags):
        labels = data[Cifar.labels]
        filenames = data[Cifar.filenames]
        datas = data[Cifar.data]
        # 转化通道
        datas = IMP.conversionChannels(datas, toChannelFirst=False)
        for i in range(data[Cifar.data].shape[0]):
            ORM.savePicture(path + str(labels[i]) + '\\', tags[labels[i]] + "_" + filenames[i],
                            IMP.reversalRGB(datas[i]))

    # 转化为RBG,多通道标准
    def toImage(self, data: np.ndarray, batchSize, pictureInfo):
        (h, w, c) = pictureInfo
        data = data.reshape((batchSize, c, h, w))
        return data

    def __conversion(self, path, pictureInfo):
        with open(path, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    def __readInfo(self, path):
        with open(path, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')

        return data[Cifar.num_cases_per_batch], data[Cifar.label_names], (
            math.ceil(math.pow(data[Cifar.num_vis] // 3, 0.5)), math.ceil(math.pow(data[Cifar.num_vis] // 3, 0.5)), 3)

    # section配置
    _section = GLOCT.TRAINING_SECTION

    # 训练集,测试集 的标签/数据路径
    _train_root_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                                    GLOCT.TRAINING_CIFAR_PATH])

    _train_data_name = GLOCF.getConfig(_section, GLOCT.TRAINING_CIFAR_TRAINDATA)

    _train_data_range = GLOCF.getConfig(_section, GLOCT.TRAINING_CIFAR_TRAINRANGE)

    _test_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                              GLOCT.TRAINING_CIFAR_PATH,
                                              GLOCT.TRAINING_CIFAR_TESTDATA])

    _info_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                              GLOCT.TRAINING_CIFAR_PATH,
                                              GLOCT.TRAINING_CIFAR_INFO])
    # 基本信息
    batch_label = 'batch_label'
    # 标签
    labels = 'labels'
    # 数据
    data = 'data'
    # 文件形容
    filenames = 'filenames'
    # 数量
    num_cases_per_batch = 'num_cases_per_batch'
    # 一张图片的尺寸
    num_vis = 'num_vis'
    # 标签名称
    label_names = 'label_names'
    conversion_picture_path = 'picture\\'
