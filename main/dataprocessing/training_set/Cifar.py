from main.configuration import *
import numpy as np
import math
import pickle


class Cifar:
    # section配置
    _section = GLOCT.TRAINING_SECTION

    # 训练集,测试集 的标签/数据路径
    _train_root_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                                    GLOCT.TRAINING_CIFAR_PATH])

    _train_data_name = GLOCF.getFilsPath(_section, GLOCT.TRAINING_CIFAR_TRAINDATA)

    _train_data_range = GLOCF.getFilsPath(_section, GLOCT.TRAINING_CIFAR_TRAINRANGE)

    _test_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                              GLOCT.TRAINING_CIFAR_PATH,
                                              GLOCT.TRAINING_CIFAR_TESTDATA])

    _info_path = GLOCF.getFilsPath(_section, [GLOCT.COMMON_CONFIG_FOLDER,
                                              GLOCT.TRAINING_CIFAR_PATH,
                                              GLOCT.TRAINING_CIFAR_INFO])
    numberOfCategories = 10
    # 基本信息
    batch_label = 'batch_label'
    # 标签
    labels = 'labels'
    # 数据
    data = 'data'
    # 文件形容
    filenames = 'filenames'

    def toConversion(self):
        filePath = Cifar._train_root_path
        fileName = list(Cifar._train_data_range)
        self.__readInfo(Cifar._info_path)
        for item in fileName:
            self.__conversion(filePath + item)

    def __conversion(self, path):
        with open(path, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')

    def __readInfo(self, path):
        with open(path, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
