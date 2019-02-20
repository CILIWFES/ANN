import os


class GlobalConstant:
    # 工程路径,三级目录
    SYS_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + "\\"
    SYS_FILES_PATH = SYS_ROOT_PATH + "files\\"

    SYS_CONFIG_FOLDER = "\\configuration\\"
    # 全局配置文件名
    SYS_GLO_CONFIG_FILENAME = "configuration.ini"
    # 全局配置文件路径
    SYS_GLO_CONFIG_PATH = SYS_FILES_PATH + SYS_CONFIG_FOLDER + SYS_GLO_CONFIG_FILENAME

    # TRAINING 训练集路径
    SYS_TRAINING_PACKAGE_PATH = SYS_FILES_PATH + "training-package\\"
    # FASHION_MNIST 标准测试集路径
    SYS_FASHION_MNIST_PATH = SYS_TRAINING_PACKAGE_PATH + "fashion-mnist\\"

    """
    配置文件中名字的统一命名
    Section:大类
    Name:大类下的项
    """

    # 通用项目
    # 文件夹路径
    COMMON_CONFIG_FOLDER = "Folder"
    COMMON_CONFIG_FILENAME = "FileName"
    COMMON_CONFIG_BUNCHNAME = "BunchName"

    """
    Section:大类
    """
    # ORM框架配置文件section
    ORM_CONFIG_SECTION = "ORM"

    """
    SEGMENTATION大类(分隔符)
    """
    SEGMENTATION_CONFIG_SECTION = "Segmentation"
    # 普通分隔符
    SEGMENTATION_COMMON = "common"

    """
    TRAINING:大类
    """
    TRAINING_FASHION_MNIST_TRAINING_DATA = "FASHION_MNIST_TrainingData"
    TRAINING_FASHION_MNIST_TEST_DATA = "FASHION_MNIST_TestData"
    TRAINING_FASHION_MNIST_TRAINING_LABEL = "FASHION_MNIST_TrainingLabel"
    TRAINING_FASHION_MNIST_TEST_LABEL = "FASHION_MNIST_TestLabel"
