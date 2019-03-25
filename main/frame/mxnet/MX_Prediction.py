import numpy as np
import mxnet as mx


class MX_Prediction:
    # 传入numpy矩阵类型,预测数据
    def prediction(self, predictiveValues, module):
        # 转化为标准矩阵
        # val_iter = mx.io.NDArrayIter(predictiveValues)
        return module.predict(predictiveValues).asnumpy()
